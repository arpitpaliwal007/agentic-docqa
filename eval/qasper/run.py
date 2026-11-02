#!/usr/bin/env python3
"""Run QASPER through unchanged production ingestion and UserRAG; --prepare-only needs no Ollama."""
import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.judge import make_judge, judge_failure
from eval.storage import Results, atomic_json, code_fingerprint, digest, prepare_workdir
from eval.run_eval import ollama_info, runtime_info
from eval.qasper.judge import QASPER_JUDGE_POLICY, judge_scientific
from eval.qasper.load import load_split, prepare_sample
from eval.qasper.metrics import answer_score, evidence_score, failure_categories, match_paragraphs, tool_hits
from eval.qasper.pipeline import QasperBackend
from eval.qasper.shorten import FORMAT_POLICY, format_answer
from eval.qasper.report import write_reports


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--split',choices=('train','validation','test'),default='test')
    p.add_argument('--n',type=int,default=250)
    p.add_argument('--minimum-per-type',type=int,default=20,help='Stratified sample floor; use 5 for a 50-question run')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--mode',choices=('per_paper','pooled'),default='per_paper')
    p.add_argument('--answer-model',default=os.environ.get('EVAL_LLM_MODEL','qwen2.5:7b'))
    p.add_argument('--judge-model')
    p.add_argument('--format-model')
    p.add_argument('--limit',type=int,help='First N manifest questions; use a separate --out for smoke')
    p.add_argument('--out',type=Path,help='Default: results/qasper/<mode>')
    p.add_argument('--cache-dir',type=Path,default=Path(os.environ.get('QASPER_CACHE_DIR',str(ROOT/'data/qasper'))))
    p.add_argument('--sample-dir',type=Path,default=ROOT/'eval/qasper')
    p.add_argument('--work-root',type=Path,default=Path(tempfile.gettempdir())/'agentic-docqa-qasper-work')
    p.add_argument('--prefix-ablation',action=argparse.BooleanOptionalAction,default=False)
    p.add_argument('--resume',action='store_true')
    p.add_argument('--prepare-only',action='store_true',help='Load/cache data and save sample manifest; no app/models')
    p.add_argument('--judge-context-chars',type=int,default=24000)
    p.add_argument('--report-every',type=int,default=10,help='JSON saves every question; refresh Markdown/CIs every N')
    p.add_argument('--drive-folder-id',help='Optional Drive API backup parent; every question is uploaded and read back before continuing')
    p.add_argument('--drive-account',help='Required Google account email for Drive API backup')
    args=p.parse_args(argv)
    args.out=args.out or ROOT/'results/qasper'/args.mode
    if args.n <= 0 or args.minimum_per_type < 0 or (args.limit is not None and args.limit <= 0) or args.judge_context_chars < 1000 or args.report_every <= 0:
        p.error('n/limit/report-every must be positive; judge context must be >=1000')
    if bool(args.drive_folder_id) != bool(args.drive_account):
        p.error('--drive-folder-id and --drive-account must be supplied together')
    return args


def process_question(session, paper, question, formatter, judge, context_chars):
    record={**question,'status':'complete','errors':[],'retrieval':{}}
    try:
        record['retrieval']=session.retrieval(paper,question)
    except Exception as error:
        record['errors'].append(f'Direct retrieval: {type(error).__name__}: {error}')
    record.update(session.ask(question['question']))  # EXACT question; backend uses new thread, empty history.
    if record.get('execution_error'):
        record['errors'].append(record['execution_error'])
    if 'answer' not in record:
        record.update(missing_prediction=True,answer_f1_short=0.,answer_f1_full=0.,evidence_f1=0.)
        for kind in ('rule','format'):
            record['unanswerable_'+kind]=0. if question['majority_type']=='none' else None
            record['wrong_refusal_'+kind]=0. if question['majority_type']!='none' else None
    else:
        formatted=format_answer(formatter,question['question'],record['answer'])
        record.update(formatting=formatted,short_answer=formatted['short_answer'],full_answer=formatted['full_answer'],
                      rule_refusal=formatted['rule_refusal'],format_refusal=formatted['format_refusal'],
                      format_refusal_disagreement=formatted['refusal_disagreement'],format_fallback=formatted['fallback'])
        short=answer_score(record['short_answer'],question['annotations'])
        full=answer_score(record['full_answer'],question['annotations'])
        record.update(answer_f1_short=short['f1'],answer_f1_full=full['f1'],best_type_short=short['answer_type'],
                      best_type_full=full['answer_type'],best_reference_index=short['reference_index'],
                      categorical_exact_match=short['categorical_exact_match'])
        predicted, mapping=match_paragraphs(paper,tool_hits(record['events']))
        record.update(predicted_evidence_ids=sorted(predicted),evidence_mapping=mapping,
                      evidence_f1=evidence_score(predicted,question['annotations']))
        majority_none=question['majority_type']=='none'
        for kind in ('rule','format'):
            refused=record[kind+'_refusal']
            record['unanswerable_'+kind]=float(refused) if majority_none else None
            record['wrong_refusal_'+kind]=float(refused) if not majority_none else None
        try:
            annotation=question['annotations'][short['reference_index']]
            judgment,attempts,latency=judge_scientific(judge,question['question'],annotation,record['answer'],record['evidence'],context_chars)
            record.update(judgment=judgment,judge_attempts=attempts,judge_latency_seconds=latency)
            if judgment is None:
                record['errors'].append(judge_failure(attempts))
            else:
                record.update(judge_correctness=judgment['correctness'],faithfulness=judgment['faithfulness'])
        except Exception as error:
            record['errors'].append(f'Judge: {type(error).__name__}: {error}')
    for event in record.get('events',[]):
        if event['kind'] in ('search','parent') and event.get('error'):
            record['errors'].append(f"Agent tool error: {event['error']}")
    record['status']='error' if record['errors'] else 'complete'
    record['failure_categories']=failure_categories(question,record)
    return record


def clean_work(work, fingerprint):
    marker=work/'.eval-owned.json'
    if not marker.is_file() or json.loads(marker.read_text()) != {'owner':'agentic-docqa-eval','fingerprint':fingerprint} or work.is_symlink():
        raise ValueError('Refusing cleanup of an unrecognized evaluation directory')
    shutil.rmtree(work)


def main(argv=None):
    args=parse_args(argv)
    papers,data_source=load_split(args.split,args.cache_dir)
    selected,manifest=prepare_sample(papers,data_source,args.n,args.seed,args.sample_dir,args.cache_dir,args.minimum_per_type)
    args.out.mkdir(parents=True,exist_ok=True)
    atomic_json(args.out/f'sample_{args.split}_{args.n}_{args.seed}.json',manifest)
    if args.prepare_only:
        atomic_json(args.out/'data-preparation.json',{'data_source':data_source,'sample':manifest,
                    'float_only_questions':sum(q['float_only'] for q in selected)})
        print(f'Prepared {len(selected)} questions from {len(set(q["paper_id"] for q in selected))} papers; no models loaded.')
        return 0
    # Apply configuration before any app import. No routing/prompt overrides.
    os.environ.update(LLM_MODEL=args.answer_model,LLM_TEMPERATURE='0',NOMIC_PREFIXES='true',QDRANT_URL='',LANGFUSE_ENABLED='false')
    import torch
    os.environ['RERANK_DEVICE']='cuda' if torch.cuda.is_available() else 'cpu'
    from app import config
    if config.RETRIEVAL_MODE!='hybrid' or config.RERANKER!='cross_encoder':
        raise ValueError('QASPER expects the existing hybrid/cross_encoder configuration; reset RETRIEVAL_MODE/RERANKER')
    judge_model=args.judge_model or args.answer_model
    format_model=args.format_model or args.answer_model
    runtime=runtime_info()
    try:
        runtime['packages']['datasets']=importlib.metadata.version('datasets')
    except importlib.metadata.PackageNotFoundError:
        runtime['packages']['datasets']='not installed (official JSON fallback)'
    models=ollama_info(list(dict.fromkeys([args.answer_model,judge_model,format_model,config.DENSE_MODEL])))
    settings={key:value for key,value in vars(config).items() if key.isupper() and key not in
              {'LLM_API_KEY','BASE_DIR','DATA_DIR','QDRANT_PATH','COOKIE_NAME','SESSION_TTL'} and isinstance(value,(str,int,float,bool,list))}
    identity={'benchmark':'QASPER','split':args.split,'n':args.n,'seed':args.seed,'mode':args.mode,
              'sample_hash':digest(manifest),'dataset_hash':data_source['dataset_hash'],'code_hash':code_fingerprint(ROOT),
              'answer_model':args.answer_model,'judge_model':judge_model,'format_model':format_model,
              'judge_policy':QASPER_JUDGE_POLICY,'format_policy':FORMAT_POLICY,'prefix_ablation':args.prefix_ablation,
              'judge_context_chars':args.judge_context_chars,'settings':settings,'runtime':runtime,'ollama':models}
    fingerprint=digest(identity)
    run_id='qasper-'+fingerprint[:16]
    if (args.out/(run_id+'.partial.json')).exists() and not args.resume:
        raise ValueError('A matching run exists. Use --resume; original results will not be overwritten.')
    now=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    weights=models['model_digests']
    same_weights=weights.get(args.answer_model) not in (None,'not pulled') and weights.get(args.answer_model)==weights.get(judge_model)
    metadata={**identity,'fingerprint':fingerprint,'run_id':run_id,'report_basename':now+'-'+run_id,
              'sampled_n':len(selected),'sample':manifest,'data_source':data_source,'self_judge':judge_model==args.answer_model or same_weights,
              'sample_float_only':sum(q['float_only'] for q in selected),'unmapped_evidence_count':sum(len(q['unmapped_evidence_ids']) for q in selected)}
    state=Results(args.out,run_id,metadata)
    backup=None
    if args.drive_folder_id:
        from eval.qasper.drive_backup import DriveBackup
        backup=DriveBackup.connect(args.drive_folder_id,args.drive_account)
        backup.bind(state,resume=args.resume)
    targets=selected[:args.limit] if args.limit else selected
    pending=[q for q in targets if not state.completed(q['id'])]
    if not pending:
        print('All selected questions are already complete; no inference repeated.')
        write_reports(state)
        if backup:
            backup.checkpoint(state)
        return 0
    work=prepare_workdir(args.work_root,run_id,fingerprint)
    config.DATA_DIR=work/'data'
    config.QDRANT_PATH=str(work/'qdrant_db')
    backend=None
    try:
        backend=QasperBackend(work,papers,args.mode,args.prefix_ablation)
        state.value['metadata']['reranker_device']=backend.reranker_device
        if config.RERANK_DEVICE=='cuda' and not backend.reranker_device.startswith('cuda'):
            raise RuntimeError('Requested CUDA reranking but cross-encoder is not on CUDA')
        formatter=make_judge(format_model,config.LLM_BASE_URL,config.LLM_API_KEY)
        judge=make_judge(judge_model,config.LLM_BASE_URL,config.LLM_API_KEY)
        groups=OrderedDict()
        for question in pending:
            groups.setdefault(question['paper_id'],[]).append(question)
        pool=sorted({q['paper_id'] for q in selected})
        for paper_id,questions in groups.items():
            session=backend.session(paper_id,pool)
            try:
                for question in questions:
                    print(f"QASPER {question['id']} ({paper_id}): {question['question']}",flush=True)
                    record=process_question(session,papers[paper_id],question,formatter,judge,args.judge_context_chars)
                    state.value.pop('summary',None)
                    state.put(record)
                    if backup:
                        backup.record(state,record)  # A failed cloud verification stops before another question.
                    refresh=len(state.value['results']) % args.report_every==0
                    write_reports(state,markdown=refresh)
                    if backup and refresh:
                        backup.checkpoint(state)
            finally:
                backend.finish_paper(paper_id)
        state.value['metadata']['indexed_papers_this_process']=backend.indexed_papers
        state.value.pop('last_error',None)
        write_reports(state)
        if backup:
            backup.checkpoint(state)
        return 1 if any(state.value['results'][q['id']]['status']=='error' for q in targets) else 0
    except BaseException as error:
        state.value['last_error']=f'{type(error).__name__}: {error}'
        state.save()
        write_reports(state)
        raise
    finally:
        try:
            if backend is not None:
                backend.close()
        finally:
            clean_work(work,fingerprint)


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f'QASPER evaluation failed: {type(error).__name__}: {error}',file=sys.stderr)
        raise SystemExit(1)
