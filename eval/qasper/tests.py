"""Handmade, offline QASPER tests. No benchmark data or model downloads."""
import contextlib
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from eval.qasper.load import canonical_paper, annotation_type, reference_text, prepare_sample, sample_questions, load_split
from eval.qasper.metrics import normalize, token_f1, set_f1, answer_score, match_paragraphs, evidence_score, retrieval_metrics
from eval.qasper.shorten import strip_sources, added_content_words
from eval.qasper.stats import mean_ci, paired_test


def raw_paper(pid='paper-a', kind='extractive', qid='question-a'):
    answer={'unanswerable':False,'yes_no':None,'extractive_spans':[], 'free_form_answer':'',
            'evidence':['Alpha beta gamma delta epsilon zeta eta theta iota kappa.'], 'highlighted_evidence':['Alpha beta']}
    if kind=='none': answer.update(unanswerable=True,evidence=[])
    elif kind=='boolean': answer['yes_no']=False
    elif kind=='extractive': answer['extractive_spans']=['Alpha beta']
    else: answer['free_form_answer']='The paper compares methods.'
    return {'id':pid,'title':'A handmade test paper','abstract':'A handmade abstract.',
            'full_text':[{'section_name':'Methods','paragraphs':['Alpha beta gamma delta epsilon zeta eta theta iota kappa.','Other unrelated findings.']}],
            'qas':[{'question_id':qid,'question':'Which method works?', 'answers':[{'answer':answer,'annotation_id':'unit-test'}]}]}


class QasperMetricTests(unittest.TestCase):
    def setUp(self):
        self.paper=canonical_paper(raw_paper())
        self.q=self.paper['questions'][0]

    def test_squad_normalization_and_hand_computed_f1(self):
        self.assertEqual(normalize('The, QUICK! an (Answer).'),'quick answer')
        self.assertAlmostEqual(token_f1('alpha alpha beta','alpha beta beta'),2/3)
        self.assertEqual(token_f1('',''),0)
        self.assertEqual(token_f1('No','Yes'),0)
        self.assertAlmostEqual(token_f1('Yes indeed','Yes'),2/3)  # Released script, not categorical EM.
        annotations=[{'reference':'wrong','type':'abstractive'},{'reference':'alpha beta','type':'extractive'}]
        self.assertEqual(answer_score('Alpha beta',annotations)['f1'],1)
        self.assertEqual(answer_score('Alpha beta',annotations)['reference_index'],1)
        self.assertEqual(answer_score('Yes indeed',[{'reference':'Yes','type':'boolean'}])['categorical_exact_match'],0)

    def test_official_script_parity_when_available(self):
        path=Path(os.environ.get('QASPER_OFFICIAL_EVALUATOR','/private/tmp/agentic-docqa-qasper-evaluator.py'))
        if not path.is_file():
            self.skipTest('Set QASPER_OFFICIAL_EVALUATOR to the pinned official evaluator; see README')
        spec=importlib.util.spec_from_file_location('official_qasper',path)
        official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
        for prediction,reference in [('The cat','a cat'),('Yes indeed','Yes'),('Unanswerable','no'),('x x y','x y y'),('','')]:
            self.assertEqual(normalize(prediction),official.normalize_answer(prediction))
            self.assertAlmostEqual(token_f1(prediction,reference),official.token_f1_score(prediction,reference))
        for predicted,gold in [([],[]),(['one'],[]),(['one','two'],['two','three']),(['one'],['one'])]:
            self.assertEqual(set_f1(predicted,gold),official.paragraph_f1_score(predicted,gold))
        gold={'q':[{'answer':'wrong','evidence':['one'],'type':'abstractive'}, {'answer':'alpha beta','evidence':['two'],'type':'extractive'}]}
        predicted={'q':{'answer':'alpha','evidence':['two']}}
        result=official.evaluate(gold,predicted)
        ours=answer_score('alpha',[{'reference':r['answer'],'type':r['type']} for r in gold['q']])
        self.assertEqual(ours['f1'],result['Answer F1'])
        self.assertEqual(max(set_f1(['two'],r['evidence']) for r in gold['q']),result['Evidence F1'])

    def test_annotation_types_and_references(self):
        for kind in ('extractive','abstractive','boolean','none'):
            answer=raw_paper(kind=kind)['qas'][0]['answers'][0]['answer']
            self.assertEqual(annotation_type(answer),kind)
        self.assertEqual(reference_text({'yes_no':False}), 'No')
        self.assertEqual(reference_text({'yes_no':True}), 'Yes')
        self.assertEqual(reference_text({'unanswerable':True}), 'Unanswerable')
        self.assertEqual(reference_text({'extractive_spans':['one','two']}),'one two')

    def test_markdown_mapping_and_hf_column_sequences(self):
        self.assertTrue(self.paper['markdown'].startswith('# A handmade test paper\n\n## Abstract'))
        self.assertIn('\n\n## Methods\n\n',self.paper['markdown'])
        self.assertIn('Alpha beta gamma delta epsilon zeta eta theta iota kappa.',self.paper['paragraphs'].values())
        raw=raw_paper();qa=raw['qas'][0]
        raw['full_text']={'section_name':['Methods'],'paragraphs':[['Alpha beta gamma delta epsilon zeta eta theta iota kappa.','Other unrelated findings.']]}
        raw['qas']={key:[value] for key,value in qa.items()}
        raw['qas']['answers']=[{'answer':[qa['answers'][0]['answer']],'annotation_id':['unit-test']}]
        self.assertEqual(canonical_paper(raw),self.paper)

    def test_evidence_exact_split_overlap_threshold_and_wrong_paper(self):
        paragraph=next(text for text in self.paper['paragraphs'].values() if text.startswith('Alpha beta'))
        pid=self.q['annotations'][0]['evidence_ids'][0]
        hit=lambda text,source=None:{'source':source or self.paper['source'],'content':text}
        self.assertIn(pid,match_paragraphs(self.paper,[hit(paragraph)])[0])
        self.assertIn(pid,match_paragraphs(self.paper,[hit('Alpha beta gamma delta epsilon'),hit('zeta eta theta iota kappa.')])[0])
        self.assertNotIn(pid,match_paragraphs(self.paper,[hit('Alpha beta gamma delta epsilon')]*2)[0])
        self.assertNotIn(pid,match_paragraphs(self.paper,[hit('Alpha beta gamma delta epsilon zeta eta theta')])[0])
        self.assertIn(pid,match_paragraphs(self.paper,[hit('Alpha beta gamma delta epsilon zeta eta theta iota')])[0])
        self.assertNotIn(pid,match_paragraphs(self.paper,[hit(paragraph,'another-paper.md')])[0])
        self.assertEqual(evidence_score([pid],self.q['annotations']),1)
        self.assertEqual(set_f1([],[]),1)
        self.assertEqual(set_f1([pid],[]),0)
        self.assertEqual(retrieval_metrics(self.paper,self.q,[hit('irrelevant'),hit(paragraph)])['mrr_at_5'],.5)

    def test_float_evidence_and_unmapped_gold_are_not_silently_discarded(self):
        raw=raw_paper();answer=raw['qas'][0]['answers'][0]['answer']
        answer['evidence']=['FLOAT SELECTED: figure 1']
        paper=canonical_paper(raw);q=paper['questions'][0]
        self.assertTrue(q['float_only']);self.assertEqual(q['float_evidence_count'],1)
        self.assertEqual(q['annotations'][0]['evidence_ids'],[])
        self.assertIsNone(retrieval_metrics(paper,q,[])['hit_at_5'])
        answer['evidence']=['A genuinely absent paragraph.','FLOAT SELECTED: table 2']
        q=canonical_paper(raw)['questions'][0]
        self.assertFalse(q['float_only'])
        self.assertEqual(len(q['unmapped_evidence_ids']),1)
        self.assertEqual(evidence_score([],q['annotations']),0)

    def test_short_answer_guard_and_source_stripping(self):
        full='The method is Alpha beta.\n\n**Sources:**\n- paper.md'
        self.assertEqual(strip_sources(full),'The method is Alpha beta.')
        self.assertEqual(added_content_words('Alpha beta','The method is Alpha beta.'),[])
        self.assertEqual(added_content_words('Alpha invented','The method is Alpha beta.'),['invented'])
        self.assertEqual(added_content_words('Unanswerable','No information.'),[])
        self.assertIn('not',added_content_words('not Alpha','Alpha'))

    def test_sampler_determinism_minimums_and_manifests(self):
        papers={}
        for kind,count in [('extractive',70),('abstractive',20),('boolean',7),('none',3)]:
            for i in range(count):
                paper=canonical_paper(raw_paper(f'{kind}-{i}',kind,f'{kind}-q{i}'));papers[paper['id']]=paper
        first,detail=sample_questions(papers,n=60,seed=42)
        second,_=sample_questions(dict(reversed(list(papers.items()))),n=60,seed=42)
        self.assertEqual([q['id'] for q in first],[q['id'] for q in second])
        self.assertEqual(detail['quotas'],{'extractive':30,'abstractive':20,'boolean':7,'none':3})
        self.assertEqual(set(detail['rare_types_take_all']),{'boolean','none'})
        with self.assertRaises(ValueError):sample_questions(papers,n=3)
        fifty,detail50=sample_questions(papers,n=50,seed=42,minimum=5)
        self.assertEqual(len(fifty),50)
        self.assertEqual(detail50['minimum_per_type'],5)
        self.assertTrue(all(detail50['quotas'][kind]>=min(5,detail50['population'][kind]) for kind in detail50['quotas']))
        with self.assertRaises(ValueError):sample_questions(papers,n=50,minimum=-1)
        with tempfile.TemporaryDirectory() as d:
            a,manifest=prepare_sample(papers,{'split':'test','dataset_hash':'stable'},60,42,Path(d)/'samples',Path(d)/'cache')
            b,again=prepare_sample(papers,{'split':'test','dataset_hash':'stable'},60,42,Path(d)/'samples',Path(d)/'cache')
            self.assertEqual(manifest,again)
            self.assertEqual(len(a),len(b))
            self.assertTrue((Path(d)/'cache/samples/sample_test_60_42.json').exists())

    def test_official_json_fallback_and_cache_without_network(self):
        content=json.dumps({'paper-a':raw_paper()}).encode()
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            info=tarfile.TarInfo('../qasper-test-v0.3.json');info.size=len(content)
            archive.addfile(info,io.BytesIO(content))
        with tempfile.TemporaryDirectory() as directory, patch('urllib.request.urlopen',return_value=io.BytesIO(stream.getvalue())) as network:
            def unavailable(*a,**k):raise ImportError('datasets absent')
            papers,metadata=load_split('test',directory,hf_loader=unavailable)
            self.assertEqual(metadata['provider'],'official-release')
            self.assertEqual(papers['paper-a']['title'],'A handmade test paper')
            network.assert_called_once()
            self.assertEqual(load_split('test',directory,hf_loader=unavailable)[1],metadata)
            network.assert_called_once()
            self.assertFalse((Path(directory).parent/'qasper-test-v0.3.json').exists())

    def test_confidence_intervals_and_paired_statistics(self):
        ci=mean_ci([0]*10,binary=True)
        self.assertAlmostEqual(ci['high'],.2775328,places=5)
        self.assertEqual(mean_ci([],binary=True)['n'],0)
        self.assertEqual(mean_ci([.2,.5,.9],seed=4,repetitions=100),mean_ci([.2,.5,.9],seed=4,repetitions=100))
        test=paired_test([0]*8,[1]*8,repetitions=100)
        self.assertAlmostEqual(test['p_value'],.0078125)
        self.assertTrue(test['significant_0_05'])
        self.assertEqual(paired_test([1,0],[1,0],repetitions=100)['p_value'],1)
        with self.assertRaises(ValueError):paired_test([.5],[1])


HAVE_MODELS=importlib.util.find_spec('langchain_core') is not None


@unittest.skipUnless(HAVE_MODELS,'install project dependencies for model-interface tests')
class QasperPipelineTests(unittest.TestCase):
    def model(self,content):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        return RunnableLambda(lambda messages,**kwargs:AIMessage(content))

    def test_formatter_cannot_add_facts_and_refusal_disagreements_are_audited(self):
        from eval.qasper.shorten import format_answer
        answer='The method is Alpha.\nSources: p.md'
        result=format_answer(self.model('{"short_answer":"invented result"}'),'Which method?',answer)
        self.assertTrue(result['fallback']);self.assertEqual(result['short_answer'],'The method is Alpha.')
        self.assertEqual(result['added_words'],['invented','result'])
        result=format_answer(self.model('{"short_answer":"Unanswerable"}'),'Which method?',answer)
        self.assertTrue(result['refusal_disagreement'])
        self.assertFalse(result['rule_refusal'])
        result=format_answer(self.model('invalid json'),'Which?',answer)
        self.assertTrue(result['fallback'])

    def test_scientific_judge_adapter_only_changes_evaluation_system_prompt(self):
        from eval.qasper.judge import ScientificJudge
        from langchain_core.messages import SystemMessage,HumanMessage
        from langchain_core.runnables import RunnableLambda
        captured=[]
        model=RunnableLambda(lambda messages,**kwargs: captured.append(messages) or None)
        human=HumanMessage('Original scientific data')
        ScientificJudge(model).bind(response_format={}).invoke([SystemMessage("Extract factual company-policy assertions about the company's policies."),human])
        self.assertIn('scientific assertions',captured[0][0].content)
        self.assertIs(captured[0][1],human)

    def test_question_processing_best_reference_traces_and_failure_flags(self):
        from eval.qasper.run import process_question
        paper=canonical_paper(raw_paper());q=paper['questions'][0]
        q['annotations'].insert(0,{**q['annotations'][0],'reference':'Other text'})
        class Session:
            def retrieval(self,paper,question):return {}
            def ask(self,question):
                self.question=question
                return {'answer':'Alpha beta.\nSources: p.md','latency_seconds':.1,'thread_id':'fresh',
                        'events':[{'kind':'search','query':'actual query','chunks':[{'source':paper['source'],'parent_id':'parent','content':next(t for t in paper['paragraphs'].values() if t.startswith('Alpha beta'))}]}],
                        'evidence':'Actual evidence','crag':{'routes':['general_chat']}}
        session=Session()
        judgment={'correctness':1,'faithfulness':1}
        with patch('eval.qasper.run.judge_scientific',return_value=(judgment,[],.1)) as judge:
            record=process_question(session,paper,q,self.model('{"short_answer":"Alpha beta"}'),None,24000)
        self.assertEqual(session.question,q['question'])
        self.assertEqual(record['answer_f1_short'],1)
        self.assertEqual(record['best_reference_index'],1)
        self.assertEqual(record['evidence_f1'],1)
        self.assertIn('routed_to_direct_chat',record['failure_categories'])
        self.assertEqual(judge.call_args.args[2]['reference'],'Alpha beta')

    def test_real_ingestion_isolated_paper_and_pooled_collections(self):
        from types import SimpleNamespace
        from app import config
        from app.db.vector_store import VectorStore
        from langchain_core.embeddings import Embeddings
        from langchain_qdrant.sparse_embeddings import SparseEmbeddings,SparseVector
        from qdrant_client import QdrantClient
        from eval.qasper.pipeline import QasperBackend
        class Dense(Embeddings):
            def embed_documents(self,texts):return [[1.,0.,0.] for text in texts]
            def embed_query(self,text):return [1.,0.,0.]
        class Sparse(SparseEmbeddings):
            def embed_documents(self,texts):return [self.embed_query(t) for t in texts]
            def embed_query(self,text):return SparseVector(indices=[0],values=[1.])
        class Rag:
            calls=[]
            def __init__(self,*a,**kw):pass
            def ask(self,thread,question,history):self.calls.append((thread,question,history));return 'Alpha beta'
            def drop_thread(self,thread):pass
        papers={pid:canonical_paper(raw_paper(pid,qid='q-'+pid)) for pid in ('p1','p2')}
        for mode in ('per_paper','pooled'):
            with tempfile.TemporaryDirectory() as d:
                root=Path(d).resolve();client=QdrantClient(':memory:')
                with patch.multiple(config,DATA_DIR=root/'data',DENSE_DIM=3),patch.multiple(VectorStore,_client=client,_dense=Dense(),_sparse=Sparse()),\
                     patch('app.core.reranker.get_reranker',return_value=SimpleNamespace(model=SimpleNamespace(device='cpu'))),patch('app.core.rag.UserRAG',Rag):
                    backend=QasperBackend(root,papers,mode,True)
                    session=backend.session('p1',['p1','p2'])
                    self.assertIs(backend.session('p1',['p1','p2']),session)
                    self.assertEqual(len(backend.indexed_papers),1 if mode=='per_paper' else 2)
                    self.assertEqual(set(session.docs.list()),{papers['p1']['source']} if mode=='per_paper' else {p['source'] for p in papers.values()})
                    session.ask('Verbatim question?');session.ask('Verbatim question?')
                    self.assertNotEqual(Rag.calls[-1][0],Rag.calls[-2][0])
                    self.assertEqual(Rag.calls[-1][1:],('Verbatim question?',[]))
                    self.assertTrue(client.collection_exists(session.collection+'_prefix_off'))
                    backend.finish_paper('p1')
                    if mode=='per_paper':self.assertFalse(client.collection_exists(session.collection))
                    backend.close()

@unittest.skipUnless(HAVE_MODELS,'install project dependencies for runner integration tests')
class QasperResumeTests(unittest.TestCase):
    def test_runner_checkpoints_resumes_and_cleans_only_owned_work(self):
        import sys
        from types import SimpleNamespace
        from app import config
        from app.db.vector_store import VectorStore  # Load numpy before temporarily patching sys.modules.
        from eval.qasper.run import main
        paper=canonical_paper(raw_paper())
        papers={paper['id']:paper}
        setups=[]
        class Session:
            def retrieval(self,p,q):
                return {f'prefix_on_{mode}':{'metrics':{'hit_at_5':1.,'mrr_at_5':1.}} for mode in ('dense','hybrid','hybrid_rerank')}
            def ask(self,question):
                self.question=question
                return {'answer':'Alpha beta','latency_seconds':.2,'thread_id':'new-thread','events':[],'crag':{'routes':['document_qa']},'evidence':''}
        class Backend:
            reranker_device='cpu'
            def __init__(self,*args):self.indexed_papers=[paper['id']];setups.append(self)
            def session(self,*args):return Session()
            def finish_paper(self,*args):pass
            def close(self):self.closed=True
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'ordinary-user-files').mkdir()
            keep=root/'ordinary-user-files/keep.txt';keep.write_text('keep')
            arguments=['--n','1','--out',str(root/'results'),'--cache-dir',str(root/'cache'),
                       '--sample-dir',str(root/'samples'),'--work-root',str(root/'work')]
            torch=SimpleNamespace(cuda=SimpleNamespace(is_available=lambda:False))
            with patch.dict(os.environ),patch.dict(sys.modules,{'torch':torch}), \
                 patch.multiple(config,DATA_DIR=root/'unused',QDRANT_PATH=str(root/'unused-db'),RERANK_DEVICE='cpu'), \
                 patch('eval.qasper.run.load_split',return_value=(papers,{'split':'test','dataset_hash':'handmade'})), \
                 patch('eval.qasper.run.runtime_info',return_value={'packages':{},'gpu':'fake'}), \
                 patch('eval.qasper.run.ollama_info',return_value={'model_digests':{}}), \
                 patch('eval.qasper.run.QasperBackend',Backend), \
                 patch('eval.qasper.run.make_judge',return_value=None), \
                 patch('eval.qasper.run.format_answer',return_value={'short_answer':'Alpha beta','full_answer':'Alpha beta',
                       'rule_refusal':False,'format_refusal':False,'refusal_disagreement':False,'fallback':False}), \
                 patch('eval.qasper.run.judge_scientific',return_value=({'correctness':1,'faithfulness':None},[],.1)), \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(arguments),0)
                self.assertEqual(main(arguments+['--resume']),0)
                with self.assertRaises(ValueError):main(arguments)
            self.assertEqual(len(setups),1)
            self.assertTrue(setups[0].closed)
            self.assertEqual(keep.read_text(),'keep')
            self.assertFalse(list((root/'work').glob('eval-*')))
            partials=list((root/'results').glob('*.partial.json'))
            self.assertEqual(len(partials),1)
            saved=json.loads(partials[0].read_text())
            self.assertEqual(saved['results']['question-a']['answer_f1_short'],1)
            self.assertEqual(saved['summary']['complete'],1)
            report=next((root/'results').glob('*.md')).read_text()
            for label in ('95% CI','not significant','float','33.63','60.92'):
                self.assertIn(label,report)

    def test_missing_agent_prediction_uses_official_zero_and_no_judge_score(self):
        from eval.qasper.run import process_question
        paper=canonical_paper(raw_paper(kind='none'));question=paper['questions'][0]
        class Session:
            def retrieval(self,*args):return {}
            def ask(self,*args):return {'execution_error':'agent failed','events':[],'latency_seconds':.1,'crag':{}}
        record=process_question(Session(),paper,question,None,None,24000)
        self.assertEqual(record['answer_f1_short'],0)
        self.assertEqual(record['evidence_f1'],0)
        self.assertEqual(record['unanswerable_rule'],0)
        self.assertNotIn('judge_correctness',record)
        self.assertIn('execution_error',record['failure_categories'])

class QasperNotebookTests(unittest.TestCase):
    def test_qasper_selector_skips_synthetic_and_builds_three_and_full_commands(self):
        from types import SimpleNamespace
        import sys
        notebook=json.loads((Path(__file__).resolve().parents[1]/'colab_eval.ipynb').read_text())
        cells={cell['id']:''.join(cell['source']) for cell in notebook['cells']}
        calls=[]
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ), contextlib.redirect_stdout(io.StringIO()):
            def mapped_path(value):
                return Path(str(value).replace('/content/drive/MyDrive',directory))
            namespace={'BENCHMARK':'QASPER','EVAL_MODE':'Full evaluation','RUN_PREFIX_ABLATION':True,
                       'ANSWER_MODEL':'answer','SELECTED_JUDGE':'judge','Path':mapped_path,'sys':sys,'os':os,
                       'subprocess':SimpleNamespace(run=lambda *a,**k:None),'warm_and_verify':lambda *a:None}
            exec(cells['cell-14'],namespace)
            exec(cells['cell-16'],namespace)
            exec(cells['cell-18'],namespace)
            namespace['run_eval_logged']=lambda command,log:calls.append((command,log))
            exec(cells['qasper-setup'],namespace)
            for cell in ('qasper-data','qasper-smoke','qasper-full'):
                exec(cells[cell],namespace)
        self.assertEqual(len(calls),3)
        self.assertIn('--prepare-only',calls[0][0])
        self.assertEqual(calls[1][0][calls[1][0].index('--limit')+1],'3')
        self.assertNotIn('--limit',calls[2][0])
        self.assertIn('--resume',calls[2][0])
        command=calls[2][0]
        for flag,value in (('--split','test'),('--n','250'),('--seed','42'),('--mode','per_paper'),('--format-model','answer')):
            self.assertEqual(command[command.index(flag)+1],value)
        self.assertIn('--prefix-ablation',command)
