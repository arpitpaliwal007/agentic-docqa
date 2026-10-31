"""Eval v2 regressions, including exact answers from the original saved Colab run."""
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from eval.metrics import (add_diagnostics, agent_retrieval_metrics, answer_metrics,
                          classify_failure, key_fact_recall, refusal_detector)
from eval.run_eval import ROOT, fixtures
from eval.report import render_report
from eval.storage import Results


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.questions = {q['id']: q for q in fixtures()[1]}
        self.saved = json.loads((ROOT / 'tests/fixtures/saved_eval_answers.json').read_text())['answers']

    def test_exact_saved_refusals_and_partial_answer(self):
        for qid, text in self.saved.items():
            with self.subTest(qid=qid):
                metric = answer_metrics(self.questions[qid], text, {'abstained': False, 'correctness': 0})
                self.assertEqual(metric['rule_refusal'], qid != 'q10')
                self.assertEqual(metric['combined_abstained'], qid != 'q10')
                self.assertEqual(metric['partial_refusal'], qid == 'q07')
        self.assertEqual(key_fact_recall(self.saved['q07'], self.questions['q07']), .5)
        self.assertEqual(key_fact_recall(self.saved['q10'], self.questions['q10']), 1)
        self.assertIsNone(key_fact_recall(self.saved['q17'], self.questions['q17']))

    def test_refusal_phrases_do_not_confuse_policy_negations(self):
        for text in ('The files do not contain that fact.', "It isn't covered.", "It doesn't say.",
                     'The rating is not specified.', 'I cannot find the deadline.', 'There is no information.'):
            self.assertTrue(refusal_detector(text), text)
        for text in ('Original packaging is not required.', 'Outbound fees are not refunded.', 'No payment card is required.'):
            self.assertFalse(refusal_detector(text), text)
        self.assertTrue(answer_metrics(self.questions['q10'], 'No box is needed.', {'abstained': True})['combined_abstained'])

    def test_fact_matching_normalizes_currency_punctuation_and_word_boundaries(self):
        question = {**self.questions['q08'], 'must_contain': ['$12 per month', '20 connected devices']}
        self.assertEqual(key_fact_recall('12 dollars per MONTH; 20 connected devices!', question), 1)
        self.assertEqual(key_fact_recall('$12 per month and 200 connected devices', question), .5)
        self.assertEqual(key_fact_recall('WRT A2 24; 24 months; Aurora Hub A2', self.questions['q14']), 1)

    def test_key_fact_disagreements_never_overwrite_scores(self):
        score = {'correctness': 0, 'abstained': False}
        self.assertTrue(answer_metrics(self.questions['q10'], self.saved['q10'], score)['judge_disagreement'])
        self.assertEqual(score['correctness'], 0)
        self.assertTrue(answer_metrics(self.questions['q06'], 'An unrelated answer', {'correctness': 1, 'abstained': False})['judge_disagreement'])
        self.assertFalse(answer_metrics(self.questions['q07'], self.saved['q07'], {'correctness': .5, 'abstained': False})['judge_disagreement'])

    def record(self, events, answer=None):
        q = copy.deepcopy(self.questions['q06'])
        q.update(status='complete', errors=[], retrieval={'hybrid_rerank': {'metrics': {'hit_at_5': 1}}},
                 e2e={'answer': answer or self.saved['q06'], 'events': events, 'judgment': {'correctness': 0, 'abstained': False}})
        add_diagnostics(q, q)
        return q

    def test_failure_label_uses_actual_agent_search_not_direct_hit(self):
        miss = [{'kind': 'search', 'query': 'owner cancel account', 'chunks': [
            {'source': 'accounts.md', 'parent_id': 'accounts_p0', 'chunk_id': 'uuid-a', 'content': 'Password reset links expire after 20 minutes.'}]}]
        record = self.record(miss)
        self.assertEqual(record['e2e']['agent_retrieval']['agent_retrieval_hit'], 0)
        self.assertEqual(classify_failure(record, record), 'retrieval miss (agent queries)')
        hit = [{'kind': 'search', 'query': 'deletion cancellation', 'chunks': [
            {'source': 'accounts.md', 'parent_id': 'accounts_p1', 'chunk_id': 'uuid-b', 'content': 'A 14-day cancellation period applies.'}]}]
        record = self.record(hit)
        self.assertEqual(classify_failure(record, record), 'generation error')
        self.assertEqual(record['e2e']['agent_retrieval']['queries'], ['deletion cancellation'])
        self.assertEqual(record['e2e']['agent_retrieval']['searches'][0]['chunks'][0]['chunk_id'], 'uuid-b')

    def test_grader_checks_use_its_exact_evidence_even_if_events_interleave(self):
        q = self.questions['q06']
        good = '[Chunk 1]\nParent ID: accounts_p1\nFile: accounts.md\nA 14-day cancellation period applies.'
        bad = '[Chunk 1]\nParent ID: refunds_p0\nFile: refunds.md\nReturns take 30 days.'
        for verdict, evidence in (('insufficient', good), ('sufficient', bad)):
            events = [{'kind': 'search', 'query': 'a different parallel branch', 'chunks': []},
                      {'kind': 'grade', 'index': 1, 'question': q['question'], 'verdict': verdict,
                       'origin': 'llm', 'evidence': evidence}]
            result = self.record(events)
            self.assertTrue(result['e2e']['agent_retrieval']['grader_issue'])
            self.assertEqual(classify_failure(result, result), 'grader issue')
        parent = [{'kind': 'parent', 'source': 'accounts.md', 'parent_id': 'accounts_p1',
                   'text': 'A 14-day cancellation period applies.', 'error': None}]
        self.assertEqual(agent_retrieval_metrics(parent, q)['agent_retrieval_hit'], 1)
        self.assertIsNone(agent_retrieval_metrics(parent, self.questions['q17'])['agent_retrieval_hit'])

    def test_report_lists_disagreements_and_six_real_ablation_conditions(self):
        record = self.record([{'kind':'search', 'query':'actual rewritten query', 'chunks':[]}])
        record['e2e']['judgment'].update(faithfulness=None, judge_run_disagreement=True)
        record['retrieval'] = {f'prefix_{prefix}_{mode}': {'metrics': {'hit_at_5':1,'mrr_at_5':1,'precision_at_5':.2,'evidence_coverage_at_5':1}}
                               for prefix in ('off','on') for mode in ('dense','hybrid','hybrid_rerank')}
        metadata = {'run_id':'test','model':'answer','judge_model':'judge','judge_policy':'single-v2-policy',
                    'self_judge':False,'expected_questions':20,'fingerprint':'test','eval_version':2}
        text = render_report({'metadata': metadata, 'results': {'q06': record}})
        self.assertIn('Wrong refusal on answerable (combined) | 1.000 | 1 |', text)
        self.assertIn('Rule vs judge refusal disagreements: q06', text)
        self.assertIn('actual rewritten query', text)
        self.assertIn('retrieval miss (agent queries)', text)
        self.assertIn('single-v2-policy', text)
        self.assertEqual(sum(line.startswith('| on |') or line.startswith('| off |') for line in text.splitlines()), 6)
        for kind in ('single_fact','multi_part','paraphrase','exact_identifier','unanswerable'):
            self.assertIn(f'| {kind} |', text)

    def test_full_rejudge_defaults_to_all_twenty_with_one_policy(self):
        from eval.rejudge import rejudge_records
        from eval.judge import JUDGE_POLICY
        records = {}
        for question in self.questions.values():
            records[question['id']] = {**question, 'status':'complete', 'errors':[], 'retrieval':{},
                'e2e': {'answer': self.saved.get(question['id'], 'Saved answer'), 'evidence':'Actual captured evidence',
                        'events':[], 'judgment':{'correctness':0,'abstained':False,'faithfulness':None}, 'judge_attempts':[]}}
        metadata = {'run_id':'all','fingerprint':'new','pipeline_fingerprint':'old','rejudged_from':'old.json',
                    'report_basename':'all','model':'answer','judge_model':'judge','self_judge':False,
                    'expected_questions':20,'judge_policy':JUDGE_POLICY,'retained_previous_judgments':0,'eval_version':2}
        source = {'metadata':metadata, 'results':records}
        before = copy.deepcopy(source)
        new_judgment = {'policy':JUDGE_POLICY,'correctness':.5,'abstained':False,'faithfulness':None,'judge_run_disagreement':False}
        with tempfile.TemporaryDirectory() as directory:
            result = Results(directory,'all',metadata)
            with patch('eval.rejudge.judge_answer',return_value=(new_judgment,[],.1)) as judge, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(rejudge_records(source,result,None,24000),0)
                self.assertEqual(judge.call_count,20)
            self.assertEqual(source,before)
            for record in result.value['results'].values():
                self.assertEqual(record['e2e']['judge_policy'],JUDGE_POLICY)
                self.assertEqual(record['e2e']['judgment']['correctness'],.5)
                self.assertEqual(record['e2e']['previous_judgment']['correctness'],0)
            self.assertNotIn('Mixed-policy report',render_report(result.value))

    def test_notebook_constructs_full_and_uniform_rejudge_commands_without_models(self):
        import subprocess
        import sys
        notebook=json.loads((ROOT/'eval/colab_eval.ipynb').read_text())
        cells={cell['id']:''.join(cell['source']) for cell in notebook['cells']}
        self.assertIn('/evaluation-results/v2',cells['cell-06'])
        self.assertIn('/evaluation-results/v2',cells['cell-12'])
        for field in ('ANSWER_MODEL','JUDGE_MODEL','RUN_PREFIX_ABLATION'):
            self.assertIn(field,cells['cell-10'])
        for mode in ('Full evaluation','Rejudge saved answers'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                source=Path(directory)/'original.json'
                source.write_text('{}')
                process=MagicMock()
                process.__enter__.return_value=process
                process.stdout=[]
                process.wait.return_value=0
                namespace={'Path':Path,'subprocess':subprocess,'sys':sys,'RESULTS_DIR':Path(directory),
                           'EVAL_MODE':mode,'RUN_PREFIX_ABLATION':True,'ANSWER_MODEL':'answer',
                           'SELECTED_JUDGE':'judge','SAVED_RESULTS':str(source)}
                with patch('subprocess.Popen',return_value=process) as spawn, contextlib.redirect_stdout(io.StringIO()):
                    exec(cells['cell-14'],namespace)
                    exec(cells['cell-16'],namespace)
                commands=[call.args[0] for call in spawn.call_args_list]
                if mode=='Full evaluation':
                    self.assertEqual(len(commands),2)
                    self.assertIn('--prefix-ablation',commands[0])
                    self.assertIn('--limit',commands[0])
                    self.assertNotIn('--limit',commands[1])
                    self.assertEqual(commands[1][commands[1].index('--judge-model')+1],'judge')
                else:
                    self.assertEqual(len(commands),1)
                    self.assertIn('eval/rejudge.py',commands[0])
                    self.assertNotIn('--only-failed',commands[0])
                    self.assertEqual(namespace['REPORT_DIR'],Path(directory)/'rejudged-all')


HAVE_GRAPH = importlib.util.find_spec('langgraph') is not None


@unittest.skipUnless(HAVE_GRAPH, 'install project dependencies for embedding and model-interface tests')
class EmbeddingAndJudgeTests(unittest.TestCase):
    def fake_embeddings(self):
        from langchain_core.embeddings import Embeddings
        class Dense(Embeddings):
            documents = []
            queries = []
            def embed_documents(self, texts):
                self.documents.extend(texts)
                return [[1.,0.,0.] if text.startswith('search_document: ') else [0.,1.,0.] for text in texts]
            def embed_query(self, text):
                self.queries.append(text)
                return [1.,0.,0.] if text.startswith('search_query: ') else [0.,1.,0.]
        return Dense()

    def test_prefix_wrapper_respects_model_flag_and_keeps_input_text(self):
        import asyncio
        from app.db.vector_store import TaskPrefixEmbeddings
        base = self.fake_embeddings()
        original = ['Raw document']
        wrapper = TaskPrefixEmbeddings(base, 'nomic-embed-text', True)
        wrapper.embed_documents(original)
        wrapper.embed_query('Raw query')
        asyncio.run(wrapper.aembed_documents(['Async document']))
        asyncio.run(wrapper.aembed_query('Async query'))
        self.assertEqual(original, ['Raw document'])
        self.assertEqual(base.documents, ['search_document: Raw document', 'search_document: Async document'])
        self.assertEqual(base.queries, ['search_query: Raw query', 'search_query: Async query'])
        for model, enabled in (('nomic-embed-text', False), ('another-model', True)):
            raw = self.fake_embeddings()
            raw.documents, raw.queries = [], []
            off = TaskPrefixEmbeddings(raw, model, enabled)
            off.embed_documents(original)
            off.embed_query('Raw query')
            self.assertEqual(raw.documents, original)
            self.assertEqual(raw.queries, ['Raw query'])

    def test_two_indexes_really_have_different_vectors_but_identical_stored_text(self):
        from app import config
        from app.db.vector_store import VectorStore
        from langchain_qdrant.sparse_embeddings import SparseEmbeddings, SparseVector
        from qdrant_client import QdrantClient
        from eval.run_eval import initialize_pipeline, score_retrieval
        class Sparse(SparseEmbeddings):
            def embed_query(self, text):
                return SparseVector(indices=[0], values=[1.])
            def embed_documents(self, texts):
                return [self.embed_query(text) for text in texts]
        client = QdrantClient(':memory:')
        self.addCleanup(client.close)
        dense = self.fake_embeddings()
        dense.documents, dense.queries = [], []
        fake_reranker = SimpleNamespace(model=SimpleNamespace(device='cpu'), rerank=lambda query, chunks: [(chunk,1.) for chunk in chunks])
        with tempfile.TemporaryDirectory() as directory, \
             patch.multiple(config, DATA_DIR=Path(directory)/'data', DENSE_DIM=3, DENSE_MODEL='nomic-embed-text', NOMIC_PREFIXES=True), \
             patch.multiple(VectorStore, _client=client, _dense=dense, _sparse=Sparse()), \
             patch('app.core.reranker.get_reranker', return_value=fake_reranker), \
             patch('app.agent.tools.get_reranker', side_effect=lambda llm, kind: fake_reranker if kind=='cross_encoder' else None):
            users, docs, vdb, _, _, _ = initialize_pipeline(Path(directory), fixtures()[0], True, True)
            name = users.collection('eval_user')
            on = client.scroll(name, limit=1000, with_vectors=True)[0]
            off = client.scroll(name+'_prefix_off', limit=1000, with_vectors=True)[0]
            self.assertTrue(on and off)
            self.assertEqual({point.payload['page_content'] for point in on}, {point.payload['page_content'] for point in off})
            self.assertTrue(all(not point.payload['page_content'].startswith('search_document: ') for point in on+off))
            self.assertTrue(all(point.vector[''] == [1.,0.,0.] for point in on))
            self.assertTrue(all(point.vector[''] == [0.,1.,0.] for point in off))
            scored = score_retrieval(fixtures()[1][0], docs, vdb, name, True)
            self.assertEqual(len(scored), 6)
            self.assertEqual({item['prefixes'] for item in scored.values()}, {False, True})
            self.assertTrue(any(query.startswith('search_query: ') for query in dense.queries))
            self.assertTrue(any(not query.startswith('search_query: ') for query in dense.queries))
            self.assertEqual(len(score_retrieval(fixtures()[1][0], docs, vdb, name, False)), 3)

    def test_two_judge_passes_average_without_key_fact_override(self):
        from eval.judge import judge_answer, JUDGE_POLICY
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        calls = []
        def respond(messages, **kwargs):
            calls.append([message.content for message in messages])
            return AIMessage(json.dumps({'correctness':0 if len(calls)==1 else 1, 'abstained':False,
                                         'claims':[], 'reason':'fake assessment'}))
        value, attempts, _ = judge_answer(RunnableLambda(respond), fixtures()[1][9], 'Original packaging is not required.', '', 24000)
        self.assertEqual(value['correctness'], .5)
        self.assertTrue(value['judge_run_disagreement'])
        self.assertEqual(value['policy'], JUDGE_POLICY)
        self.assertEqual(len(value['passes']), 2)
        self.assertNotEqual(calls[0][0], calls[1][0])
        self.assertEqual(calls[0][1], calls[1][1])
        self.assertEqual([item['pass'] for item in attempts], [1,2])

    def test_a_failed_pass_never_creates_a_single_pass_average(self):
        from eval.judge import judge_answer
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda
        calls = []
        def respond(messages, **kwargs):
            calls.append(messages)
            return AIMessage('{"correctness":1,"abstained":false,"claims":[],"reason":"ok"}' if len(calls)==1 else 'invalid JSON')
        value, attempts, _ = judge_answer(RunnableLambda(respond), fixtures()[1][0], '30 calendar days', '', 24000)
        self.assertIsNone(value)
        self.assertEqual(len(attempts), 3)
        self.assertEqual(sum('error' in attempt for attempt in attempts), 2)


if __name__ == '__main__':
    unittest.main()
