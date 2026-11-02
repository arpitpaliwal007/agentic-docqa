"""Scientific-domain adapter for the existing audited two-pass evaluation judge."""
from eval.judge import JUDGE_POLICY, judge_answer

QASPER_JUDGE_POLICY = JUDGE_POLICY + '+qasper-scientific-v1'


class ScientificJudge:
    def __init__(self, model):
        self.model = model

    def bind(self, **kwargs):
        return ScientificJudge(self.model.bind(**kwargs))

    def invoke(self, messages):
        from langchain_core.messages import SystemMessage
        converted = []
        for message in messages:
            if isinstance(message, SystemMessage):
                text = message.content.replace("company's policies", 'research paper').replace('company-policy', 'scientific')
                text = text.replace('company policies', 'scientific findings').replace('factual policy', 'factual scientific')
                text = ('The question and references concern a scientific research paper, not company policies. '
                        'Extract scientific assertions including methods, experiments, datasets, numbers and conclusions.\n' + text)
                converted.append(SystemMessage(text))
            else:
                converted.append(message)
        return self.model.invoke(converted)


def judge_scientific(model, question, annotation, answer, evidence, context_chars=24000):
    payload = {'question':question, 'reference_answer':annotation['reference'], 'answerable':annotation['type'] != 'none'}
    value, attempts, latency = judge_answer(ScientificJudge(model), payload, answer, evidence, context_chars)
    if value is not None:
        value['qasper_policy'] = QASPER_JUDGE_POLICY
    return value, attempts, latency
