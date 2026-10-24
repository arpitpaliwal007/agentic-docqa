import operator
from typing import Annotated

from langgraph.graph import MessagesState


def answers_reducer(old: list, new: list) -> list:
    # a {"reset": True} item clears answers from the previous turn
    if new and any(x.get("reset") for x in new):
        return [x for x in new if not x.get("reset")]
    return old + new


class State(MessagesState):
    """Outer graph: routing, rewriting, fan-out, aggregation."""
    route: str
    clarification: str
    summary: str
    original_query: str
    questions: list[str]
    is_clear: bool
    answers: Annotated[list[dict], answers_reducer]
    documents: list[str]
    previews: list[dict]
    history: list[dict]


class AgentState(MessagesState):
    """Inner subgraph: one per rewritten sub-question."""
    question: str
    index: int
    context_summary: str
    grade_feedback: str
    retries: int
    seen: Annotated[set, operator.or_]
    tool_calls: Annotated[int, operator.add]
    iterations: Annotated[int, operator.add]
    answers: list[dict]
