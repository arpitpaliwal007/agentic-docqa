"""Two-level LangGraph.

Outer:  router -> {direct_chat | inventory | overview | clarify | no_documents | summarize}
        summarize -> rewrite -> (unclear: await_clarification, interrupt) | Send(research) x N
        research results -> aggregate

Inner (one per rewritten question):
        orchestrator -> tools -> grade -> maybe_compress -> (compress) -> orchestrator ...
                                      \\-> orchestrator (retry with feedback)
                                      \\-> fallback (retry limit)
        orchestrator with no tool calls -> collect
"""
from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Send

from app import config
from app.agent import nodes as n
from app.agent.state import AgentState, State


def build_research_graph(llm, tools):
    g = StateGraph(AgentState)
    g.add_node("orchestrator", partial(n.orchestrator, llm_tools=llm.bind_tools(tools, parallel_tool_calls=False)))
    g.add_node("tools", ToolNode(tools))
    g.add_node("grade", partial(n.grade, llm=llm), destinations=("maybe_compress", "orchestrator", "fallback"))
    g.add_node("maybe_compress", n.maybe_compress, destinations=("compress", "orchestrator"))
    g.add_node("compress", partial(n.compress, llm=llm))
    g.add_node("fallback", partial(n.fallback, llm=llm))
    g.add_node("collect", n.collect)
    g.add_edge(START, "orchestrator")
    g.add_conditional_edges("orchestrator", n.route_orchestrator,
                            {"tools": "tools", "fallback": "fallback", "collect": "collect"})
    g.add_edge("tools", "grade")
    g.add_edge("compress", "orchestrator")
    g.add_edge("fallback", "collect")
    g.add_edge("collect", END)
    return g.compile()


def _route_after_router(state):
    route = state.get("route", "document_qa")
    if route == "general_chat":
        return "direct_chat"
    if route == "document_inventory":
        return "inventory"
    if route == "needs_clarification":
        return "clarify"
    if not state.get("documents"):
        return "no_documents"
    return "overview" if route == "document_overview" else "summarize"


def _route_after_rewrite(state):
    if not state.get("is_clear"):
        return "await_clarification"
    return [Send("research", {"question": q, "index": i}) for i, q in enumerate(state["questions"])]


def build_graph(llm, tools, checkpointer=None):
    research = build_research_graph(llm, tools)

    def run_research(payload):
        # run the subgraph in isolation so its tool chatter never lands in the outer messages
        out = research.invoke({"question": payload["question"], "index": payload["index"], "messages": []},
                              {"recursion_limit": config.RECURSION_LIMIT})
        return {"answers": out.get("answers", [])}

    g = StateGraph(State)
    g.add_node("router", partial(n.router, llm=llm))
    g.add_node("direct_chat", partial(n.direct_chat, llm=llm))
    g.add_node("inventory", n.inventory)
    g.add_node("no_documents", n.no_documents)
    g.add_node("clarify", n.clarify)
    g.add_node("overview", partial(n.overview, llm=llm))
    g.add_node("summarize", partial(n.summarize, llm=llm))
    g.add_node("rewrite", partial(n.rewrite, llm=llm))
    g.add_node("await_clarification", n.await_clarification)
    g.add_node("research", run_research)
    g.add_node("aggregate", partial(n.aggregate, llm=llm))

    g.add_edge(START, "router")
    g.add_conditional_edges("router", _route_after_router,
                            ["direct_chat", "inventory", "clarify", "no_documents", "overview", "summarize"])
    for terminal in ("direct_chat", "inventory", "clarify", "no_documents", "overview"):
        g.add_edge(terminal, END)
    g.add_edge("summarize", "rewrite")
    g.add_conditional_edges("rewrite", _route_after_rewrite, ["await_clarification", "research"])
    g.add_edge("await_clarification", "rewrite")
    g.add_edge("research", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile(checkpointer=checkpointer or InMemorySaver(), interrupt_before=["await_clarification"])
