"""Graph nodes. Outer graph first, then the per-question research subgraph."""
import json
import re

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage, SystemMessage, ToolMessage
from langgraph.types import Command

from app import config
from app.agent import prompts
from app.utils import estimate_tokens, parse_json
from app.agent.observability import emit

ROUTES = {"general_chat", "document_inventory", "document_overview", "document_qa", "needs_clarification"}

# Cheap keyword routing, used when the model's answer is missing or unusable.
_INVENTORY = re.compile(r"\b(what|which|list)\b.*\b(documents?|files?)\b.*\b(have|uploaded|there|library)\b|\blist (my |the )?(documents|files)\b", re.I)
_OVERVIEW = re.compile(r"\b(summari[sz]e|overview of|what is) each\b|\beach (document|file|paper)\b.*\babout\b", re.I)
_DOCQ = re.compile(r"\b(document|doc|file|paper|pdf|report|according to|in the)\b", re.I)


def keyword_route(text):
    if _OVERVIEW.search(text):
        return "document_overview"
    if _INVENTORY.search(text):
        return "document_inventory"
    if _DOCQ.search(text):
        return "document_qa"
    return "general_chat"


def _history_text(history, limit=8):
    lines = [f"{h['role']}: {h['content']}" for h in history[-limit:]
             if h.get("role") in {"user", "assistant"} and h.get("content")]
    return "\n".join(lines) or "(none)"


def _last_text(state):
    return str(state["messages"][-1].content).strip()


# ------------------------------------------------------------- outer graph
def router(state, llm, observer=None):
    text = _last_text(state)
    fallback = keyword_route(text)
    try:
        reply = llm.invoke([
            SystemMessage(prompts.ROUTER),
            HumanMessage(f"documents: {state.get('documents', [])}\n"
                         f"history:\n{_history_text(state.get('history', []))}\n\nmessage: {text}"),
        ])
        data = parse_json(reply.content) or {}
        route = data.get("route")
        if route not in ROUTES:
            route = fallback
        # the model likes "general_chat" too much; trust the doc keywords over it
        if route == "general_chat" and fallback != "general_chat" and state.get("documents"):
            route = fallback
        emit(observer, "route", route=route)
        return {"route": route, "clarification": str(data.get("clarification") or "")}
    except Exception:
        emit(observer, "route", route=fallback)
        return {"route": fallback, "clarification": ""}


def direct_chat(state, llm):
    msgs = [SystemMessage(prompts.DIRECT)]
    for h in state.get("history", [])[-8:]:
        if h.get("content"):
            msgs.append(HumanMessage(h["content"]) if h["role"] == "user" else AIMessage(h["content"]))
    msgs.append(HumanMessage(_last_text(state)))
    return {"messages": [llm.invoke(msgs)]}


def inventory(state):
    docs = state.get("documents", [])
    text = (f"You have {len(docs)} document(s):\n\n" + "\n".join(f"- {d}" for d in docs)) if docs \
        else "No documents uploaded yet. Add some on the Documents page."
    return {"messages": [AIMessage(text)]}


def no_documents(state):
    return {"messages": [AIMessage("There are no documents to search yet. Upload a PDF or Markdown "
                                   "file on the Documents page, then ask again.")]}


def clarify(state):
    return {"messages": [AIMessage(state.get("clarification") or "Could you say a bit more about what you need?")]}


def overview(state, llm, observer=None):
    payload = {"previews": state.get("previews", []), "request": _last_text(state)}
    emit(observer, "preview", content=json.dumps(payload["previews"], ensure_ascii=False))
    return {"messages": [llm.invoke([SystemMessage(prompts.OVERVIEW),
                                     HumanMessage(json.dumps(payload, ensure_ascii=False))])]}


def summarize(state, llm):
    history = [h for h in state.get("history", []) if h.get("content")]
    if len(history) < 4:
        return {"summary": "", "answers": [{"reset": True}]}
    convo = "\n".join(f"{h['role']}: {h['content']}" for h in history[-6:])
    reply = llm.invoke([SystemMessage(prompts.SUMMARY), HumanMessage(convo)])
    return {"summary": reply.content, "answers": [{"reset": True}]}


def rewrite(state, llm):
    query = _last_text(state)
    ctx = ""
    if state.get("summary"):
        ctx += f"Conversation summary:\n{state['summary']}\n\n"
    ctx += f"Recent messages:\n{_history_text(state.get('history', []))}\n\nQuery:\n{query}"
    data = parse_json(llm.invoke([SystemMessage(prompts.REWRITE), HumanMessage(ctx)]).content)
    clear_msgs = [RemoveMessage(id=m.id) for m in state["messages"]]
    if data is None:  # unparseable: just search with the raw query
        return {"is_clear": True, "questions": [query], "original_query": query, "messages": clear_msgs}
    questions = [q for q in (data.get("questions") or []) if str(q).strip()][:3]
    if data.get("is_clear", True) and questions:
        return {"is_clear": True, "questions": questions, "original_query": query, "messages": clear_msgs}
    ask = str(data.get("clarification") or "").strip() or "Could you make the question a bit more specific?"
    return {"is_clear": False, "messages": [AIMessage(ask)]}


def await_clarification(state):
    # graph interrupts before this node; resuming re-runs rewrite with the user's reply
    return {}


def aggregate(state, llm):
    answers = sorted(state.get("answers", []), key=lambda a: a["index"])
    unique = list(dict.fromkeys(a["answer"].strip() for a in answers if a.get("answer", "").strip()))
    if not unique:
        return {"messages": [AIMessage("I couldn't produce an answer from the documents.")]}
    if len(unique) == 1:
        return {"messages": [AIMessage(unique[0])]}
    parts = "\n\n".join(f"Part {i}:\n{a}" for i, a in enumerate(unique, 1))
    reply = llm.invoke([SystemMessage(prompts.AGGREGATE),
                        HumanMessage(f"Original question: {state.get('original_query')}\n\n{parts}")])
    return {"messages": [AIMessage(reply.content)]}


# --------------------------------------------------------- research subgraph
def _is_grade(m):
    return isinstance(m, AIMessage) and not m.tool_calls and str(m.content).startswith("[grade]")


def _work_messages(messages):
    return [m for m in messages if not _is_grade(m)]


def _latest_tool_output(messages):
    out = []
    for m in reversed(messages):
        if isinstance(m, ToolMessage):
            out.append(str(m.content)[:1600])
        elif out:
            break
    return "\n\n".join(reversed(out))


def orchestrator(state, llm_tools):
    sys = [SystemMessage(prompts.ORCHESTRATOR)]
    if state.get("grade_feedback"):
        sys.append(HumanMessage(f"[Retrieval feedback]\n{state['grade_feedback']}"))
    if state.get("context_summary"):
        sys.append(HumanMessage(f"[Notes from earlier research]\n{state['context_summary']}"))
    work = _work_messages(state.get("messages", []))
    if not work:
        q = HumanMessage(state["question"])
        reply = llm_tools.invoke(sys + [q, HumanMessage("Start by calling search_child_chunks.")])
        return {"messages": [q, reply], "tool_calls": len(reply.tool_calls), "iterations": 1}
    reply = llm_tools.invoke(sys + work)
    return {"messages": [reply], "tool_calls": len(reply.tool_calls), "iterations": 1}


def route_orchestrator(state):
    if state.get("iterations", 0) >= config.MAX_ITERATIONS or state.get("tool_calls", 0) > config.MAX_TOOL_CALLS:
        return "fallback"
    return "tools" if state["messages"][-1].tool_calls else "collect"


def _grade_msg(grade, reason):
    return AIMessage(f"[grade] {grade}: {reason}")


def grade(state, llm, observer=None):
    """CRAG step: is what we just retrieved good enough to keep going on?"""
    output = _latest_tool_output(state["messages"])
    retries = state.get("retries", 0)

    def retry(reason, origin="llm"):
        emit(observer, "grade", index=state.get("index"), question=state.get("question"),
             evidence=output,
             verdict="insufficient", reason=reason, origin=origin,
             retry_scheduled=retries + 1 <= config.CRAG_MAX_RETRIES,
             retry_count=retries + 1)
        if retries + 1 > config.CRAG_MAX_RETRIES:
            return Command(update={"messages": [_grade_msg("insufficient", "retry limit reached")],
                                   "retries": retries + 1, "grade_feedback": reason}, goto="fallback")
        return Command(update={"messages": [_grade_msg("insufficient", reason)],
                               "retries": retries + 1, "grade_feedback": reason}, goto="orchestrator")

    if not output:
        return Command(update={}, goto="orchestrator")
    if "NO_RELEVANT_CHUNKS" in output or "RETRIEVAL_ERROR" in output:
        return retry("Nothing relevant came back. Try different wording or a broader query.", "retrieval_failure")
    # parent fetches are follow-ups on hits we already graded
    if "Parent ID:" in output and "[Chunk" not in output:
        return Command(update={}, goto="maybe_compress")
    try:
        data = parse_json(llm.invoke([SystemMessage(prompts.GRADE),
                                      HumanMessage(f"Question: {state['question']}\n\nRetrieved:\n{output}")]).content) or {}
    except Exception:
        data = {}
    if str(data.get("grade", "")).lower() == "sufficient":
        emit(observer, "grade", index=state.get("index"), question=state.get("question"),
             evidence=output,
             verdict="sufficient", reason=data.get("reason", ""), origin="llm", retry_scheduled=False,
             retry_count=retries)
        return Command(update={"messages": [_grade_msg("sufficient", data.get("reason", ""))],
                               "retries": 0, "grade_feedback": ""}, goto="maybe_compress")
    return retry(data.get("reason") or "Results don't answer the question well. Search with a more focused query.",
                 "llm" if str(data.get("grade", "")).lower() == "insufficient" else "judge_parse_or_call_failure")


def maybe_compress(state):
    last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage) and m.tool_calls), None)
    seen = set()
    for tc in (last_ai.tool_calls if last_ai else []):
        if tc["name"] == "search_child_chunks":
            seen.add(f"search::{tc['args'].get('query', '')}")
        elif tc["name"] == "retrieve_parent_chunks":
            seen.add(f"parent::{tc['args'].get('parent_id', '')}")
    summary_tokens = estimate_tokens([HumanMessage(state.get("context_summary", ""))])
    used = estimate_tokens(_work_messages(state["messages"])) + summary_tokens
    budget = config.COMPRESS_BASE_TOKENS + int(summary_tokens * config.COMPRESS_GROWTH)
    return Command(update={"seen": seen}, goto="compress" if used > budget else "orchestrator")


def compress(state, llm):
    work = _work_messages(state["messages"])
    log = f"Question: {state['question']}\n\n"
    if state.get("context_summary"):
        log += f"Earlier notes:\n{state['context_summary']}\n\n"
    for m in work[1:]:
        if isinstance(m, ToolMessage):
            log += f"[tool {m.name}]\n{m.content}\n\n"
        elif isinstance(m, AIMessage):
            calls = ", ".join(f"{t['name']}({t['args']})" for t in m.tool_calls)
            log += f"[assistant{' -> ' + calls if calls else ''}]\n{m.content}\n\n"
    notes = llm.invoke([SystemMessage(prompts.COMPRESS), HumanMessage(log)]).content
    done = sorted(state.get("seen", set()))
    if done:
        notes += "\n\nAlready executed (don't repeat):\n" + "\n".join(f"- {d}" for d in done)
    # keep the original question message, drop the rest
    return {"context_summary": notes, "messages": [RemoveMessage(id=m.id) for m in state["messages"][1:]]}


def fallback(state, llm, observer=None):
    reason = "retry_limit" if state.get("retries", 0) > config.CRAG_MAX_RETRIES else "agent_budget"
    emit(observer, "fallback", index=state.get("index"), question=state.get("question"), reason=reason)
    data = list(dict.fromkeys(str(m.content) for m in state["messages"] if isinstance(m, ToolMessage)))
    ctx = ""
    if state.get("context_summary"):
        ctx += f"Notes:\n{state['context_summary']}\n\n"
    ctx += "\n\n".join(f"--- source {i} ---\n{d}" for i, d in enumerate(data, 1)) or "Nothing was retrieved."
    reply = llm.invoke([SystemMessage(prompts.FALLBACK), HumanMessage(f"Question: {state['question']}\n\n{ctx}")])
    return {"messages": [reply]}


def collect(state):
    last = state["messages"][-1]
    ok = isinstance(last, AIMessage) and last.content and not last.tool_calls
    text = last.content if ok else "I couldn't produce an answer for this part."
    return {"answers": [{"index": state["index"], "question": state["question"], "answer": text}]}
