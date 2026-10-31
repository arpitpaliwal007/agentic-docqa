"""Per-user RAG runtime: LLM, tools bound to the user's own collection, compiled graph."""
import threading

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app import config
from app.agent.graph import build_graph
from app.agent.tools import build_tools
from app.core.documents import DocumentManager
from app.core.user_store import UserStore
from app.db.vector_store import VectorStore

_llm = None
_checkpointer = InMemorySaver()   # shared; threads are keyed by uuid so users can't collide


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=config.LLM_MODEL, base_url=config.LLM_BASE_URL,
                          api_key=config.LLM_API_KEY, temperature=config.LLM_TEMPERATURE)
    return _llm


def _callbacks():
    if not config.LANGFUSE_ENABLED:
        return []
    try:
        from langfuse.langchain import CallbackHandler
        return [CallbackHandler()]
    except Exception:
        return []


class UserRAG:
    def __init__(self, users: UserStore, username: str, observer=None):
        self.username = username
        udir = users.user_dir(username)
        self.docs = DocumentManager(udir, users.collection(username))
        store = VectorStore().store(users.collection(username))
        self.graph = build_graph(get_llm(), build_tools(store, self.docs.parents, get_llm(), observer=observer),
                                 _checkpointer, observer=observer)

    def _cfg(self, thread_id):
        return {"configurable": {"thread_id": thread_id}, "recursion_limit": config.RECURSION_LIMIT,
                "callbacks": _callbacks()}

    def ask(self, thread_id, question, history):
        cfg = self._cfg(thread_id)
        inputs = {
            "messages": [HumanMessage(question)],
            "documents": self.docs.list(),
            "previews": self.docs.previews(),
            "history": history,
        }
        snapshot = self.graph.get_state(cfg)
        if snapshot.next and "await_clarification" in snapshot.next:
            # the user is answering our clarifying question: add it and resume
            self.graph.update_state(cfg, inputs)
            result = self.graph.invoke(None, cfg)
        else:
            result = self.graph.invoke(inputs, cfg)
        msgs = result.get("messages") or []
        return str(msgs[-1].content) if msgs else "No answer was produced."

    def ask_stream(self, thread_id, question, history):
        """Yield ("step", node_name) as the graph moves, then ("answer", text)."""
        cfg = self._cfg(thread_id)
        inputs = {"messages": [HumanMessage(question)], "documents": self.docs.list(),
                  "previews": self.docs.previews(), "history": history}
        snapshot = self.graph.get_state(cfg)
        if snapshot.next and "await_clarification" in snapshot.next:
            self.graph.update_state(cfg, inputs)
            inputs = None
        for update in self.graph.stream(inputs, cfg, stream_mode="updates"):
            for node in update:
                if not node.startswith("__"):
                    yield "step", node
        msgs = self.graph.get_state(cfg).values.get("messages") or []
        yield "answer", str(msgs[-1].content) if msgs else "No answer was produced."

    def drop_thread(self, thread_id):
        try:
            _checkpointer.delete_thread(thread_id)
        except Exception:
            pass


class RAGRegistry:
    def __init__(self, users):
        self.users = users
        self._by_user = {}
        self._lock = threading.Lock()

    def get(self, username):
        with self._lock:
            if username not in self._by_user:
                self._by_user[username] = UserRAG(self.users, username)
            return self._by_user[username]
