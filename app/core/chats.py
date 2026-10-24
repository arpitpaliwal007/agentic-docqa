"""Chat list management and the document-version staleness rule.

Every upload or delete bumps the user's docs_version. Each chat remembers the
version its LangGraph thread was grounded on. When a stale chat gets a new
message, its thread is replaced with a fresh one, so the agent can't carry
retrieval context built from documents that no longer exist (or changed).
The visible history stays; only the agent's working memory resets.
"""
import time
import uuid

from app.core.user_store import new_chat


class ChatService:
    def __init__(self, users, registry):
        self.users = users
        self.registry = registry

    # ---- versioning
    def bump_version(self, username):
        state = self.users.load_state(username)
        state["docs_version"] += 1
        self.users.save_state(username, state)
        return state["docs_version"]

    def _refresh_if_stale(self, username, chat, version):
        if chat["docs_version"] == version:
            return False
        self.registry.get(username).drop_thread(chat["thread_id"])
        chat["thread_id"] = str(uuid.uuid4())
        chat["docs_version"] = version
        chat["history"].append({"role": "system", "content": "Documents changed; answers below use the current library.",
                                "meta": True})
        return True

    # ---- chat CRUD
    def list(self, username):
        s = self.users.load_state(username)
        ordered = sorted(s["order"], key=lambda c: (not s["chats"][c]["pinned"], s["order"].index(c)))
        return {"selected": s["selected"], "docs_version": s["docs_version"],
                "chats": [{"id": c, "title": s["chats"][c]["title"], "pinned": s["chats"][c]["pinned"],
                           "stale": s["chats"][c]["docs_version"] != s["docs_version"]} for c in ordered]}

    def get(self, username, chat_id):
        s = self.users.load_state(username)
        chat = s["chats"].get(chat_id)
        if chat is None:
            raise KeyError(chat_id)
        return chat

    def create(self, username):
        s = self.users.load_state(username)
        s["seq"] += 1
        cid = f"c{s['seq']}"
        s["chats"][cid] = new_chat(s["docs_version"])
        s["order"].insert(0, cid)
        s["selected"] = cid
        self.users.save_state(username, s)
        return cid

    def rename(self, username, chat_id, title):
        s = self.users.load_state(username)
        s["chats"][chat_id]["title"] = title.strip()[:80] or "Untitled"
        self.users.save_state(username, s)

    def toggle_pin(self, username, chat_id):
        s = self.users.load_state(username)
        s["chats"][chat_id]["pinned"] = not s["chats"][chat_id]["pinned"]
        self.users.save_state(username, s)

    def delete(self, username, chat_id):
        s = self.users.load_state(username)
        chat = s["chats"].pop(chat_id, None)
        if chat:
            self.registry.get(username).drop_thread(chat["thread_id"])
            s["order"].remove(chat_id)
        if not s["order"]:
            s["seq"] += 1
            cid = f"c{s['seq']}"
            s["chats"][cid] = new_chat(s["docs_version"])
            s["order"] = [cid]
        if s["selected"] not in s["chats"]:
            s["selected"] = s["order"][0]
        self.users.save_state(username, s)

    def select(self, username, chat_id):
        s = self.users.load_state(username)
        if chat_id in s["chats"]:
            s["selected"] = chat_id
            self.users.save_state(username, s)

    # ---- the actual message turn
    def send(self, username, chat_id, text):
        s = self.users.load_state(username)
        chat = s["chats"][chat_id]
        refreshed = self._refresh_if_stale(username, chat, s["docs_version"])
        visible = [h for h in chat["history"] if not h.get("meta")]
        answer = self.registry.get(username).ask(chat["thread_id"], text, visible)
        now = int(time.time())
        chat["history"] += [{"role": "user", "content": text, "ts": now},
                            {"role": "assistant", "content": answer, "ts": now}]
        if chat["title"] == "New chat":
            chat["title"] = text[:48] + ("…" if len(text) > 48 else "")
        self.users.save_state(username, s)
        return {"answer": answer, "refreshed": refreshed}

    def send_stream(self, username, chat_id, text):
        s = self.users.load_state(username)
        chat = s["chats"][chat_id]
        if self._refresh_if_stale(username, chat, s["docs_version"]):
            yield "refreshed", ""
        visible = [h for h in chat["history"] if not h.get("meta")]
        answer = ""
        for kind, value in self.registry.get(username).ask_stream(chat["thread_id"], text, visible):
            if kind == "answer":
                answer = value
            yield kind, value
        now = int(time.time())
        chat["history"] += [{"role": "user", "content": text, "ts": now},
                            {"role": "assistant", "content": answer, "ts": now}]
        if chat["title"] == "New chat":
            chat["title"] = text[:48] + ("…" if len(text) > 48 else "")
        self.users.save_state(username, s)
