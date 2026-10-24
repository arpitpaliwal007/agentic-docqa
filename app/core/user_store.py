"""Users, sessions and per-user chat state, all as JSON on disk.

Each user gets their own markdown dir, parent store and Qdrant collection,
so nobody can retrieve anyone else's documents.
"""
import hashlib
import json
import secrets
import threading
import time
import uuid

import bcrypt

from app import config


def new_chat(docs_version):
    return {
        "title": "New chat",
        "thread_id": str(uuid.uuid4()),
        "docs_version": docs_version,   # corpus version this thread was grounded on
        "history": [],
        "pinned": False,
        "created_at": int(time.time()),
    }


class UserStore:
    def __init__(self, root=None):
        self.root = config.DATA_DIR if root is None else root
        self.root.mkdir(parents=True, exist_ok=True)
        self.users_f = self.root / "users.json"
        self.sessions_f = self.root / "sessions.json"
        (self.root / "chats").mkdir(exist_ok=True)
        (self.root / "users").mkdir(exist_ok=True)
        self._lock = threading.Lock()

    # ---- json helpers
    def _read(self, f, default):
        try:
            return json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return default

    def _write(self, f, data):
        tmp = f.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(f)

    @staticmethod
    def uid(username):
        return hashlib.sha256(username.encode()).hexdigest()[:16]

    # ---- accounts
    def register(self, username, password):
        username = (username or "").strip()
        if not 3 <= len(username) <= 32 or " " in username:
            return False, "Username must be 3-32 characters with no spaces."
        if len(password or "") < 6:
            return False, "Password must be at least 6 characters."
        with self._lock:
            users = self._read(self.users_f, {})
            if username in users:
                return False, "That username is taken."
            users[username] = {
                "hash": bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode(),
                "created_at": int(time.time()),
            }
            self._write(self.users_f, users)
        return True, "Account created."

    def verify(self, username, password):
        user = self._read(self.users_f, {}).get((username or "").strip())
        if not user:
            return False
        try:
            return bcrypt.checkpw((password or "").encode(), user["hash"].encode())
        except Exception:
            return False

    # ---- sessions
    def create_session(self, username):
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with self._lock:
            sessions = {t: s for t, s in self._read(self.sessions_f, {}).items() if s["exp"] > now}
            sessions[token] = {"user": username, "exp": now + config.SESSION_TTL}
            self._write(self.sessions_f, sessions)
        return token

    def user_for(self, token):
        if not token:
            return None
        s = self._read(self.sessions_f, {}).get(token)
        return s["user"] if s and s["exp"] > time.time() else None

    def end_session(self, token):
        with self._lock:
            sessions = self._read(self.sessions_f, {})
            if sessions.pop(token, None):
                self._write(self.sessions_f, sessions)

    # ---- chat state
    def _chat_f(self, username):
        return self.root / "chats" / f"{self.uid(username)}.json"

    def load_state(self, username):
        state = self._read(self._chat_f(username), None)
        if not state:
            cid = "c1"
            state = {"docs_version": 0, "selected": cid, "order": [cid], "chats": {cid: new_chat(0)}, "seq": 1}
            self.save_state(username, state)
        return state

    def save_state(self, username, state):
        with self._lock:
            self._write(self._chat_f(username), state)

    # ---- per-user storage locations
    def user_dir(self, username):
        d = self.root / "users" / self.uid(username)
        (d / "markdown").mkdir(parents=True, exist_ok=True)
        (d / "uploads").mkdir(exist_ok=True)
        (d / "parents").mkdir(exist_ok=True)
        return d

    def collection(self, username):
        return f"children_{self.uid(username)}"
