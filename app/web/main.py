"""FastAPI app: auth pages, chat page, documents page, JSON + SSE APIs."""
import json
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool

from app import config
from app.core.chats import ChatService
from app.core.rag import RAGRegistry
from app.core.user_store import UserStore

users = UserStore()
registry = RAGRegistry(users)
chats = ChatService(users, registry)
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app = FastAPI(title="DocQA Agent")


def current_user(request: Request):
    return users.user_for(request.cookies.get(config.COOKIE_NAME))


def require_user(request: Request):
    user = current_user(request)
    if not user:
        raise HTTPException(401, "Sign in first.")
    return user


# ------------------------------------------------------------------- pages
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return RedirectResponse("/chat" if current_user(request) else "/login")


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request, error: str = ""):
    return templates.TemplateResponse(request, "auth.html", {"mode": "login", "error": error})


@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request, error: str = ""):
    return templates.TemplateResponse(request, "auth.html", {"mode": "register", "error": error})


@app.post("/login")
def login(username: str = Form(...), password: str = Form(...)):
    if not users.verify(username, password):
        return RedirectResponse("/login?error=Wrong+username+or+password", status_code=303)
    resp = RedirectResponse("/chat", status_code=303)
    resp.set_cookie(config.COOKIE_NAME, users.create_session(username.strip()), httponly=True,
                    samesite="lax", max_age=config.SESSION_TTL)
    return resp


@app.post("/register")
def register(username: str = Form(...), password: str = Form(...)):
    ok, msg = users.register(username, password)
    if not ok:
        return RedirectResponse(f"/register?error={msg.replace(' ', '+')}", status_code=303)
    return login(username, password)


@app.post("/logout")
def logout(request: Request):
    users.end_session(request.cookies.get(config.COOKIE_NAME))
    resp = RedirectResponse("/login", status_code=303)
    resp.delete_cookie(config.COOKIE_NAME)
    return resp


@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login")
    return templates.TemplateResponse(request, "chat.html", {"user": user})


@app.get("/documents", response_class=HTMLResponse)
def documents_page(request: Request):
    user = current_user(request)
    if not user:
        return RedirectResponse("/login")
    return templates.TemplateResponse(request, "documents.html", {"user": user})


# --------------------------------------------------------------- chat api
@app.get("/api/chats")
def list_chats(user=Depends(require_user)):
    return chats.list(user)


@app.post("/api/chats")
def new_chat(user=Depends(require_user)):
    return {"id": chats.create(user)}


@app.get("/api/chats/{cid}")
def get_chat(cid: str, user=Depends(require_user)):
    try:
        chat = chats.get(user, cid)
    except KeyError:
        raise HTTPException(404, "No such chat.")
    chats.select(user, cid)
    return {"id": cid, "title": chat["title"], "history": chat["history"]}


@app.patch("/api/chats/{cid}")
async def update_chat(cid: str, request: Request, user=Depends(require_user)):
    body = await request.json()
    if "title" in body:
        chats.rename(user, cid, body["title"])
    if body.get("toggle_pin"):
        chats.toggle_pin(user, cid)
    return {"ok": True}


@app.delete("/api/chats/{cid}")
def delete_chat(cid: str, user=Depends(require_user)):
    chats.delete(user, cid)
    return {"ok": True}


@app.post("/api/chats/{cid}/messages")
async def send_message(cid: str, request: Request, user=Depends(require_user)):
    text = str((await request.json()).get("text", "")).strip()
    if not text:
        raise HTTPException(400, "Message is empty.")
    return await run_in_threadpool(chats.send, user, cid, text)


@app.post("/api/chats/{cid}/stream")
async def stream_message(cid: str, request: Request, user=Depends(require_user)):
    text = str((await request.json()).get("text", "")).strip()
    if not text:
        raise HTTPException(400, "Message is empty.")

    def events():
        try:
            for kind, value in chats.send_stream(user, cid, text):
                yield f"event: {kind}\ndata: {json.dumps(value)}\n\n"
        except Exception as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"
        yield "event: done\ndata: \"\"\n\n"

    return StreamingResponse(iterate_in_threadpool(events()), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


# ---------------------------------------------------------- documents api
@app.get("/api/documents")
def list_documents(user=Depends(require_user)):
    docs = registry.get(user).docs
    return {"documents": docs.list(), "stats": docs.stats()}


@app.post("/api/documents")
async def upload(files: list[UploadFile] = File(...), user=Depends(require_user)):
    docs = registry.get(user).docs
    results, errors = [], []
    for f in files:
        try:
            results.append(await run_in_threadpool(docs.add, f.filename, await f.read()))
        except Exception as e:
            errors.append({"name": f.filename, "error": str(e)})
    if results:
        chats.bump_version(user)
    return {"added": results, "errors": errors}


@app.delete("/api/documents/{name}")
def remove(name: str, user=Depends(require_user)):
    docs = registry.get(user).docs
    if name not in docs.list():
        raise HTTPException(404, "No such document.")
    docs.remove(name)
    chats.bump_version(user)
    return {"ok": True}


@app.get("/health")
def health():
    return {"status": "ok"}
