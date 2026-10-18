import os
import sys
import json
import asyncio
import tempfile
import random
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Literal, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Depends, Request, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, EmailStr, field_validator
from mcp import StdioServerParameters
from dotenv import load_dotenv
from groq import Groq
import asyncpg
import bcrypt
import uvicorn
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from exponent_server_sdk import PushClient, PushMessage
import httpx 

from shared import ctx, chat_statuses, MCPSessionPool
from client import (
    get_ollama_tools, generate_user_profile, generate_user_suggestions,
    build_push_recommendation, decay_profile, app as graph_app,
)
from auth import (
    CurrentUser, create_access_token, get_current_user,
    login_rate_limiter, signup_rate_limiter, client_ip,
)
from logger import logger

# --- KATALOG ROUTER'I İÇERİ AKTARILIYOR ---
from catalog import router as catalog_router

load_dotenv()

_REQUIRED_ENV_VARS = [
    "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST",
    "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
    "AUTH_KEY",          
    "WHISPER_API_KEY",   
]

def _validate_env() -> None:
    missing = [k for k in _REQUIRED_ENV_VARS if not os.getenv(k)]
    if missing:
        raise RuntimeError(
            "Eksik ortam değişkenleri: " + ", ".join(missing) +
            "\n.env dosyanı veya deployment ortamının env değişkenlerini kontrol et."
        )

_validate_env()

client = Groq(api_key=os.environ.get("WHISPER_API_KEY"))

DB_POOL_MIN_SIZE = int(os.getenv("DB_POOL_MIN_SIZE", "2"))
DB_POOL_MAX_SIZE = int(os.getenv("DB_POOL_MAX_SIZE", "10"))
MCP_POOL_SIZE = int(os.getenv("MCP_POOL_SIZE", "3"))

# --- YENİ: KULLANICI BAZLI KİLİT SÖZLÜĞÜ (LOCKS) ---
user_workflow_locks = {}

def get_user_lock(user_id: int) -> asyncio.Lock:
    if user_id not in user_workflow_locks:
        user_workflow_locks[user_id] = asyncio.Lock()
    return user_workflow_locks[user_id]


class FavoriteRequest(BaseModel):
    movie_id: str
    title: str
    genres: str = None
    director: str = None
    cast_members: str = None
    poster_url: str = None
    imdb_rating: str = None
    trailer_url: str = None

class TranscriptionResponse(BaseModel):
    text: str
    success: bool

class TokenRequest(BaseModel):
    token: str

MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_MB", "15")) * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}

def remove_file(path: str):
    if os.path.exists(path):
        os.remove(path)

def _mcp_env() -> dict:
    keys = ["AUTH_KEY", "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"]
    return {k: os.getenv(k) for k in keys if os.getenv(k)}

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS recommendation_feedback (
    id          BIGSERIAL PRIMARY KEY,
    user_id     BIGINT      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    movie_id    TEXT        NOT NULL,
    title       TEXT,
    source      TEXT        NOT NULL CHECK (source IN ('chat', 'push')),
    feedback    SMALLINT    CHECK (feedback IN (-1, 1)),   -- 1 = like, -1 = dislike, NULL = sadece gösterildi
    session_id  TEXT,
    shown_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    feedback_at TIMESTAMPTZ,
    UNIQUE (user_id, movie_id, source)
);
CREATE INDEX IF NOT EXISTS idx_rec_feedback_user ON recommendation_feedback (user_id, feedback);
-- öneri kaynağı (similar / collaborative / persona_genre): hangi hattın işe yaradığını ölçmek için
ALTER TABLE recommendations ADD COLUMN IF NOT EXISTS source TEXT;
"""

async def _ensure_schema() -> None:
    async with ctx.db_pool.acquire() as conn:
        await conn.execute(_SCHEMA_SQL)

@asynccontextmanager
async def lifespan(app: FastAPI):
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=_mcp_env(),
    )
    scheduler = AsyncIOScheduler()

    logger.info("DB havuzu, MCP session havuzu ve Scheduler başlatılıyor...")
    try:
        ctx.db_pool = await asyncpg.create_pool(
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
        )
        logger.info(f"Postgres pool hazır (min={DB_POOL_MIN_SIZE}, max={DB_POOL_MAX_SIZE}).")
        await _ensure_schema()

        mcp_pool = MCPSessionPool(pool_size=MCP_POOL_SIZE)
        await mcp_pool.start(server_params, get_ollama_tools)
        ctx.session = mcp_pool
        ctx.tools = mcp_pool.tools
        logger.info(f"{len(ctx.tools)} adet tool, {MCP_POOL_SIZE} paralel MCP session ile yüklendi.")

        scheduler.add_job(send_random_notifications, 'interval', minutes=30)
        scheduler.start()
        logger.info("Zamanlayıcı (Scheduler) başlatıldı.")

        yield
    finally:
        if scheduler.running:
            scheduler.shutdown()
        logger.info("Sunucu kapatılıyor, kaynaklar temizleniyor...")
        if ctx.session:
            await ctx.session.aclose()
        if ctx.db_pool:
            await ctx.db_pool.close()
        await ctx.exit_stack.aclose()

app = FastAPI(title="Movie Explorer AI API", lifespan=lifespan)

origins = [
    "https://cinewave-react-website.vercel.app",
    "http://localhost:3000", # Yerel geliştirmeleriniz için
    "http://localhost:5173", # Vite kullanıyorsanız
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- REACT FİLM SİTESİ İÇİN KATALOG ENDPOINT'LERİ ---
# /catalog/... (home, browse, search, detay) burada dahil ediliyor.
app.include_router(catalog_router)

@app.get("/health")
async def health():
    checks = {"db": False, "mcp": False}

    if ctx.db_pool is not None:
        try:
            async with ctx.db_pool.acquire() as conn:
                await conn.fetchval("SELECT 1")
            checks["db"] = True
        except Exception:
            logger.error("Health check: DB erişilemedi", exc_info=True)

    checks["mcp"] = bool(ctx.session) and len(ctx.tools or []) > 0

    healthy = all(checks.values())
    status_code = 200 if healthy else 503
    return JSONResponse(status_code=status_code, content={"status": "ok" if healthy else "degraded", "checks": checks})

# --- KULLANICI / YETKİLENDİRME (AUTH) İŞLEMLERİ ---

class UserSignup(BaseModel):
    username: str
    email: EmailStr
    password: str

    @field_validator('password')
    @classmethod
    def truncate_password(cls, v: str) -> str:
        return v[:71]

class UserLogin(BaseModel):
    username: str
    password: str

class UpdateProfileRequest(BaseModel):
    new_username: str

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

def hash_password(password: str):
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password, hashed_password):
    password_byte = plain_password.encode('utf-8')
    hashed_byte = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_byte, hashed_byte)

@app.post("/signup")
async def signup(user: UserSignup, request: Request):
    signup_rate_limiter.check(client_ip(request))
    try:
        async with ctx.db_pool.acquire() as conn:
            async with conn.transaction():
                existing = await conn.fetchrow(
                    "SELECT id FROM users WHERE username = $1 OR email = $2",
                    user.username, user.email
                )
                if existing:
                    raise HTTPException(status_code=400, detail="Kullanıcı adı veya email zaten kayıtlı.")

                hashed_pwd = hash_password(user.password)
                new_user_id = await conn.fetchval(
                    "INSERT INTO users (username, email, password_hash) VALUES ($1, $2, $3) RETURNING id",
                    user.username, user.email, hashed_pwd
                )
                await conn.execute("INSERT INTO user_profiles (user_id) VALUES ($1)", new_user_id)

        return {"message": "Kayıt başarılı! Giriş yapabilirsiniz."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Kayıt (signup) hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Kayıt işlemi sırasında bir hata oluştu.")

@app.post("/login")
async def login(user: UserLogin, request: Request):
    login_rate_limiter.check(client_ip(request))
    try:
        async with ctx.db_pool.acquire() as conn:
            record = await conn.fetchrow("SELECT id, password_hash FROM users WHERE username = $1", user.username)

        if record and verify_password(user.password, record["password_hash"]):
            access_token = create_access_token(user_id=record["id"], username=user.username)
            return {
                "status": "success",
                "access_token": access_token,
                "token_type": "bearer",
                "user_id": record["id"],
                "username": user.username,
                "message": f"Hoş geldin {user.username}!"
            }
        raise HTTPException(status_code=401, detail="Hatalı kullanıcı adı veya şifre.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Giriş (login) hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Giriş işlemi sırasında bir hata oluştu.")

@app.post("/update-profile")
async def update_profile(req: UpdateProfileRequest, current_user: CurrentUser = Depends(get_current_user)):
    try:
        async with ctx.db_pool.acquire() as conn:
            async with conn.transaction():
                taken = await conn.fetchrow("SELECT id FROM users WHERE username = $1 AND id != $2", req.new_username, current_user.user_id)
                if taken:
                    raise HTTPException(status_code=400, detail="Bu kullanıcı adı zaten alınmış.")
                await conn.execute("UPDATE users SET username = $1 WHERE id = $2", req.new_username, current_user.user_id)
        return {"status": "success", "username": req.new_username}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Profil güncellenemedi.")

@app.post("/change-password")
async def change_password(req: ChangePasswordRequest, current_user: CurrentUser = Depends(get_current_user)):
    try:
        async with ctx.db_pool.acquire() as conn:
            record = await conn.fetchrow("SELECT password_hash FROM users WHERE id = $1", current_user.user_id)
            if not record or not verify_password(req.old_password, record["password_hash"]):
                raise HTTPException(status_code=400, detail="Mevcut şifreniz hatalı.")

            hashed_pwd = hash_password(req.new_password)
            await conn.execute("UPDATE users SET password_hash = $1 WHERE id = $2", hashed_pwd, current_user.user_id)
        return {"status": "success", "message": "Şifreniz başarıyla değiştirildi."}
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=500, detail="Şifre değiştirilemedi.")


# --- SOHBET / ÖNERİ MOTORU ---

class ChatRequest(BaseModel):
    prompt: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    tool_calls: list = []
    tool_results: list = []

def _compact_content(role: str, content: str) -> str:
    if role != "assistant":
        return content
    try:
        data = json.loads(content)
    except Exception:
        return content
    if not isinstance(data, dict) or data.get("type") != "movie_list":
        return content
    titles = ", ".join(m.get("Film", "") for m in data.get("movies", []) if m.get("Film"))
    text = data.get("text", "")
    return f"{text} (Önerilen filmler: {titles})" if titles else text

async def get_recent_chats(user_id: int, limit: int = 10):
    async with ctx.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT role, content FROM chat_history WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2",
            user_id, limit
        )
    return [{"role": r["role"], "content": _compact_content(r["role"], r["content"])} for r in reversed(rows)]

async def get_user_persona(user_id: int):
    async with ctx.db_pool.acquire() as conn:
        row = await conn.fetchrow("SELECT persona_summary FROM user_profiles WHERE user_id = $1", user_id)
    return row[0] if row else None

async def get_blocked_and_seen(user_id: int):
    """blocked: dislike edilenler (sert engel). seen: favoriler + gösterilmiş her şey (skorda hafif ceza)."""
    async with ctx.db_pool.acquire() as conn:
        fb = await conn.fetch("SELECT movie_id, feedback FROM recommendation_feedback WHERE user_id = $1", user_id)
        favs = await conn.fetch("SELECT movie_id FROM favorites WHERE user_id = $1", user_id)
    blocked = [str(r["movie_id"]) for r in fb if r["feedback"] == -1]
    seen = [str(r["movie_id"]) for r in fb] + [str(r["movie_id"]) for r in favs]
    return blocked, seen

@app.post("/chat")
async def chat(
    request: ChatRequest,
    background_tasks: BackgroundTasks,
    current_user: CurrentUser = Depends(get_current_user),
):
    if not ctx.session:
        raise HTTPException(status_code=503, detail="MCP Session hazır değil.")

    user_persona = await get_user_persona(current_user.user_id)
    chat_history = await get_recent_chats(current_user.user_id, limit=10)
    try:
        blocked_ids, seen_ids = await get_blocked_and_seen(current_user.user_id)
    except Exception:
        logger.warning("blocked/seen listesi alınamadı; filtre atlanıyor", exc_info=True)
        blocked_ids, seen_ids = [], []

    initial_state = {
        "prompt": request.prompt,
        "persona": user_persona or "{}",
        "messages": chat_history + [{"role": "user", "content": request.prompt}],
        "tools": ctx.tools,
        "intent": "",
        "final_output": "",
        "tool_calls": [],
        "tool_results": [],
        "session_id": request.session_id,
        "user_id": current_user.user_id,
        "blocked_ids": blocked_ids,
        "seen_ids": seen_ids,
    }

    try:
        chat_statuses[request.session_id] = "🔍 Sorgu analiz ediliyor..."
        result = await graph_app.ainvoke(initial_state)
        answer = result.get("final_output", "Üzgünüm, şu an öneri yapamıyorum.")
        tool_calls = result.get("tool_calls", [])
        tool_results = result.get("tool_results", [])
    except Exception as e:
        logger.error("Chat graph çalıştırma hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Şu an öneri üretilemedi, lütfen tekrar deneyin.")
    finally:
        chat_statuses.pop(request.session_id, None)

    async with ctx.db_pool.acquire() as conn:
        await conn.execute("INSERT INTO chat_history (user_id, role, content, session_id) VALUES ($1, $2, $3, $4)", current_user.user_id, "user", request.prompt, request.session_id)
        await conn.execute("INSERT INTO chat_history (user_id, role, content, session_id) VALUES ($1, $2, $3, $4)", current_user.user_id, "assistant", answer, request.session_id)
        msg_count = await conn.fetchval("SELECT COUNT(*) FROM chat_history WHERE user_id = $1", current_user.user_id)
    
    await _log_chat_impressions(current_user.user_id, request.session_id, answer)

    # Tetikleyici, her 5 mesajda çalışır. Ancak artık görev içinde Kilit mekanizması var.
    if msg_count > 0 and msg_count % 5 == 0:
        background_tasks.add_task(profile_and_recommendation_workflow, current_user.user_id)

    return {"answer": answer, "tool_calls": tool_calls, "tool_results": tool_results}

# --- ÖNERİ GERİ BİLDİRİMİ (LIKE / DISLIKE) ---

class FeedbackRequest(BaseModel):
    movie_id: str
    feedback: Literal["like", "dislike", "clear"]
    source: Literal["chat", "push"] = "chat"
    title: Optional[str] = None
    session_id: Optional[str] = None

_FEEDBACK_VALUE = {"like": 1, "dislike": -1, "clear": None}

async def _log_chat_impressions(user_id: int, session_id: str, answer: str) -> None:
    """Sohbette gösterilen filmleri 'gösterildi' olarak kaydeder (like oranının paydası). Asla chat'i bozmaz."""
    try:
        data = json.loads(answer)
        if not isinstance(data, dict) or data.get("type") != "movie_list":
            return
        rows = [(user_id, str(m["movie_id"]), m.get("Film"), session_id)
                for m in data.get("movies", []) if m.get("movie_id")]
        if not rows:
            return
        async with ctx.db_pool.acquire() as conn:
            await conn.executemany(
                """INSERT INTO recommendation_feedback (user_id, movie_id, title, source, session_id)
                   VALUES ($1, $2, $3, 'chat', $4)
                   ON CONFLICT (user_id, movie_id, source) DO NOTHING""",
                rows,
            )
    except Exception:
        logger.warning("Impression kaydı başarısız", exc_info=True)

@app.post("/feedback")
async def submit_feedback(req: FeedbackRequest, current_user: CurrentUser = Depends(get_current_user)):
    value = _FEEDBACK_VALUE[req.feedback]
    try:
        async with ctx.db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO recommendation_feedback
                       (user_id, movie_id, title, source, feedback, session_id, feedback_at)
                   VALUES ($1, $2, $3, $4, $5::smallint, $6, CASE WHEN $5::smallint IS NULL THEN NULL ELSE now() END)
                   ON CONFLICT (user_id, movie_id, source) DO UPDATE
                   SET feedback    = EXCLUDED.feedback,
                       feedback_at = EXCLUDED.feedback_at,
                       title       = COALESCE(recommendation_feedback.title, EXCLUDED.title)""",
                current_user.user_id, req.movie_id, req.title, req.source, value, req.session_id,
            )
        return {"status": "success", "feedback": req.feedback}
    except Exception:
        logger.error("Feedback kaydı hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Geri bildirim kaydedilemedi.")

@app.get("/feedback/ids")
async def get_feedback_ids(current_user: CurrentUser = Depends(get_current_user)):
    """UI'ın sayfa yenilenince buton durumlarını geri yüklemesi için."""
    async with ctx.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT movie_id, feedback FROM recommendation_feedback WHERE user_id = $1 AND feedback IS NOT NULL",
            current_user.user_id,
        )
    return {
        "liked": [r["movie_id"] for r in rows if r["feedback"] == 1],
        "disliked": [r["movie_id"] for r in rows if r["feedback"] == -1],
    }

@app.get("/feedback/stats")
async def get_feedback_stats(current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """SELECT source,
                      COUNT(*)                              AS shown,
                      COUNT(*) FILTER (WHERE feedback = 1)  AS likes,
                      COUNT(*) FILTER (WHERE feedback = -1) AS dislikes
               FROM recommendation_feedback WHERE user_id = $1 GROUP BY source""",
            current_user.user_id,
        )
    out = {}
    for r in rows:
        rated = r["likes"] + r["dislikes"]
        out[r["source"]] = {
            "shown": r["shown"], "likes": r["likes"], "dislikes": r["dislikes"],
            "like_rate": round(r["likes"] / rated, 3) if rated else None,   # oy verenler içinde
            "like_per_shown": round(r["likes"] / r["shown"], 3) if r["shown"] else None,
        }
    return out

@app.get("/sessions")
async def get_sessions(current_user: CurrentUser = Depends(get_current_user), limit: int = Query(default=20, ge=1, le=100), offset: int = Query(default=0, ge=0)):
    try:
        async with ctx.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT session_id,
                       MIN(created_at) as created,
                       (SELECT content FROM chat_history ch2 WHERE ch2.session_id = ch1.session_id AND role='user' ORDER BY created_at ASC LIMIT 1) as title
                FROM chat_history ch1
                WHERE user_id = $1
                GROUP BY session_id
                ORDER BY created DESC
                LIMIT $2 OFFSET $3
                """,
                current_user.user_id, limit, offset
            )
            total = await conn.fetchval("SELECT COUNT(DISTINCT session_id) FROM chat_history WHERE user_id = $1", current_user.user_id)
        sessions = [{"session_id": r["session_id"], "title": r["title"] if r["title"] else "Yeni Sohbet"} for r in rows]
        return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}
    except Exception:
        raise HTTPException(status_code=500, detail="Oturumlar alınamadı.")

@app.get("/chat/suggestions")
async def get_chat_suggestions(current_user: CurrentUser = Depends(get_current_user)):
    default_suggestions = ["Bilim kurgu filmi öner", "Nolan filmi öner", "Tim Burton filmi öner"]
    try:
        history_list = await get_recent_chats(current_user.user_id, limit=10)
        favorites_dict = await get_user_favorites(current_user.user_id)
        fav_items = [f"{f['Film']} ({f['Türler'] or ''})" for f in favorites_dict.get("favorites", [])]

        if not history_list and not fav_items:
            return {"suggestions": default_suggestions}

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_list])
        favorites_text = ", ".join(fav_items) if fav_items else "Henüz favori eklenmemiş."
        suggestions = await generate_user_suggestions(history_text, favorites_text)
        return {"suggestions": suggestions}
    except Exception:
        return {"suggestions": default_suggestions}

@app.get("/chat/{session_id}")
async def get_chat_history(session_id: str, current_user: CurrentUser = Depends(get_current_user), limit: int = Query(default=100), offset: int = Query(default=0)):
    try:
        async with ctx.db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT role, content FROM chat_history WHERE user_id = $1 AND session_id = $2 ORDER BY created_at ASC LIMIT $3 OFFSET $4", current_user.user_id, session_id, limit, offset)
            total = await conn.fetchval("SELECT COUNT(*) FROM chat_history WHERE user_id = $1 AND session_id = $2", current_user.user_id, session_id)
        history = [{"role": r["role"], "content": r["content"]} for r in rows]
        return {"history": history, "total": total, "limit": limit, "offset": offset}
    except Exception:
        raise HTTPException(status_code=500, detail="Sohbet geçmişi alınamadı.")



@app.delete("/chat/{session_id}")
async def delete_session(session_id: str, current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        await conn.execute("DELETE FROM chat_history WHERE user_id = $1 AND session_id = $2", current_user.user_id, session_id)
    return {"status": "success"}



@app.get("/chat/status/{session_id}")
async def get_chat_status(session_id: str, current_user: CurrentUser = Depends(get_current_user)):
    return {"status": chat_statuses.get(session_id, "")}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...), current_user: CurrentUser = Depends(get_current_user)):
    temp_file_path = None
    try:
        file_ext = os.path.splitext(audio.filename or "")[1].lower() or ".wav"
        if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(status_code=400, detail="Desteklenmeyen ses formatı.")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            total = 0
            while chunk := await audio.read(1024 * 1024): 
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise HTTPException(status_code=413, detail="Dosya çok büyük.")
                temp_file.write(chunk)
            temp_file_path = temp_file.name

        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(model="whisper-large-v3", file=audio_file, language="tr")

        return TranscriptionResponse(text=transcription.text.strip(), success=True)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


# --- FAVORİLER ---

async def get_user_favorites(user_id: int, limit: int | None = None, offset: int = 0):
    try:
        async with ctx.db_pool.acquire() as conn:
            if limit is not None:
                rows = await conn.fetch("SELECT * FROM favorites WHERE user_id = $1 ORDER BY created_at DESC LIMIT $2 OFFSET $3", user_id, limit, offset)
                total = await conn.fetchval("SELECT COUNT(*) FROM favorites WHERE user_id = $1", user_id)
            else:
                rows = await conn.fetch("SELECT * FROM favorites WHERE user_id = $1 ORDER BY created_at DESC", user_id)
                total = len(rows)

        favorites = [{
            "movie_id": r["movie_id"], "Film": r["title"], "Türler": r["genres"],
            "Director": r["director"], "Cast": r["cast_members"], "Poster": r["poster_url"],
            "TMDB Puanı": r["imdb_rating"], "Fragman": r["trailer_url"], 
        } for r in rows]

        return {"favorites": favorites, "total": total}
    except Exception:
        raise HTTPException(status_code=500, detail="Favoriler alınamadı.")

@app.post("/favorites")
async def add_favorite(fav: FavoriteRequest, current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        await conn.execute(
            """INSERT INTO favorites (user_id, movie_id, title, genres, director, cast_members, poster_url, imdb_rating, trailer_url)
               VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9) ON CONFLICT (user_id, movie_id) DO NOTHING""",
            current_user.user_id, fav.movie_id, fav.title, fav.genres, fav.director, fav.cast_members, fav.poster_url, fav.imdb_rating, fav.trailer_url 
        )
    return {"status": "success"}

@app.delete("/favorites/{movie_id}")
async def delete_favorite(movie_id: str, current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        await conn.execute("DELETE FROM favorites WHERE user_id = $1 AND movie_id = $2", current_user.user_id, movie_id)
    return {"status": "success"}

@app.get("/favorites")
async def get_favorites(current_user: CurrentUser = Depends(get_current_user), limit: int = Query(default=20), offset: int = Query(default=0)):
    return await get_user_favorites(current_user.user_id, limit=limit, offset=offset)

@app.get("/favorites/ids")
async def get_favorite_ids(current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT movie_id FROM favorites WHERE user_id = $1", current_user.user_id)
    return {"favorite_ids": [r["movie_id"] for r in rows]}


# --- BİLDİRİM VE PROFİLLEME (ARKA PLAN GÖREVLERİ) ---

@app.post("/update-push-token")
async def update_push_token(req: TokenRequest, current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        await conn.execute("UPDATE users SET expo_push_token = $1 WHERE id = $2", req.token, current_user.user_id)
    return {"status": "success"}

@app.get("/recommendations")
async def get_user_recommendations(current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        rows = await conn.fetch("SELECT id, movie_id, title, message, created_at, is_read FROM recommendations WHERE user_id = $1 ORDER BY created_at DESC LIMIT 50", current_user.user_id)
    return [{"id": r["id"], "movie_id": r["movie_id"], "title": r["title"], "body": r["message"], "date": r["created_at"], "is_read": r["is_read"]} for r in rows]

@app.patch("/recommendations/read")
async def mark_recommendations_read(current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        await conn.execute("UPDATE recommendations SET is_read = true WHERE user_id = $1", current_user.user_id)
    return {"status": "success"}

async def profile_update_task(user_id: int):
    try:
        # 1. Eski profil + son güncellemeden bu yana geçen gün
        async with ctx.db_pool.acquire() as conn:
            row = await conn.fetchrow(
                """SELECT persona_summary,
                          COALESCE(EXTRACT(EPOCH FROM (now() - last_updated)) / 86400.0, 0) AS days
                   FROM user_profiles WHERE user_id = $1""", user_id)
        raw_profile = (row["persona_summary"] if row and row["persona_summary"] else "{}")
        days = min(float(row["days"]) if row else 0.0, 365.0)

        # 2. Zamana bağlı yaşlandırma KODDA (LLM'e bırakılmaz)
        aged_profile = json.dumps(decay_profile(raw_profile, days), ensure_ascii=False)

        # 3. Sohbet geçmişi
        history_list = await get_recent_chats(user_id, limit=10)
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_list])

        # 4. YALNIZCA son güncellemeden sonraki favori ve geri bildirimler (aksi halde eski favoriler
        #    her seferinde yeniden güçlenir ve decay etkisiz kalır)
        async with ctx.db_pool.acquire() as conn:
            new_favs = await conn.fetch(
                """SELECT title, genres FROM favorites
                   WHERE user_id = $1
                     AND created_at > COALESCE((SELECT last_updated FROM user_profiles WHERE user_id = $1), 'epoch')
                   ORDER BY created_at DESC LIMIT 30""", user_id)
            feedbacks = await conn.fetch(
                """SELECT title, feedback FROM recommendation_feedback
                   WHERE user_id = $1 AND feedback IS NOT NULL
                     AND feedback_at > COALESCE((SELECT last_updated FROM user_profiles WHERE user_id = $1), 'epoch')
                   ORDER BY feedback_at DESC LIMIT 30""", user_id)
        favorites_text = (", ".join(f"{f['title']} ({f['genres'] or ''})" for f in new_favs)
                          if new_favs else "Yeni favori yok.")
        feedback_text = ("\n".join(f"{f['title']}: {'Beğendi (Like)' if f['feedback'] == 1 else 'Beğenmedi (Dislike)'}"
                                   for f in feedbacks) if feedbacks else "Yeni geri bildirim yok.")

        combined_interactions = f"SOHBETLER:\n{history_text}\n\nGERİ BİLDİRİMLER (LIKE/DISLIKE):\n{feedback_text}"

        # 5. Yeni profili üret (LLM yalnızca yeni sinyalleri ekler; çıktı şemaya zorlanır)
        new_profile_json = await generate_user_profile(combined_interactions, favorites_text, aged_profile)

        # 6. Veritabanına kaydet
        async with ctx.db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE user_profiles SET persona_summary = $1, last_updated = CURRENT_TIMESTAMP WHERE user_id = $2",
                new_profile_json, user_id
            )
    except Exception as e:
        logger.error(f"Profilleme hatası (user_id={user_id})", exc_info=True)

def _parse_cards(result) -> list:
    try:
        data = json.loads(result.content[0].text)
        return [c for c in data if isinstance(c, dict)] if isinstance(data, list) else []
    except Exception:
        return []

async def get_movie_card_by_title(title: str):
    if not title: return None
    try:
        result = await ctx.session.call_tool("get_movie_card", {"title": title, "year": 0})
        cards = _parse_cards(result)
        return cards[0] if cards else None
    except Exception:
        logger.warning(f"Film kartı alınamadı: {title!r}", exc_info=True)
        return None

async def get_movie_card_by_id(movie_id):
    """Başlık araması yerine TMDB id'siyle tam kart (aynı adlı filmlerin karışmasını önler)."""
    try:
        mid = int(movie_id)
    except (TypeError, ValueError):
        return None
    try:
        result = await ctx.session.call_tool("enrich_movies", {"movie_ids": [mid]})
        cards = _parse_cards(result)
        return cards[0] if cards else None
    except Exception:
        logger.warning(f"Film kartı (id) alınamadı: {movie_id!r}", exc_info=True)
        return None

async def generate_and_save_recommendation_task(user_id: int):
    try:
        persona = await get_user_persona(user_id)
        if not persona:
            logger.info(f"Öneri atlandı (user={user_id}): persona yok")
            return

        favs = (await get_user_favorites(user_id)).get("favorites", [])
        async with ctx.db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT movie_id, title FROM recommendations WHERE user_id = $1", user_id)
            # sohbette/push'ta gösterilmiş HER film (beğenilen, beğenilmeyen, nötr) tekrar önerilmesin
            shown = await conn.fetch(
                "SELECT movie_id, title FROM recommendation_feedback WHERE user_id = $1", user_id)
            liked_rows = await conn.fetch(
                """SELECT movie_id, title FROM recommendation_feedback
                   WHERE user_id = $1 AND feedback = 1 ORDER BY feedback_at DESC LIMIT 10""", user_id)

        # Favoriler + geçmiş öneriler + gösterilenler (dislike dahil): id VE başlık ile dışla
        exclude_ids = ({str(r["movie_id"]) for r in rows if r["movie_id"]} | {str(f["movie_id"]) for f in favs}
                       | {str(s["movie_id"]) for s in shown if s["movie_id"]})
        exclude_titles = ({r["title"].lower().strip() for r in rows if r["title"]}
                          | {f["Film"].lower().strip() for f in favs if f["Film"]}
                          | {s["title"].lower().strip() for s in shown if s["title"]})
        liked = [{"movie_id": str(r["movie_id"]), "Film": r["title"]} for r in liked_rows if r["title"]]

        rec = await build_push_recommendation(user_id, persona, favs, exclude_ids, exclude_titles, liked=liked)
        if not rec:
            return

        async with ctx.db_pool.acquire() as conn:
            await conn.execute(
                """INSERT INTO recommendations (user_id, movie_id, title, message, reason, source)
                   SELECT $1::bigint, $2::text, $3::text, $4::text, $5::text, $6::text
                   WHERE NOT EXISTS (SELECT 1 FROM recommendations WHERE user_id = $1::bigint AND movie_id = $2::text)""",
                user_id, rec["movie_id"], rec["title"], rec["message"], rec["reason"], rec.get("source"),
            )
        logger.info(f"Öneri kaydedildi (user={user_id}): {rec['title']!r} ({rec['reason']})")
    except Exception:
        logger.error(f"Öneri üretme hatası (user={user_id})", exc_info=True)

# --- KİLİT MEKANİZMASI ENTEGRE EDİLMİŞ İŞ AKIŞI ---
async def profile_and_recommendation_workflow(user_id: int):
    lock = get_user_lock(user_id)
    
    # Eğer kilitliyse (yani bu kullanıcının önceki profilleme görevi hala sürüyorsa) çıkış yap.
    if lock.locked():
        logger.info(f"Kullanıcı (id={user_id}) için profilleme görevi zaten çalışıyor, Kilit atlandı.")
        return

    async with lock:
        try:
            await profile_update_task(user_id)
            await generate_and_save_recommendation_task(user_id)
        except Exception:
            logger.error(f"Workflow çalıştırılırken hata (user={user_id})", exc_info=True)

async def send_random_notifications():
    try:
        async with ctx.db_pool.acquire() as conn:
            records = await conn.fetch(
                """
                SELECT DISTINCT ON (r.user_id) r.id, r.user_id, r.movie_id, r.title, r.message, u.expo_push_token 
                FROM recommendations r JOIN users u ON r.user_id = u.id
                WHERE r.is_pushed = false AND u.expo_push_token IS NOT NULL
                ORDER BY r.user_id, r.created_at DESC LIMIT 50
                """
            )
            for row in records:
                try:
                    card = await get_movie_card_by_id(row["movie_id"]) or await get_movie_card_by_title(row["title"])
                    msg = PushMessage(
                        to=row["expo_push_token"],
                        title="Senin İçin Bir Film Buldum 🍿",
                        body=row["message"],
                        data={"type": "movie_recommendation", "movies": [card] if card else []},
                    )
                    await asyncio.to_thread(PushClient().publish, msg)
                    await conn.execute("UPDATE recommendations SET is_pushed = true WHERE id = $1", row["id"])
                except Exception:
                    logger.error(f"Push gönderilemedi (rec_id={row['id']})", exc_info=True)
    except Exception:
        logger.error("Bildirim döngüsü hatası", exc_info=True)

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "3000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )