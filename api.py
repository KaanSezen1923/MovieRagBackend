import os
import sys
import json
import asyncio
import tempfile
import random
from contextlib import AsyncExitStack, asynccontextmanager
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
    generate_push_message, app as graph_app,
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

    initial_state = {
        "prompt": request.prompt,
        "persona": user_persona or "Film sever bir kullanıcı.",
        "messages": chat_history + [{"role": "user", "content": request.prompt}],
        "tools": ctx.tools,
        "intent": "",
        "final_output": "",
        "tool_calls": [],
        "tool_results": [],
        "session_id": request.session_id,
        "user_id": current_user.user_id
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
    
    if msg_count > 0 and msg_count % 5 == 0:
        background_tasks.add_task(profile_and_recommendation_workflow, current_user.user_id)

    return {"answer": answer, "tool_calls": tool_calls, "tool_results": tool_results}

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
        rows = await conn.fetch("SELECT id, title, message, created_at, is_read FROM recommendations WHERE user_id = $1 ORDER BY created_at DESC LIMIT 50", current_user.user_id)
    return [{"id": r["id"], "title": r["title"], "body": r["message"], "date": r["created_at"], "is_read": r["is_read"]} for r in rows]

@app.patch("/recommendations/read")
async def mark_recommendations_read(current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        await conn.execute("UPDATE recommendations SET is_read = true WHERE user_id = $1", current_user.user_id)
    return {"status": "success"}

async def profile_update_task(user_id: int):
    try:
        history_list = await get_recent_chats(user_id, limit=10)
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_list])
        favorites_dict = await get_user_favorites(user_id)
        fav_items = [f"{f['Film']} ({f['Türler'] or ''})" for f in favorites_dict.get("favorites", [])]
        favorites_text = ", ".join(fav_items) if fav_items else "Henüz favori eklenmemiş."

        new_profile = await generate_user_profile(history_text, favorites_text)
        async with ctx.db_pool.acquire() as conn:
            await conn.execute("UPDATE user_profiles SET persona_summary = $1, last_updated = CURRENT_TIMESTAMP WHERE user_id = $2", new_profile, user_id)
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
        return None

async def generate_and_save_recommendation_task(user_id: int):
    try:
        persona = await get_user_persona(user_id)
        if not persona: return
            
        favs_dict = await get_user_favorites(user_id)
        fav_titles = [f["Film"] for f in favs_dict.get("favorites", [])][:5]
        
        async with ctx.db_pool.acquire() as conn:
            rows = await conn.fetch("SELECT title FROM recommendations WHERE user_id = $1 ORDER BY created_at DESC LIMIT 30", user_id)
            past_recs = [r["title"].lower().strip() for r in rows if r["title"]]
        
        res = await generate_push_message(persona, fav_titles, past_recs)
        if not res: return
        message_text, movie_title = res
        
        if movie_title and movie_title.lower().strip() in past_recs: return
        
        movie_id = None
        if movie_title:
            card = await get_movie_card_by_title(movie_title)
            if card:
                movie_id = str(card.get("movie_id"))
                movie_title = card.get("Film", movie_title)
        
        if movie_id and movie_title and message_text:
            async with ctx.db_pool.acquire() as conn:
                await conn.execute("INSERT INTO recommendations (user_id, movie_id, title, message) VALUES ($1, $2, $3, $4)", user_id, movie_id, movie_title, message_text)
    except Exception:
        pass

async def profile_and_recommendation_workflow(user_id: int):
    await profile_update_task(user_id)
    await generate_and_save_recommendation_task(user_id)

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
                    card = await get_movie_card_by_title(row["title"])
                    msg = PushMessage(
                        to=row["expo_push_token"],
                        title="Senin İçin Bir Film Buldum 🍿",
                        body=row["message"],
                        data={"type": "movie_recommendation", "movies": [card] if card else []},
                    )
                    await asyncio.to_thread(PushClient().publish, msg)
                    await conn.execute("UPDATE recommendations SET is_pushed = true WHERE id = $1", row["id"])
                except Exception:
                    pass
    except Exception:
        pass

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "3000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
