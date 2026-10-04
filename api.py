import os
import sys
import json
import asyncio
from contextlib import AsyncExitStack, asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Depends, Request, Query
from pydantic import BaseModel, EmailStr, field_validator
from mcp import StdioServerParameters
from dotenv import load_dotenv
from shared import ctx, chat_statuses, MCPSessionPool
from client import (
    get_ollama_tools, generate_user_profile, generate_user_suggestions,
    generate_push_message, app as graph_app,
)
from auth import (
    CurrentUser, create_access_token, get_current_user,
    login_rate_limiter, signup_rate_limiter, client_ip,
)
import asyncpg
import bcrypt
import tempfile
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from exponent_server_sdk import PushClient, PushMessage
import random
import uvicorn
from groq import Groq
import wave
import io
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from logger import logger
import httpx 


load_dotenv()

# --- Zorunlu ortam değişkenleri: eksikse burada, net bir mesajla dur.
# (JWT_SECRET_KEY zaten auth.py import edilirken kontrol ediliyor; burada
# DB/Neo4j/harici API anahtarları gibi geri kalanları topluca doğruluyoruz.)
_REQUIRED_ENV_VARS = [
    "DB_NAME", "DB_USER", "DB_PASSWORD", "DB_HOST",
    "NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD",
    "AUTH_KEY",          # TMDB API key (server.py için)
    "WHISPER_API_KEY",   # Groq API key (transkripsiyon için)
    "NVIDIA_API_KEY",    # NVIDIA Chat API key (client.py için)
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

# --- Havuz boyutları (ortam değişkeniyle ayarlanabilir) ---
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
    trailer_url: str = None # YENİ EKLENDİ

class TranscriptionResponse(BaseModel):
    text: str
    success: bool



class TokenRequest(BaseModel):
    token: str

# /transcribe için üst sınırlar
MAX_AUDIO_BYTES = int(os.getenv("MAX_AUDIO_MB", "15")) * 1024 * 1024
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".ogg", ".webm", ".flac"}

def remove_file(path: str):
    """Arka planda geçici dosyaları silmek için yardımcı fonksiyon"""
    if os.path.exists(path):
        os.remove(path)


def _mcp_env() -> dict:
    """MCP sunucusuna (server.py) aktarılacak ortam değişkenleri."""
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
        # --- Postgres connection pool: her istekte yeni bağlantı açmayı bitirir ---
        ctx.db_pool = await asyncpg.create_pool(
            database=os.getenv("DB_NAME"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            host=os.getenv("DB_HOST"),
            min_size=DB_POOL_MIN_SIZE,
            max_size=DB_POOL_MAX_SIZE,
        )
        logger.info(f"Postgres pool hazır (min={DB_POOL_MIN_SIZE}, max={DB_POOL_MAX_SIZE}).")

        # --- MCP session havuzu: tek stdio session'ın tüm istekleri sıraya
        # sokmasını önlemek için birden fazla paralel session açılır ---
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


@app.get("/health")
async def health():
    """
    Load balancer / orchestrator (Docker healthcheck, k8s liveness-readiness vb.) için.
    Sadece process ayakta mı değil, DB ve MCP session havuzu gerçekten hazır mı diye bakar.
    """
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


class ChatRequest(BaseModel):
    prompt: str
    session_id: str

class ChatResponse(BaseModel):
    answer: str
    tool_calls: list = []
    tool_results: list = []

def hash_password(password: str):
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')

def verify_password(plain_password, hashed_password):
    password_byte = plain_password.encode('utf-8')
    hashed_byte = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_byte, hashed_byte)


# --- Veritabanı erişim katmanı (asyncpg pool, event loop'u bloklamaz) ---

async def save_chat_to_db(user_id: int, role: str, content: str, session_id: str):
    try:
        async with ctx.db_pool.acquire() as conn:
            await conn.execute(
                "INSERT INTO chat_history (user_id, role, content, session_id) VALUES ($1, $2, $3, $4)",
                user_id, role, content, session_id
            )
    except Exception as e:
       logger.error("DB Kayıt Hatası", exc_info=True)

async def save_recommendation_to_db(user_id: int, movie_id: str, title: str, message: str):
    try:
        async with ctx.db_pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO recommendations (user_id, movie_id, title, message) 
                VALUES ($1, $2, $3, $4)
                """,
                user_id, movie_id, title, message
            )
    except Exception as e:
        logger.error("Öneri veritabanına kaydedilemedi", exc_info=True)


async def get_user_persona(user_id: int):
    async with ctx.db_pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT persona_summary FROM user_profiles WHERE user_id = $1",
            user_id
        )
    return row[0] if row else None

async def generate_and_save_recommendation_task(user_id: int):
    try:
        persona = await get_user_persona(user_id)
        if not persona:
            return
            
        favs_dict = await get_user_favorites(user_id)
        favs = favs_dict.get("favorites", [])
        fav_titles = [f["Film"] for f in favs][:5]
        
        # 1. Daha önce önerilen filmleri veritabanından çek (Son 30 öneri)
        async with ctx.db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT title FROM recommendations WHERE user_id = $1 ORDER BY created_at DESC LIMIT 30", 
                user_id
            )
            past_recs = [r["title"].lower().strip() for r in rows if r["title"]]
        
        # 2. LLM'den mesaj ve film adı üret (Geçmiş önerileri hariç tutmasını söyleyerek)
        res = await generate_push_message(persona, fav_titles, past_recs)
        if not res:
            return
            
        message_text, movie_title = res
        movie_id = None
        
        # 3. Çifte Güvenlik: LLM kuralı çiğneyip yine aynı filmi önerdiyse kaydetme!
        if movie_title and movie_title.lower().strip() in past_recs:
            logger.info(f"İptal: '{movie_title}' kullanıcısına zaten önerilmiş.")
            return
        
        # Film adından detayları ve movie_id'yi bul
        if movie_title:
            card = await get_movie_card_by_title(movie_title)
            if card:
                movie_id = str(card.get("movie_id"))
                movie_title = card.get("Film", movie_title)
        
        # Eğer LLM'in önerdiği spesifik film TMDB'den bulunamadıysa yedek mekanizma çalışsın
        if not movie_id:
            cards = await get_movies_for_push(user_id)
            if cards:
                card = cards[0]
                movie_id = str(card.get("movie_id"))
                movie_title = card.get("Film")
                
            # Yedek mekanizma da aynısını bulduysa iptal et
            if movie_title and movie_title.lower().strip() in past_recs:
                logger.info(f"İptal: Yedek mekanizma '{movie_title}' filmini buldu ama zaten önerilmiş.")
                return
        
        # 4. Veritabanına kaydet
        if movie_id and movie_title and message_text:
            await save_recommendation_to_db(user_id, movie_id, movie_title, message_text)
            logger.info(f"Kullanıcı {user_id} için arka planda yeni öneri db'ye eklendi: {movie_title}")
            
    except Exception as e:
        logger.error(f"Öneri oluşturma task'i hatası (user_id={user_id})", exc_info=True)


def _compact_content(role: str, content: str) -> str:
    """Asistan mesajı film kartı JSON'uysa LLM bağlamı için kısa metne indirger."""
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

async def get_user_favorites(user_id: int, limit: int | None = None, offset: int = 0):
    try:
        async with ctx.db_pool.acquire() as conn:
            if limit is not None:
                rows = await conn.fetch(
                    """
                    SELECT movie_id, title, genres, director, cast_members, poster_url, imdb_rating, trailer_url
                    FROM favorites
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    LIMIT $2 OFFSET $3
                    """,
                    user_id, limit, offset
                )
                total = await conn.fetchval(
                    "SELECT COUNT(*) FROM favorites WHERE user_id = $1", user_id
                )
            else:
                # limit verilmediyse (internal çağrılar: persona/öneri/push üretimi)
                # eski davranış korunur — tam liste döner.
                rows = await conn.fetch(
                    """
                    SELECT movie_id, title, genres, director, cast_members, poster_url, imdb_rating, trailer_url
                    FROM favorites
                    WHERE user_id = $1
                    ORDER BY created_at DESC
                    """,
                    user_id
                )
                total = len(rows)

        favorites = [
            {
                "movie_id": r["movie_id"],
                "Film": r["title"],
                "Türler": r["genres"],
                "Director": r["director"],
                "Cast": r["cast_members"],
                "Poster": r["poster_url"],
                "TMDB Puanı": r["imdb_rating"],  # not: server.py ile aynı etiket kullanılsın diye "IMDb"den değiştirildi
                "Fragman": r["trailer_url"], # YENİ EKLENDİ
            } for r in rows
        ]

        return {"favorites": favorites, "total": total}
    except Exception as e:
        logger.error("Favori getirme hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Favoriler alınamadı.")

async def update_persona_in_db(user_id: int, summary: str):
    async with ctx.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE user_profiles SET persona_summary = $1, last_updated = CURRENT_TIMESTAMP WHERE user_id = $2",
            summary, user_id
        )

async def profile_update_task(user_id: int):
    try:
        history_list = await get_recent_chats(user_id, limit=10)
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_list])

        favorites_dict = await get_user_favorites(user_id)
        favs = favorites_dict.get("favorites", [])
        fav_items = [f"{f['Film']} ({f['Türler'] or ''})" for f in favs]
        favorites_text = ", ".join(fav_items) if fav_items else "Henüz favori eklenmemiş."

        new_profile = await generate_user_profile(history_text, favorites_text)

        await update_persona_in_db(user_id, new_profile)
        logger.info(f"Kullanıcı {user_id} için profil güncellendi.")
    except Exception as e:
        logger.error(f"Profilleme hatası (user_id={user_id})", exc_info=True)

@app.get("/chat/suggestions")
async def get_chat_suggestions(current_user: CurrentUser = Depends(get_current_user)):
    default_suggestions = ["Bilim kurgu filmi öner", "Nolan filmi öner", "Tim Burton filmi öner"]
    try:
        history_list = await get_recent_chats(current_user.user_id, limit=10)

        favorites_dict = await get_user_favorites(current_user.user_id)
        favs = favorites_dict.get("favorites", [])
        fav_items = [f"{f['Film']} ({f['Türler'] or ''})" for f in favs]

        if not history_list and not fav_items:
            return {"suggestions": default_suggestions}

        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history_list])
        favorites_text = ", ".join(fav_items) if fav_items else "Henüz favori eklenmemiş."

        suggestions = await generate_user_suggestions(history_text, favorites_text)
        return {"suggestions": suggestions}
    except Exception as e:
        logger.error("Öneri butonları üretme hatası", exc_info=True)
        return {"suggestions": default_suggestions}

@app.get("/chat/status/{session_id}")
async def get_chat_status(session_id: str, current_user: CurrentUser = Depends(get_current_user)):
    status = chat_statuses.get(session_id, "")
    return {"status": status}

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

                await conn.execute(
                    "INSERT INTO user_profiles (user_id) VALUES ($1)",
                    new_user_id
                )

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
            record = await conn.fetchrow(
                "SELECT id, password_hash FROM users WHERE username = $1", user.username
            )

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
                taken = await conn.fetchrow(
                    "SELECT id FROM users WHERE username = $1 AND id != $2",
                    req.new_username, current_user.user_id
                )
                if taken:
                    raise HTTPException(status_code=400, detail="Bu kullanıcı adı zaten alınmış.")

                await conn.execute(
                    "UPDATE users SET username = $1 WHERE id = $2",
                    req.new_username, current_user.user_id
                )

        return {"status": "success", "username": req.new_username}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Profil güncelleme hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Profil güncellenemedi.")

@app.post("/change-password")
async def change_password(req: ChangePasswordRequest, current_user: CurrentUser = Depends(get_current_user)):
    try:
        async with ctx.db_pool.acquire() as conn:
            record = await conn.fetchrow(
                "SELECT password_hash FROM users WHERE id = $1", current_user.user_id
            )
            if not record or not verify_password(req.old_password, record["password_hash"]):
                raise HTTPException(status_code=400, detail="Mevcut şifreniz hatalı.")

            hashed_pwd = hash_password(req.new_password)
            await conn.execute(
                "UPDATE users SET password_hash = $1 WHERE id = $2",
                hashed_pwd, current_user.user_id
            )
        return {"status": "success", "message": "Şifreniz başarıyla değiştirildi."}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Şifre değiştirme hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Şifre değiştirilemedi.")



@app.get("/sessions")
async def get_sessions(
    current_user: CurrentUser = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
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
            total = await conn.fetchval(
                "SELECT COUNT(DISTINCT session_id) FROM chat_history WHERE user_id = $1",
                current_user.user_id
            )
        sessions = [{"session_id": r["session_id"], "title": r["title"] if r["title"] else "Yeni Sohbet"} for r in rows]
        return {"sessions": sessions, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        logger.error("Oturum listesi getirme hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Oturumlar alınamadı.")

@app.get("/chat/{session_id}")
async def get_chat_history(
    session_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
):
    try:
        async with ctx.db_pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT role, content FROM chat_history
                WHERE user_id = $1 AND session_id = $2
                ORDER BY created_at ASC
                LIMIT $3 OFFSET $4
                """,
                current_user.user_id, session_id, limit, offset
            )
            total = await conn.fetchval(
                "SELECT COUNT(*) FROM chat_history WHERE user_id = $1 AND session_id = $2",
                current_user.user_id, session_id
            )
        history = [{"role": r["role"], "content": r["content"]} for r in rows]
        return {"history": history, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        logger.error("Sohbet geçmişi getirme hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Sohbet geçmişi alınamadı.")

@app.delete("/chat/{session_id}")
async def delete_session(session_id: str, current_user: CurrentUser = Depends(get_current_user)):
    try:
        async with ctx.db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM chat_history WHERE user_id = $1 AND session_id = $2",
                current_user.user_id, session_id
            )
        return {"status": "success"}
    except Exception as e:
        logger.error("Sohbet silme hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Sohbet silinemedi.")

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(audio: UploadFile = File(...), current_user: CurrentUser = Depends(get_current_user)):
    temp_file_path = None
    try:
        file_ext = os.path.splitext(audio.filename or "")[1].lower()
        if not file_ext:
            file_ext = ".wav"
        if file_ext not in ALLOWED_AUDIO_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Desteklenmeyen ses formatı: {file_ext}. İzin verilenler: "
                       f"{', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}",
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            total = 0
            while chunk := await audio.read(1024 * 1024):  # 1 MB'lık parçalar halinde oku
                total += len(chunk)
                if total > MAX_AUDIO_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"Dosya çok büyük (limit: {MAX_AUDIO_BYTES // (1024 * 1024)} MB).",
                    )
                temp_file.write(chunk)
            if total == 0:
                raise HTTPException(status_code=400, detail="Dosya içeriği boş.")
            temp_file_path = temp_file.name

        with open(temp_file_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model="whisper-large-v3",
                file=audio_file,
                language="tr"
            )

        return TranscriptionResponse(text=transcription.text.strip(), success=True)

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Transkripsiyon hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Ses dosyası işlenemedi, lütfen tekrar deneyin.")
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


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
        "session_id": request.session_id
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

    await save_chat_to_db(current_user.user_id, "user", request.prompt, request.session_id)
    await save_chat_to_db(current_user.user_id, "assistant", answer, request.session_id)

    msg_count = await get_user_message_count(current_user.user_id)
    if msg_count > 0 and msg_count % 5 == 0:
        background_tasks.add_task(profile_update_task, current_user.user_id)
         background_tasks.add_task(generate_and_save_recommendation_task, current_user.user_id)

   

    return {"answer": answer, "tool_calls": tool_calls, "tool_results": tool_results}

async def get_user_message_count(user_id: int):
    async with ctx.db_pool.acquire() as conn:
        return await conn.fetchval(
            "SELECT COUNT(*) FROM chat_history WHERE user_id = $1", user_id
        )

@app.post("/favorites")
async def add_favorite(fav: FavoriteRequest, current_user: CurrentUser = Depends(get_current_user)):
    try:
        async with ctx.db_pool.acquire() as conn:
            # YENİ EKLENDİ: trailer_url
            await conn.execute(
                """
                INSERT INTO favorites (user_id, movie_id, title, genres, director, cast_members, poster_url, imdb_rating, trailer_url)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (user_id, movie_id) DO NOTHING
                """,
                current_user.user_id, fav.movie_id, fav.title,
                fav.genres, fav.director, fav.cast_members,
                fav.poster_url, fav.imdb_rating, fav.trailer_url # YENİ PARAMETRE
            )
        return {"status": "success", "message": "Film detaylarıyla birlikte favorilere eklendi."}
    except Exception as e:
        logger.error("Favori ekleme hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Favori eklenemedi.")

@app.delete("/favorites/{movie_id}")
async def delete_favorite(movie_id: str, current_user: CurrentUser = Depends(get_current_user)):
    try:
        async with ctx.db_pool.acquire() as conn:
            await conn.execute(
                "DELETE FROM favorites WHERE user_id = $1 AND movie_id = $2",
                current_user.user_id, movie_id
            )
        return {"status": "success"}
    except Exception as e:
        logger.error("Favori silme hatası", exc_info=True)
        raise HTTPException(status_code=500, detail="Favori silinemedi.")

@app.get("/favorites")
async def get_favorites(
    current_user: CurrentUser = Depends(get_current_user),
    limit: int = Query(default=20, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
):
    # get_user_favorites zaten aynı sorguyu yapıyor (bkz. yukarıda) — tekrar yazmak yerine onu kullanıyoruz.
    return await get_user_favorites(current_user.user_id, limit=limit, offset=offset)

@app.get("/favorites/ids")
async def get_favorite_ids(current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT movie_id FROM favorites WHERE user_id = $1", current_user.user_id
        )
    return {"favorite_ids": [r["movie_id"] for r in rows]}

@app.get("/discover")
async def discover_movies(category: str = Query(default="Popüler"), current_user: CurrentUser = Depends(get_current_user)):
    # 1. Kategoriye göre TMDB uç noktasını (endpoint) belirle
    tmdb_endpoints = {
        "Popüler": "popular",
        "Vizyondakiler": "now_playing",
        "En Çok Oy Alanlar": "top_rated",
        "Yakında": "upcoming"
    }
    endpoint = tmdb_endpoints.get(category, "popular")
    auth_key = os.getenv("AUTH_KEY") # .env dosyanızdaki TMDB API anahtarını kullanıyoruz
    
    if not auth_key:
        raise HTTPException(status_code=500, detail="TMDB AUTH_KEY yapılandırılmamış.")

    headers = {
        "Authorization": f"Bearer {auth_key}",
        "accept": "application/json"
    }
    
    async with httpx.AsyncClient(timeout=15.0) as client:
        try:
            # 2. Filmlerin ana listesini çek
            url = f"https://api.themoviedb.org/3/movie/{endpoint}?language=tr-TR&page=1"
            response = await client.get(url, headers=headers)
            
            if response.status_code != 200:
                raise HTTPException(status_code=500, detail="TMDB API'ye ulaşılamadı.")
                
            # Performansı yüksek tutmak için ilk 15 filmi alıyoruz
            results = response.json().get("results", [])[:15]
            
            # 3. Yönetmen, Oyuncu ve Fragman bilgileri için paralel istekler at
            async def fetch_movie_detail(movie_id):
                detail_url = f"https://api.themoviedb.org/3/movie/{movie_id}?language=tr-TR&append_to_response=credits,videos"
                res = await client.get(detail_url, headers=headers)
                return res.json() if res.status_code == 200 else None

            # Tüm filmlerin detaylarını aynı anda (paralel) çekiyoruz
            details = await asyncio.gather(*[fetch_movie_detail(m["id"]) for m in results])
            
            movies = []
            for d in details:
                if not d:
                    continue
                
                # Yönetmeni bul
                crew = d.get("credits", {}).get("crew", [])
                director = next((c["name"] for c in crew if c["job"] == "Director"), "Bilinmiyor")
                
                # Başrol oyuncularını bul (İlk 3 oyuncu)
                cast_list = d.get("credits", {}).get("cast", [])
                cast = ", ".join([c["name"] for c in cast_list[:3]]) if cast_list else "Bilinmiyor"
                
                # Youtube fragmanını bul
                videos = d.get("videos", {}).get("results", [])
                trailer = next((f"https://www.youtube.com/watch?v={v['key']}" for v in videos if v["site"] == "YouTube" and v["type"] == "Trailer"), "")
                
                # Türleri listele
                genres = [g["name"] for g in d.get("genres", [])]
                
                poster_path = d.get("poster_path")
                poster_url = f"https://image.tmdb.org/t/p/w500{poster_path}" if poster_path else "https://via.placeholder.com/500x750?text=No+Poster"
                
                # Mobil uygulamanın normalizeMovie fonksiyonuna uygun formata getiriyoruz
                movies.append({
                    "id": str(d.get("id")),
                    "Film": d.get("title"), # "title" yerine "Film"
                    "Özet": d.get("overview") or d.get("tagline") or "Özet bulunamadı.", # "overview" yerine "Özet"
                    "Poster": poster_url, # "poster_url" yerine "Poster"
                    "IMDb": str(round(d.get("vote_average", 0), 1)), # "imdb_rating" yerine "IMDb"
                    "Director": director, # "director" yerine "Director"
                    "Cast": cast, # "cast" yerine "Cast"
                    "Türler": ", ".join(genres), # Dizi yerine virgüllü string olarak gönder
                    "Fragman": trailer # "trailer_url" yerine "Fragman"
                })
                
            return {"movies": movies}
            
        except httpx.RequestError as e:
            logger.error(f"HTTP İstek Hatası: {e}")
            raise HTTPException(status_code=500, detail="Film servisine bağlanılamadı.")

@app.post("/update-push-token")
async def update_push_token(req: TokenRequest, current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        await conn.execute(
            "UPDATE users SET expo_push_token = $1 WHERE id = $2",
            req.token, current_user.user_id
        )
    return {"status": "success"}

@app.get("/recommendations")
async def get_user_recommendations(current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        rows = await conn.fetch(
            """
            SELECT id, title, message, created_at, is_read 
            FROM recommendations 
            WHERE user_id = $1 
            ORDER BY created_at DESC LIMIT 50
            """, 
            current_user.user_id
        )
    return [{"id": r["id"], "title": r["title"], "body": r["message"], "date": r["created_at"], "is_read": r["is_read"]} for r in rows]

@app.patch("/recommendations/read")
async def mark_recommendations_read(current_user: CurrentUser = Depends(get_current_user)):
    async with ctx.db_pool.acquire() as conn:
        await conn.execute("UPDATE recommendations SET is_read = true WHERE user_id = $1", current_user.user_id)
    return {"status": "success"}

def _parse_cards(result) -> list:
    """MCP tool sonucundan (JSON liste) film kartlarını çıkarır."""
    try:
        data = json.loads(result.content[0].text)
        return [c for c in data if isinstance(c, dict)] if isinstance(data, list) else []
    except Exception:
        return []

async def generate_push_notification(user_id: int):
    persona = await get_user_persona(user_id)
    if not persona:
        return None
    favs_dict = await get_user_favorites(user_id)
    favs = favs_dict["favorites"]
    fav_titles = [f["Film"] for f in favs][:5]
    return await generate_push_message(persona, fav_titles)

async def get_movies_for_push(user_id: int):
    """Favori türlerden rastgele 2'sini seçip MCP üzerinden kart getirir."""
    try:
        favs_dict = await get_user_favorites(user_id)
        favs = favs_dict.get("favorites", [])
        fav_titles = {f["Film"].strip().lower() for f in favs if f.get("Film")}
        genres = {g.strip() for f in favs for g in (f.get("Türler") or "").split(",") if g.strip()}
        picked = random.sample(sorted(genres), k=min(2, len(genres))) if genres else ["Action", "Drama"]

        result = await ctx.session.call_tool(
            "search_movies_by_filters",
            {"genre_name": ",".join(picked), "min_rating": 6.5},
        )
        cards = [c for c in _parse_cards(result) if c.get("Film", "").strip().lower() not in fav_titles]
        return cards[:3]
    except Exception as e:
        logger.error("Push bildirimi için film üretme hatası", exc_info=True)
        return []

async def get_movie_card_by_title(title: str):
    if not title:
        return None
    try:
        result = await ctx.session.call_tool("get_movie_card", {"title": title})
        cards = _parse_cards(result)
        return cards[0] if cards else None
    except Exception as e:
        logger.error(f"Film kartı getirme hatası (title={title})", exc_info=True)
        return None

async def send_random_notifications():
    try:
        async with ctx.db_pool.acquire() as conn:
            # Sadece push edilmemiş olanları (is_pushed=false) ve push_token'ı olanları bul
            records = await conn.fetch(
                """
                SELECT DISTINCT ON (r.user_id) r.id, r.user_id, r.movie_id, r.title, r.message, u.expo_push_token 
                FROM recommendations r
                JOIN users u ON r.user_id = u.id
                WHERE r.is_pushed = false AND u.expo_push_token IS NOT NULL
                ORDER BY r.user_id, r.created_at DESC -- DİKKAT: DISTINCT ON sonrası sıralama şarttır
                LIMIT 50
                """
            )
            
            if not records:
                return

            for row in records:
                try:
                    # Push bildirimine filmin görseli/detayları gitsin diye kart bilgisini alıyoruz
                    card = await get_movie_card_by_title(row["title"])
                    recommended_movies = [card] if card else []
                    
                    msg = PushMessage(
                        to=row["expo_push_token"],
                        title="Senin İçin Bir Film Buldum 🍿",
                        body=row["message"],
                        data={"type": "movie_recommendation", "movies": recommended_movies},
                    )
                    # Expo'ya gönder (Asenkron)
                    await asyncio.to_thread(PushClient().publish, msg)
                    
                    # Gönderim başarılı olunca veritabanında "is_pushed = true" yapıyoruz
                    await conn.execute(
                        "UPDATE recommendations SET is_pushed = true WHERE id = $1",
                        row["id"]
                    )
                    logger.info(f"Hazır bildirim gönderildi ve db güncellendi: {row['user_id']}")
                except Exception as e:
                    logger.error(f"Tekil bildirim gönderme hatası (id={row['id']})", exc_info=True)
                    
    except Exception as e:
        logger.error("Zamanlayıcı bildirim döngüsü hatası", exc_info=True)


if __name__ == "__main__":
    # NOT: reload=True SADECE local geliştirmede kullanılır (dosya izleme, ekstra process açar).
    # Production'da bu script'i direkt çalıştırma; şunlardan birini kullan:
    #   uvicorn api:app --host 0.0.0.0 --port 3000 --workers 4
    #   gunicorn api:app -k uvicorn.workers.UvicornWorker --workers 4 --bind 0.0.0.0:3000
    # Yerelde reload istiyorsan: RELOAD=1 python api.py
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "3000")),
        reload=os.getenv("RELOAD", "false").lower() == "true",
    )
