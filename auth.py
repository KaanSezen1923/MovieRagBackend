"""
JWT tabanlı kimlik doğrulama ve basit rate limiting.

Gerekli ortam değişkeni:
  JWT_SECRET_KEY      - ZORUNLU. En az 32 karakter rastgele bir string.
                         Üretmek için: python -c "import secrets; print(secrets.token_hex(32))"
Opsiyonel:
  JWT_EXPIRE_MINUTES  - Token geçerlilik süresi (dakika). Varsayılan: 7 gün.
"""
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "JWT_SECRET_KEY ortam değişkeni tanımlı değil. Önce üretip .env dosyana ekle:\n"
        '  python -c "import secrets; print(secrets.token_hex(32))"'
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", str(60 * 24 * 7)))  # 7 gün

_security = HTTPBearer()


@dataclass
class CurrentUser:
    user_id: int
    username: str


def create_access_token(user_id: int, username: str) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "username": username,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)).timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(_security),
) -> CurrentUser:
    """
    'Authorization: Bearer <token>' header'ını doğrular ve isteği yapan kullanıcıyı döner.
    Endpoint'lerde artık user_id'yi body/URL'den almak yerine bu dependency'den al:

        @app.get("/favorites")
        async def get_favorites(current_user: CurrentUser = Depends(get_current_user)):
            ...current_user.user_id...
    """
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Oturum süresi doldu, tekrar giriş yapın.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Geçersiz kimlik doğrulama bilgisi.",
        )

    try:
        user_id = int(payload.get("sub"))
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Geçersiz token içeriği.")

    return CurrentUser(user_id=user_id, username=payload.get("username", ""))


class RateLimiter:
    """
    Basit, bellek-içi sliding-window rate limiter.

    NOT: Bu implementasyon tek process için çalışır (in-memory).
    Birden fazla API instance'ı ile yatay ölçeklediğinde bunun yerine
    Redis tabanlı bir limiter (örn. slowapi + redis, veya fastapi-limiter)
    kullanman gerekir — yoksa her instance kendi sayacını tutar ve limit
    instance sayısı kadar gevşemiş olur.
    """

    def __init__(self, max_attempts: int, window_seconds: int):
        self.max_attempts = max_attempts
        self.window_seconds = window_seconds
        self._hits: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> None:
        now = time.time()
        window_start = now - self.window_seconds
        hits = [t for t in self._hits[key] if t > window_start]
        if len(hits) >= self.max_attempts:
            retry_after = max(int(self.window_seconds - (now - hits[0])), 1)
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Çok fazla deneme yapıldı. {retry_after} saniye sonra tekrar deneyin.",
            )
        hits.append(now)
        self._hits[key] = hits


# Uygulama bir reverse proxy'nin (nginx, Caddy, cloud load balancer vb.) ARKASINDA
# çalışıyorsa ve o proxy X-Forwarded-For'u güvenilir şekilde set ediyorsa (istemcinin
# gönderdiği değeri EZEREK) TRUST_PROXY_HEADERS=true yap. Aksi halde bu header'a
# güvenmek, herkesin rastgele bir IP uydurup rate limiter'ı bypass etmesine izin verir.
TRUST_PROXY_HEADERS = os.getenv("TRUST_PROXY_HEADERS", "false").lower() == "true"


def client_ip(request: Request) -> str:
    if TRUST_PROXY_HEADERS:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # Proxy zinciri varsa ilk değer orijinal istemcidir (proxy'nin EKLEDİĞİ
            # kısım, istemcinin gönderdiği kısmın SONUNA eklenir).
            return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


# Login: 5 deneme / 5 dakika (IP başına) — brute force koruması
login_rate_limiter = RateLimiter(max_attempts=5, window_seconds=300)
# Signup: 3 kayıt / saat (IP başına) — otomatik hesap açma spam'ini yavaşlatır
signup_rate_limiter = RateLimiter(max_attempts=3, window_seconds=3600)