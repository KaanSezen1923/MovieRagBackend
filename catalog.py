"""
Katalog (film + dizi) endpoint'leri: TMDB'ye ince, önbellekli, herkese açık proxy.

Kurulum (api.py içinde):
    from catalog import router as catalog_router
    app.include_router(catalog_router)

Tüm endpoint'ler giriş GEREKTİRMEZ (ziyaretçiler de gezebilsin).
Cevaplar tek bir "kart" biçimine normalize edilir; film/dizi farkı React'e sızmaz.
"""
import asyncio
import os
import time
from typing import Literal, Optional

import httpx
from fastapi import APIRouter, HTTPException, Query

router = APIRouter(prefix="/catalog", tags=["catalog"])

TMDB = "https://api.themoviedb.org/3"
IMG = "https://image.tmdb.org/t/p"
MediaType = Literal["movie", "tv"]

# ---------------------------------------------------------------- TMDB istemcisi + cache
_http: Optional[httpx.AsyncClient] = None
_cache: dict = {}
_CACHE_MAX = 1000


def _client() -> httpx.AsyncClient:
    global _http
    if _http is None:
        _http = httpx.AsyncClient(
            timeout=10.0,
            headers={"Authorization": f"Bearer {os.getenv('AUTH_KEY')}", "accept": "application/json"},
        )
    return _http


async def aclose() -> None:
    """lifespan kapanışında çağrılır."""
    global _http
    if _http is not None:
        await _http.aclose()
        _http = None


async def tmdb(path: str, ttl: int = 600, **params) -> dict:
    params.setdefault("language", "tr-TR")
    key = (path, tuple(sorted((k, str(v)) for k, v in params.items())))
    hit = _cache.get(key)
    if hit and time.time() - hit[1] < ttl:
        return hit[0]
    try:
        r = await _client().get(f"{TMDB}{path}", params=params)
    except httpx.RequestError:
        raise HTTPException(status_code=502, detail="TMDB servisine bağlanılamadı.")
    if r.status_code == 404:
        raise HTTPException(status_code=404, detail="İçerik bulunamadı.")
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="TMDB API'ye ulaşılamadı.")
    data = r.json()
    if len(_cache) >= _CACHE_MAX:
        _cache.pop(next(iter(_cache)))
    _cache[key] = (data, time.time())
    return data


# ---------------------------------------------------------------- normalizasyon
def _img(path: Optional[str], size: str) -> Optional[str]:
    return f"{IMG}/{size}{path}" if path else None


def card(d: dict, media_type: Optional[str] = None) -> Optional[dict]:
    """Film ve dizi için ORTAK kart. media_type verilmediyse (trending/search) d['media_type']'a bakar."""
    mt = media_type or d.get("media_type")
    if mt not in ("movie", "tv"):          # search/multi içindeki 'person' kayıtlarını at
        return None
    date = d.get("release_date") or d.get("first_air_date") or ""
    return {
        "id": d["id"],
        "media_type": mt,
        "title": d.get("title") or d.get("name") or "",
        "overview": d.get("overview") or "",
        "poster": _img(d.get("poster_path"), "w500"),
        "backdrop": _img(d.get("backdrop_path"), "w1280"),
        "year": date[:4],
        "rating": round(d.get("vote_average") or 0, 1),
        "votes": d.get("vote_count") or 0,
    }


def cards(results: list, media_type: Optional[str] = None) -> list:
    return [c for c in (card(r, media_type) for r in results) if c]


def _providers(block: dict) -> dict:
    tr = (block or {}).get("results", {}).get("TR", {})
    pick = lambda k: [{"name": p["provider_name"], "logo": _img(p.get("logo_path"), "w92")} for p in tr.get(k, [])]
    return {"flatrate": pick("flatrate"), "rent": pick("rent"), "buy": pick("buy")}


# ---------------------------------------------------------------- ana sayfa (tek istekte tüm satırlar)
HOME_ROWS = {
    "trending":         ("/trending/all/week", None),
    "popular_movies":   ("/movie/popular", "movie"),
    "popular_tv":       ("/tv/popular", "tv"),
    "top_rated_movies": ("/movie/top_rated", "movie"),
    "top_rated_tv":     ("/tv/top_rated", "tv"),
    "now_playing":      ("/movie/now_playing", "movie"),
    "airing_today":     ("/tv/airing_today", "tv"),
}


@router.get("/home")
async def home():
    results = await asyncio.gather(*[tmdb(path, ttl=900) for path, _ in HOME_ROWS.values()])
    rows = {}
    for (key, (_, mt)), data in zip(HOME_ROWS.items(), results):
        rows[key] = cards(data.get("results", [])[:20], mt)
    hero = [c for c in rows["trending"] if c["backdrop"]][:5]
    return {"hero": hero, "rows": rows}


# ---------------------------------------------------------------- gezinme / filtreleme
CATEGORIES = {
    "movie": {"popular", "top_rated", "now_playing", "upcoming"},
    "tv": {"popular", "top_rated", "airing_today", "on_the_air"},
}


@router.get("/browse")
async def browse(
    media_type: MediaType = "movie",
    category: Optional[str] = None,                       # popular | top_rated | now_playing ...
    genre: Optional[int] = None,                          # TMDB tür id'si (/catalog/genres)
    year: Optional[int] = Query(default=None, ge=1900, le=2100),
    sort: Literal["popular", "rating", "newest"] = "popular",
    page: int = Query(default=1, ge=1, le=500),
):
    if category:
        if category not in CATEGORIES[media_type]:
            raise HTTPException(status_code=400, detail=f"Geçersiz kategori: {category}")
        data = await tmdb(f"/{media_type}/{category}", page=page)
    else:
        sort_by = {
            "popular": "popularity.desc",
            "rating": "vote_average.desc",
            "newest": "primary_release_date.desc" if media_type == "movie" else "first_air_date.desc",
        }[sort]
        params = {"sort_by": sort_by, "page": page, "include_adult": "false"}
        if sort == "rating":
            params["vote_count.gte"] = 200               # 3 oylu 10.0'lar üste çıkmasın
        if genre:
            params["with_genres"] = genre
        if year:
            params["primary_release_year" if media_type == "movie" else "first_air_date_year"] = year
        data = await tmdb(f"/discover/{media_type}", **params)
    return {
        "page": data.get("page", page),
        "total_pages": min(data.get("total_pages", 1), 500),   # TMDB 500. sayfadan sonrasını vermez
        "results": cards(data.get("results", []), media_type),
    }


@router.get("/genres")
async def genres(media_type: MediaType = "movie"):
    data = await tmdb(f"/genre/{media_type}/list", ttl=86400)
    return data.get("genres", [])


@router.get("/search")
async def search(q: str = Query(..., min_length=2, max_length=100), page: int = Query(default=1, ge=1, le=50)):
    data = await tmdb("/search/multi", ttl=300, query=q, page=page, include_adult="false")
    return {
        "page": data.get("page", page),
        "total_pages": min(data.get("total_pages", 1), 50),
        "results": cards(data.get("results", [])),
    }


# ---------------------------------------------------------------- detay
@router.get("/{media_type}/{item_id}")
async def detail(media_type: MediaType, item_id: int):
    d = await tmdb(
        f"/{media_type}/{item_id}", ttl=1800,
        append_to_response="credits,videos,watch/providers",
        include_video_language="tr,en,null",
    )
    out = card(d, media_type)

    credits = d.get("credits", {})
    if media_type == "movie":
        directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
    else:
        directors = [p["name"] for p in d.get("created_by", [])]           # dizide "yaratıcılar"
    cast = [
        {"id": c["id"], "name": c["name"], "character": c.get("character", ""), "photo": _img(c.get("profile_path"), "w185")}
        for c in credits.get("cast", [])[:12]
    ]

    vids = [v for v in d.get("videos", {}).get("results", [])
            if v.get("site") == "YouTube" and v.get("key") and v.get("type") in ("Trailer", "Teaser")]
    vids.sort(key=lambda v: (v["type"] != "Trailer", not v.get("official", False)))

    out.update({
        "tagline": d.get("tagline") or "",
        "status": d.get("status") or "",
        "genres": d.get("genres", []),
        "runtime": d.get("runtime") or ((d.get("episode_run_time") or [None])[0]),
        "directors": directors,
        "cast": cast,
        "trailer": vids[0]["key"] if vids else None,                        # YouTube video anahtarı
        "providers": _providers(d.get("watch/providers")),
    })
    if media_type == "tv":
        out.update({
            "number_of_seasons": d.get("number_of_seasons"),
            "number_of_episodes": d.get("number_of_episodes"),
            "seasons": [
                {"season_number": s["season_number"], "name": s.get("name"), "episode_count": s.get("episode_count"),
                 "air_date": s.get("air_date"), "poster": _img(s.get("poster_path"), "w342")}
                for s in d.get("seasons", []) if s.get("season_number", 0) > 0     # 0 = "Özel bölümler"
            ],
        })
    return out


@router.get("/tv/{item_id}/season/{season_number}")
async def season(item_id: int, season_number: int):
    d = await tmdb(f"/tv/{item_id}/season/{season_number}", ttl=3600)
    return {
        "name": d.get("name"),
        "overview": d.get("overview") or "",
        "episodes": [
            {"episode_number": e["episode_number"], "name": e.get("name"), "overview": e.get("overview") or "",
             "still": _img(e.get("still_path"), "w300"), "air_date": e.get("air_date"),
             "runtime": e.get("runtime"), "rating": round(e.get("vote_average") or 0, 1)}
            for e in d.get("episodes", [])
        ],
    }


@router.get("/{media_type}/{item_id}/similar")
async def similar(media_type: MediaType, item_id: int):
    """TMDB'nin kendi önerileri. Dizilerde ve tavsiye sisteminin tanımadığı filmlerde fallback."""
    data = await tmdb(f"/{media_type}/{item_id}/recommendations", ttl=3600)
    results = data.get("results") or (await tmdb(f"/{media_type}/{item_id}/similar", ttl=3600)).get("results", [])
    return {"results": cards(results[:18], media_type)}