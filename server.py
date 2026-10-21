"""
MCP sunucusu: film ve dizi arama araçları.

MİMARİ DEĞİŞİKLİK — "önce aday, sonra zenginleştir":
  Eski sürümde her arama aracı, döndürdüğü HER filmi TMDB'ye 1 ağır istekle (videos+credits+providers)
  zenginleştiriyordu; LLM sonra bunların sadece 1-5'ini seçiyordu. Şimdi:
    * Arama araçları (search_*, get_similar_movies) HAFİF aday kartları döner: Neo4j'den ya da
      TMDB liste yanıtından, ek istek olmadan.
    * enrich_movies(ids) yalnızca SON seçilen filmler için fragman/platform/kadro getirir.

RENDER & BULUT DAYANIKLILIĞI DÜZELTMELERİ:
  * Render'da yerel Ollama (localhost:11434) çalışmadığında veya vektör index'ine erişilemediğinde
    search_movies_semantically ve search_series_semantically "Semantik arama modeli aktif değil."
    dönüp patlamak yerine otomatik olarak Hibrit Yedek Arama (Neo4j Anahtar Kelime + Özet/Başlık
    Eşleşmesi + TMDB Keyword Discover & Search + Yerel JSON Veri Seti) çalıştırır.
  * get_keyword_id tam eşleşmelere (örn. "vampire" -> 3133) ve tekil/çoğul dönüşümüne öncelik verir;
    çoklu anahtar kelimeleri "|" (OR) ile birleştirir.
  * search_series_by_filters ve search_movies_by_filters açık bir keyword ("vampire" vb.) verildiğinde
    önce tür kısıtlarını gevşetir; keyword kısıtını en son gevşetir ve TMDB sonuçlarını Bayesian
    ağırlıklı puanla sıralar (40 oylu alakasız yapımların üste çıkmasını önler).
  * Neo4j Aura bağlantı kopmalarına karşı keep_alive ve max_connection_lifetime eklendi.
"""
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

load_dotenv()

from movie_common import QUERY_PREFIX, VECTOR_INDEX, bayes_score, norm, get_embedder

AUTH_KEY = os.getenv("AUTH_KEY")
BASE_URL = "https://api.themoviedb.org/3"
DEFAULT_LIMIT = 20
BAYES_M, BAYES_C = 500, 6.5

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
neo4j_driver = GraphDatabase.driver(
    NEO4J_URI,
    auth=(NEO4J_USER, NEO4J_PASSWORD),
    max_connection_lifetime=300,
    keep_alive=True,
    connection_timeout=10,
)

EMBED_HOST = os.getenv("OLLAMA_EMBED_HOST")
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

embeddings_model = None
try:
    _m = get_embedder()
    _m.embed_query("ping")          # gerçek bağlantı testi
    embeddings_model = _m
except Exception as e:
    print(f"Embedding modeli yüklenemedi (hibrit yedek arama kullanılacak): {e}", file=sys.stderr)

mcp = FastMCP("TMDB Movie Explorer")

http = requests.Session()
http.headers.update({"accept": "application/json", "Authorization": f"Bearer {AUTH_KEY}"})
http.mount("https://", HTTPAdapter(
    max_retries=Retry(total=2, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"]),
    pool_maxsize=16))


def tmdb_get(path: str, timeout: int = 8, **params) -> dict:
    r = http.get(f"{BASE_URL}{path}", params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------- türler
GENRE_DICT = {
    "Action": 28, "Aksiyon": 28, "Adventure": 12, "Macera": 12,
    "Animation": 16, "Animasyon": 16, "Comedy": 35, "Komedi": 35,
    "Crime": 80, "Suç": 80, "Documentary": 99, "Belgesel": 99,
    "Drama": 18, "Dram": 18, "Family": 10751, "Aile": 10751,
    "Fantasy": 14, "Fantastik": 14, "History": 36, "Tarih": 36,
    "Horror": 27, "Korku": 27, "Music": 10402, "Müzik": 10402,
    "Mystery": 9648, "Gizem": 9648, "Romance": 10749, "Romantik": 10749,
    "Science Fiction": 878, "Sci-Fi": 878, "Bilim Kurgu": 878,
    "TV Movie": 10770, "TV Filmi": 10770, "Thriller": 53, "Gerilim": 53,
    "War": 10752, "Savaş": 10752, "Western": 37,
}
REVERSE_GENRE_DICT = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy", 80: "Crime", 99: "Documentary",
    18: "Drama", 10751: "Family", 14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction", 10770: "TV Movie",
    53: "Thriller", 10752: "War", 37: "Western",
}
GENRE_LOOKUP = {norm(k): v for k, v in GENRE_DICT.items()}


def genre_ids(csv) -> list:
    """'Science Fiction, Korku' -> [878, 27]. Türkçe/İngilizce, büyük/küçük harf fark etmez."""
    ids = []
    for g in (csv or "").split(","):
        gid = GENRE_LOOKUP.get(norm(g))
        if gid and gid not in ids:
            ids.append(gid)
    return ids


# ---------------------------------------------------------------- cache
def ttl_cache(ttl=3600, maxsize=500):
    cache = {}

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(sorted(kwargs.items()))
            hit = cache.get(key)
            if hit and time.time() - hit[1] < ttl:
                return hit[0]
            result = func(*args, **kwargs)
            if len(cache) >= maxsize:
                cache.pop(next(iter(cache)))
            cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator


# ---------------------------------------------------------------- Türkçe/İngilizce Tema ve Anahtar Kelime Çözümleme
_TR_EN_THEME_MAP = {
    "vampir": "vampire",
    "vampirler": "vampire",
    "zombi": "zombie",
    "zombiler": "zombie",
    "zombili": "zombie",
    "kurt adam": "werewolf",
    "kurtadam": "werewolf",
    "cadi": "witch",
    "cadilar": "witch",
    "buyucu": "wizard",
    "buyu": "magic",
    "sihir": "magic",
    "hayalet": "ghost",
    "cin": "demon",
    "iblis": "demon",
    "seytan": "exorcism",
    "seri katil": "serial killer",
    "dedektif": "detective",
    "polisiye": "detective",
    "cinayet": "murder",
    "zaman yolculugu": "time travel",
    "zamanda yolculuk": "time travel",
    "yapay zeka": "artificial intelligence",
    "uzay": "space",
    "astronot": "astronaut",
    "uzayli": "alien",
    "uzaylilar": "alien",
    "soygun": "heist",
    "banka soygunu": "bank robbery",
    "hapishane": "prison",
    "siberpunk": "cyberpunk",
    "distopya": "dystopia",
    "distopik": "dystopia",
    "kiyamet": "apocalypse",
    "kiyamet sonrasi": "post-apocalyptic",
    "salgin": "pandemic",
    "intikam": "revenge",
    "hafiza kaybi": "amnesia",
    "hayatta kalma": "survival",
    "canavar": "monster",
    "ejderha": "dragon",
    "casus": "spy",
    "ajan": "spy",
    "mafya": "mafia",
    "suikastci": "assassin",
    "mahkeme": "courtroom",
    "fener": "lighthouse",
    "deniz feneri": "lighthouse",
    "simulasyon": "simulation",
    "ters kose": "plot twist",
    "kara komedi": "dark comedy",
    "klostrofobik": "claustrophobic",
    "orta cag": "medieval",
    "mitoloji": "mythology",
    "super kahraman": "superhero",
    "okul": "high school",
    "lise": "high school",
    "genclik": "teen",
    "samuray": "samurai",
    "kovboy": "western",
    "korsan": "pirate",
    "denizalti": "submarine",
    "keskin nisanci": "sniper",
}

_SEMANTIC_STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "without",
    "by", "from", "into", "about", "as", "is", "are", "was", "were", "be", "been", "being",
    "that", "which", "who", "whom", "whose", "where", "when", "why", "how", "this", "these", "those",
    "tv", "series", "show", "shows", "television", "movie", "movies", "film", "films", "cinema",
    "featuring", "features", "centered", "around", "focusing", "focused", "story", "stories",
    "plot", "theme", "themed", "tone", "atmosphere", "style", "like", "similar", "good", "best",
    "great", "popular", "top", "new", "old", "set", "world", "life", "people", "man", "woman",
    "young", "group", "person", "character", "characters", "protagonist", "main", "recommend",
    "recommendation", "recommendations", "suggest", "suggestion",
    # Türkçe genel dolgu kelimeleri
    "bir", "ve", "veya", "ama", "fakat", "ancak", "icin", "gibi", "kadar", "daha", "cok", "en",
    "bana", "sana", "oner", "oneri", "onerisi", "onerileri", "tavsiye", "et", "eder", "misin",
    "film", "filmi", "filmleri", "dizi", "dizisi", "dizileri", "yapim", "yapimlar", "izle",
    "izlemek", "istiyorum", "ariyorum", "var", "mi", "olan", "gecen", "icinde", "konulu", "temali",
    "tarzi", "tarzinda", "hakkinda", "ustune", "uzerine", "kurulu", "odakli", "barindiran",
}


def _singularize_en(word: str) -> str:
    w = word.strip().lower()
    if len(w) <= 3:
        return w
    if w.endswith("wolves"):
        return w[:-6] + "wolf"
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if w.endswith(("ches", "shes", "xes", "zes", "sses")):
        return w[:-2]
    if w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


def _extract_semantic_terms(query: str) -> list[str]:
    """
    İngilizce veya Türkçe semantik sorgudan Neo4j/TMDB aramasında kullanılacak
    öz (anahtar) terimleri çıkarır.
    Örn: "TV series featuring vampires." -> ["vampire"]
         "vampir dizileri öner." -> ["vampire"]
    """
    q_norm = norm(query or "")
    if not q_norm:
        return []

    terms: list[str] = []
    # 1. Çok kelimeli veya Türkçe bilinen kalıpları yakala
    for tr_key, en_val in sorted(_TR_EN_THEME_MAP.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(tr_key)}\b", q_norm):
            if en_val not in terms:
                terms.append(en_val)

    # 2. İngilizce çok kelimeli bilinen ifadeleri koru
    for phrase in ("time travel", "serial killer", "artificial intelligence", "post apocalyptic",
                   "post-apocalyptic", "dark comedy", "bank robbery", "high school", "plot twist",
                   "world war", "space opera", "true crime", "martial arts", "found footage"):
        if phrase in q_norm and phrase not in terms:
            terms.append(phrase.replace("post apocalyptic", "post-apocalyptic"))

    # 3. Tekil kelimeleri ayıkla
    tokens = re.findall(r"[a-z0-9]+", q_norm)
    for tok in tokens:
        if len(tok) < 3 or tok in _SEMANTIC_STOPWORDS:
            continue
        mapped = _TR_EN_THEME_MAP.get(tok)
        if mapped:
            if mapped not in terms:
                terms.append(mapped)
            continue
        sing = _singularize_en(tok)
        if sing in _SEMANTIC_STOPWORDS:
            continue
        if sing not in terms:
            terms.append(sing)

    return terms[:6]


# ---------------------------------------------------------------- id çözümleme
@ttl_cache(ttl=86400)
def get_person_id(name: str):
    res = tmdb_get("/search/person", query=name, language="en-US").get("results", [])
    return res[0]["id"] if res else None


@ttl_cache(ttl=86400)
def get_keyword_id(keyword: str):
    """
    TMDB anahtar kelime ID'sini bulur.
    - Birebir isim eşleşmelerine (örn. 'vampire' -> 3133) öncelik verir.
    - Çoğul kelimelerde ('vampires') tekil haliyle de eşleştirir.
    - Virgülle ayrılmış birden fazla keyword varsa '|' (OR) ile birleştirir.
    """
    if not keyword:
        return None
    parts = [p.strip() for p in str(keyword).split(",") if p.strip()]
    resolved_ids = []
    for raw_kw in parts:
        q_norm = norm(raw_kw)
        q_mapped = _TR_EN_THEME_MAP.get(q_norm, raw_kw.strip())
        q_sing = _singularize_en(q_mapped)
        candidates_to_try = [q_mapped]
        if q_sing != q_mapped.lower():
            candidates_to_try.append(q_sing)

        found_id = None
        for q in candidates_to_try:
            try:
                res = tmdb_get("/search/keyword", query=q).get("results", [])
            except Exception:
                res = []
            if not res:
                continue
            targets = {norm(q), norm(q_sing), norm(q_mapped)}
            exact = [r for r in res if norm(r.get("name", "")) in targets]
            found_id = (exact[0] if exact else res[0])["id"]
            if exact:
                break
        if found_id is not None and str(found_id) not in resolved_ids:
            resolved_ids.append(str(found_id))

    if not resolved_ids:
        return None
    return int(resolved_ids[0]) if len(resolved_ids) == 1 else "|".join(resolved_ids)


@ttl_cache(ttl=3600)
def get_movie_id(title: str, year: int = 0):
    """
    Başlıktan TMDB id'si. Birebir başlık eşleşmelerini öne alır.
    Yıl belirtilmişse, o yılda çıkmış filmlere kesin öncelik verir,
    aksi takdirde en çok oylananı (vote_count) seçer.
    """
    params = {"query": title, "language": "en-US"}
    if year:
        params["year"] = year

    try:
        results = tmdb_get("/search/movie", **params).get("results", [])
    except Exception:
        return None

    if not results:
        return None

    t = norm(title)
    exact = [r for r in results if norm(r.get("title", "")) == t or norm(r.get("original_title", "")) == t]
    pool = exact or results

    if year:
        year_matches = [
            r for r in pool
            if r.get("release_date") and r.get("release_date").startswith(str(year))
        ]
        if year_matches:
            pool = year_matches

    return max(pool, key=lambda r: r.get("vote_count", 0))["id"]


@ttl_cache(ttl=3600)
def resolve_movie_id(title: str = "", year: int = 0, movie_id: int = 0):
    """
    movie_id verilmişse önce onu doğrular: o id'li FİLMİN adı (tr/en/orijinal) verilen başlıkla uyuşuyorsa
    id'yi kullanır. Uyuşmazsa ya da 404 ise (örn. favorilerde media_type olmadığı için dizi id'si geldiyse)
    eski davranışa, yani başlıkla aramaya düşer.
    """
    if movie_id:
        for lang in ("tr-TR", "en-US"):
            try:
                d = tmdb_get(f"/movie/{int(movie_id)}", language=lang)
            except Exception:
                break          # 404 / ağ hatası: başlığa düş
            names = {norm(d.get("title", "")), norm(d.get("original_title", ""))}
            if not title or norm(title) in names:
                return d["id"]
    return get_movie_id(title, year) if title else None


# ---------------------------------------------------------------- kartlar
def _clip(text: str, n: int = 400) -> str:
    """Kelime sınırında kes (eski [:200] cümle ortasında kesiyordu)."""
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip(",;:- ") + "…"


def lite_card(*, mid, title, year="", rating=None, votes=None, genres=(), overview="", poster=None,
              director="", cast="", source="", score=None, relaxed=None, media_type="movie") -> dict:
    """Ek API isteği gerektirmeyen HAFİF aday kartı. Anahtarlar tam kartla uyumlu."""
    card = {
        "movie_id": str(mid),
        "media_type": media_type,
        "Film": title or "",
        "Director": director,
        "Cast": cast,
        "Yıl": str(year or ""),
        "TMDB Puanı": f"{float(rating):.1f}" if rating is not None else "",
        "vote_count": int(votes or 0),
        "Türler": ", ".join(g for g in genres if g),
        "Özet": _clip(overview),
        "Poster": f"https://image.tmdb.org/t/p/w500{poster}" if poster else "",
        "Fragman": "",
        "Şu Anki Platform(lar)": "",
        "Neden Önerildi": "",
        "source": source,
    }
    if score is not None:
        card["score"] = round(float(score), 4)
    if relaxed:
        card["relaxed"] = relaxed
    return card


def _pick_trailer(videos: list) -> str:
    yt = [v for v in videos if v.get("site") == "YouTube" and v.get("key")]
    yt.sort(key=lambda v: (not v.get("official", False), v.get("type") != "Trailer"))
    for v in yt:
        if v.get("type") in ("Trailer", "Teaser"):
            return f"https://www.youtube.com/watch?v={v['key']}"
    return ""


def fetch_movie_card(movie_id, fallback: dict | None = None) -> dict | None:
    """TAM kart (fragman, platform, kadro). Sadece son seçilen filmler için çağrılır."""
    fallback = fallback or {}
    try:
        d = tmdb_get(f"/movie/{movie_id}", language="tr-TR",
                     append_to_response="videos,credits,watch/providers",
                     include_video_language="en,tr,null")
    except Exception:
        return None

    overview = d.get("overview") or fallback.get("overview")
    if not overview:   # Türkçe özet yoksa İngilizceye düş (eski kod 'Özet bulunamadı' yazıyordu)
        try:
            overview = tmdb_get(f"/movie/{movie_id}", language="en-US").get("overview")
        except Exception:
            overview = None

    credits = d.get("credits", {})
    directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
    cast = [c["name"] for c in credits.get("cast", [])[:4]]
    tr = d.get("watch/providers", {}).get("results", {}).get("TR", {})
    flat = [p["provider_name"] for p in tr.get("flatrate", [])]
    rentbuy = sorted({p["provider_name"] for k in ("rent", "buy") for p in tr.get(k, [])})
    if flat:
        platform = ", ".join(flat)
    elif rentbuy:
        platform = "Kiralık/satın alınabilir: " + ", ".join(rentbuy[:4])
    else:
        platform = "Abonelik platformunda yok"
    poster = d.get("poster_path") or fallback.get("poster_path")

    return {
        "movie_id": str(d.get("id", movie_id)),
        "media_type": "movie",
        "Film": d.get("title") or d.get("original_title") or fallback.get("title", ""),
        "Director": ", ".join(directors) or fallback.get("director", ""),
        "Cast": ", ".join(cast) or fallback.get("cast", ""),
        "Yıl": (d.get("release_date") or "")[:4],
        "TMDB Puanı": f"{d.get('vote_average', 0):.1f}",
        "vote_count": int(d.get("vote_count") or 0),
        "Türler": ", ".join(REVERSE_GENRE_DICT.get(g["id"], g["name"]) for g in d.get("genres", [])),
        "Özet": _clip(overview) or "Özet bulunamadı.",
        "Poster": f"https://image.tmdb.org/t/p/w500{poster}" if poster else "",
        "Fragman": _pick_trailer(d.get("videos", {}).get("results", [])),
        "Şu Anki Platform(lar)": platform,
        "Neden Önerildi": "",
    }


def cards_from_ids(ids: list) -> list:
    with ThreadPoolExecutor(max_workers=5) as ex:
        cards = list(ex.map(fetch_movie_card, ids))   # map sırayı korur
    return [c for c in cards if c]


def _cards_from_tmdb_results(results: list, source: str, relaxed=None) -> list:
    return [
        lite_card(
            mid=r["id"], title=r.get("title"), year=(r.get("release_date") or "")[:4],
            rating=r.get("vote_average"), votes=r.get("vote_count"),
            genres=[REVERSE_GENRE_DICT.get(g) for g in r.get("genre_ids", [])],
            overview=r.get("overview"), poster=r.get("poster_path"),
            source=source, relaxed=relaxed)
        for r in results if r.get("id")
    ]


# Neo4j kayıtlarını aday karta çeviren ortak RETURN bloğu
_GRAPH_RETURN = """
  movie.id AS id, movie.title AS title, movie.year AS year,
  movie.vote_average AS rating, movie.vote_count AS votes,
  movie.overview AS overview, movie.poster_path AS poster,
  [(movie)-[:HAS_GENRE]->(g:Genre) | g.name] AS genres,
  [(movie)<-[:DIRECTED]-(d:Person) | d.name] AS director,
  [(movie)<-[:ACTED_IN]-(c:Person) | c.name] AS cast
"""
_WR = "(coalesce(movie.vote_count,0) * coalesce(movie.vote_average,0) + $m * $c) / (coalesce(movie.vote_count,0) + $m)"


def _run_graph(cypher: str, source: str, **params) -> list:
    params.setdefault("m", BAYES_M)
    params.setdefault("c", BAYES_C)
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        rows = [dict(r) for r in session.run(cypher, **params)]
    return [
        lite_card(
            mid=r["id"], title=r["title"], year=r.get("year") or "", rating=r.get("rating"),
            votes=r.get("votes"), genres=r.get("genres") or [], overview=r.get("overview"),
            poster=r.get("poster"), director=", ".join(r.get("director") or []),
            cast=", ".join((r.get("cast") or [])[:4]), source=source,
            score=r.get("score") if r.get("score") is not None else r.get("wr"))
        for r in rows
    ]


# ---------------------------------------------------------------- Yerel JSON Yedek Arama (Neo4j/TMDB boşsa)
def _search_local_json(
    is_series: bool,
    terms: list[str],
    excl_gids: list[int],
    min_rating: float,
    max_rating: float,
    year_from: int,
    year_to: int,
    min_votes: int,
    limit: int,
    source: str,
) -> list[dict]:
    """Neo4j ve TMDB erişilemez veya boş dönerse yerel JSON veri setinden (movies_data / series_data) arar."""
    filename = "series_data.json" if is_series else "movies_data.json"
    path = os.path.join(_ROOT_DIR, filename)
    if not os.path.exists(path) or not terms:
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
    except Exception:
        return []

    excl_set = set(excl_gids or [])
    m_bayes = SERIES_BAYES_M if is_series else BAYES_M
    scored = []
    for it in items:
        votes = int(it.get("vote_count") or 0)
        if votes < min_votes:
            continue
        rating = float(it.get("vote_average") or 0.0)
        if rating < min_rating or rating > max_rating:
            continue
        date_str = (it.get("first_air_date") if is_series else it.get("release_date")) or ""
        yr = int(date_str[:4]) if date_str[:4].isdigit() else 0
        if year_from and (not yr or yr < year_from):
            continue
        if year_to and (not yr or yr > year_to):
            continue
        gids = {g["id"] for g in (it.get("genres") or []) if isinstance(g, dict) and "id" in g}
        if excl_set & gids:
            continue

        kws = [norm(k.get("name", "")) for k in (it.get("keywords") or []) if isinstance(k, dict)]
        title_str = (it.get("name") if is_series else it.get("title")) or ""
        ov_norm = norm(it.get("overview") or "")
        tg_norm = norm(it.get("tagline") or "")
        tn_norm = norm(title_str)

        kw_hits = sum(1 for t in terms if any(t in kw for kw in kws))
        txt_hits = sum(1 for t in terms if t in ov_norm or t in tg_norm or t in tn_norm)
        if (kw_hits + txt_hits) == 0:
            continue

        wr = bayes_score(rating, votes, m=m_bayes, c=BAYES_C)
        sc = kw_hits * 2.5 + txt_hits * 1.0 + 0.15 * wr
        creators = it.get("creators" if is_series else "directors") or []
        cast_list = it.get("cast") or []
        card = lite_card(
            mid=it["id"],
            title=title_str,
            year=str(yr or ""),
            rating=rating,
            votes=votes,
            genres=[g.get("name", "") for g in (it.get("genres") or []) if isinstance(g, dict)],
            overview=it.get("overview") or "",
            poster=it.get("poster_path"),
            director=", ".join(c.get("name", "") for c in creators if isinstance(c, dict)),
            cast=", ".join(c.get("name", "") for c in cast_list[:4] if isinstance(c, dict)),
            source=source,
            score=sc,
            media_type="tv" if is_series else "movie",
        )
        scored.append((sc, card))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:limit]]


def _semantic_fallback_movies(
    semantic_query: str,
    excl_gids: list[int],
    min_rating: float,
    max_rating: float,
    year_from: int,
    year_to: int,
    min_votes: int,
    limit: int,
) -> list[dict]:
    """
    Embedding modeli aktif olmadığında (örn. Render'da yerel Ollama yoksa)
    veya vektör index'i hata verdiğinde devreye giren hibrit semantik arama:
      1) Neo4j Keyword + Overview/Tagline/Title eşleşmesi
      2) TMDB Keyword Discover + TMDB Search
      3) Yerel JSON yedeği
    """
    terms = _extract_semantic_terms(semantic_query)
    if not terms:
        terms = [norm(semantic_query)[:40]] if norm(semantic_query) else []
    out, seen = [], set()
    excl_set = set(excl_gids or [])

    # 1. Neo4j üzerinde anahtar kelime + özet/başlık araması
    if terms:
        cypher = f"""
        MATCH (movie:Movie)
        WHERE movie.vote_count >= $min_votes
          AND movie.vote_average >= $min_rating AND movie.vote_average <= $max_rating
          AND ($year_from = 0 OR movie.year >= $year_from)
          AND ($year_to = 0 OR movie.year <= $year_to)
          AND NOT EXISTS {{ MATCH (movie)-[:HAS_GENRE]->(x:Genre) WHERE x.id IN $excl }}
        OPTIONAL MATCH (movie)-[:HAS_KEYWORD]->(k:Keyword)
        WITH movie, collect(k.name_norm) AS kw_names,
             toLower(coalesce(movie.overview, '')) AS ov,
             toLower(coalesce(movie.tagline, '')) AS tg,
             coalesce(movie.title_norm, '') AS tn
        WITH movie,
             size([t IN $terms WHERE any(kw IN kw_names WHERE kw CONTAINS t)]) AS kw_hits,
             size([t IN $terms WHERE ov CONTAINS t OR tg CONTAINS t OR tn CONTAINS t]) AS txt_hits
        WHERE (kw_hits + txt_hits) > 0
        WITH movie, (kw_hits * 2.5 + txt_hits * 1.0) + 0.15 * {_WR} AS score
        RETURN {_GRAPH_RETURN}, score
        ORDER BY score DESC LIMIT $limit
        """
        try:
            for c in _run_graph(
                cypher, "semantic",
                terms=terms, min_votes=min_votes, min_rating=min_rating, max_rating=max_rating,
                year_from=year_from, year_to=year_to, excl=list(excl_set), limit=limit,
            ):
                if c["movie_id"] not in seen:
                    seen.add(c["movie_id"])
                    out.append(c)
        except Exception:
            pass

    # 2. TMDB Keyword Discover + TMDB Metin Araması
    if len(out) < limit and terms:
        kids = []
        for t in terms[:3]:
            kid = get_keyword_id(t)
            if kid is not None and str(kid) not in kids:
                kids.append(str(kid))
        if kids:
            params = {
                "include_adult": "false",
                "language": "en-US",
                "sort_by": "vote_count.desc",
                "vote_count.gte": max(int(min_votes), 1),
                "with_keywords": "|".join(kids),
            }
            if excl_set:
                params["without_genres"] = "|".join(map(str, excl_set))
            if min_rating > 0:
                params["vote_average.gte"] = min_rating
            if max_rating < 10:
                params["vote_average.lte"] = max_rating
            if year_from:
                params["primary_release_date.gte"] = f"{year_from}-01-01"
            if year_to:
                params["primary_release_date.lte"] = f"{year_to}-12-31"
            try:
                res = tmdb_get("/discover/movie", page=1, **params).get("results", [])
                res = [r for r in res if not (set(r.get("genre_ids", [])) & excl_set)]
                res.sort(key=lambda r: bayes_score(r.get("vote_average"), r.get("vote_count"), m=BAYES_M), reverse=True)
                for c in _cards_from_tmdb_results(res, "semantic"):
                    if c["movie_id"] not in seen:
                        seen.add(c["movie_id"])
                        out.append(c)
            except Exception:
                pass

    if len(out) < limit and terms:
        try:
            q_str = " ".join(terms[:2])
            res = tmdb_get("/search/movie", query=q_str, language="en-US", include_adult="false").get("results", [])
            valid_res = []
            for r in res:
                vc = int(r.get("vote_count") or 0)
                va = float(r.get("vote_average") or 0.0)
                rd = (r.get("release_date") or "")[:4]
                yr = int(rd) if rd.isdigit() else 0
                if vc < min_votes or va < min_rating or va > max_rating:
                    continue
                if year_from and (not yr or yr < year_from):
                    continue
                if year_to and (not yr or yr > year_to):
                    continue
                if set(r.get("genre_ids", [])) & excl_set:
                    continue
                valid_res.append(r)
            valid_res.sort(key=lambda r: bayes_score(r.get("vote_average"), r.get("vote_count"), m=BAYES_M), reverse=True)
            for c in _cards_from_tmdb_results(valid_res, "semantic"):
                if c["movie_id"] not in seen:
                    seen.add(c["movie_id"])
                    out.append(c)
        except Exception:
            pass

    # 3. Yerel JSON yedeği
    if len(out) < limit and terms:
        for c in _search_local_json(
            False, terms, list(excl_set), min_rating, max_rating,
            year_from, year_to, min_votes, limit, "semantic",
        ):
            if c["movie_id"] not in seen:
                seen.add(c["movie_id"])
                out.append(c)

    return out[:limit]


# ================================================================ MCP araçları
@mcp.tool()
@ttl_cache(ttl=1800)
def search_movies_by_filters(
    genre_name: str = None,
    exclude_genres: str = None,
    actor_name: str = None,
    director_name: str = None,
    keyword: str = None,
    min_rating: float = 0.0,
    max_rating: float = 10.0,
    year_from: int = 0,
    year_to: int = 0,
    min_votes: int = 200,
    sort_by: str = "vote_average.desc",
    only_streamable: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """
    TMDB canlı filtre araması (hafif aday kartları döner).
    genre_name / exclude_genres: virgülle ayrılmış türler ("Science Fiction,Western").
    min_rating / max_rating: TMDB puan aralığı. year_from / year_to: yıl aralığı (0 = sınırsız).
    min_votes: güvenilirlik eşiği (düşük oylu gürültüyü eler). Kült/az bilinen film için 20-50 ver.
    sort_by: "vote_average.desc" (kalite), "popularity.desc" (güncel popüler), "vote_count.desc".
    only_streamable: True ise yalnızca TR'de abonelikle izlenebilenler.
    """
    params = {"include_adult": "false", "language": "en-US", "sort_by": sort_by,
              "vote_count.gte": max(int(min_votes), 1)}
    incl = genre_ids(genre_name)
    excl = genre_ids(exclude_genres)
    if incl:
        params["with_genres"] = ",".join(map(str, incl))            # virgül = AND
    if excl:
        params["without_genres"] = "|".join(map(str, excl))
    if actor_name and (aid := get_person_id(actor_name)):
        params["with_cast"] = aid
    if director_name and (did := get_person_id(director_name)):
        params["with_crew"] = did
    kid = get_keyword_id(keyword) if keyword else None
    if kid:
        params["with_keywords"] = kid
    if min_rating > 0:
        params["vote_average.gte"] = min_rating
    if max_rating < 10:
        params["vote_average.lte"] = max_rating
    if year_from:
        params["primary_release_date.gte"] = f"{year_from}-01-01"
    if year_to:
        params["primary_release_date.lte"] = f"{year_to}-12-31"
    if only_streamable:
        params.update(watch_region="TR", with_watch_monetization_types="flatrate")

    def fetch(p):
        out = []
        seen_ids = set()
        for page in (1, 2):
            if len(out) >= limit:
                break
            try:
                res = tmdb_get("/discover/movie", page=page, **p).get("results", [])
            except Exception:
                break
            for r in res:
                if r.get("id") and r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    out.append(r)
            if len(res) < 20:
                break
        # Düşük oylu şişirilmiş puanların popüler klasikleri ezmemesi için popüler sayfayı da harmanla
        if p.get("sort_by") == "vote_average.desc" and (kid or min_votes < 150):
            try:
                pop_p = {**p, "sort_by": "popularity.desc"}
                for r in tmdb_get("/discover/movie", page=1, **pop_p).get("results", []):
                    if r.get("id") and r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        out.append(r)
            except Exception:
                pass
            out.sort(
                key=lambda r: bayes_score(r.get("vote_average"), r.get("vote_count"), m=BAYES_M),
                reverse=True,
            )
        return out

    relaxed = []
    movies = fetch(params)
    # Kısıt gevşetme: açıkça keyword verilmişse ÖNCE türleri gevşet, keyword'ü koru!
    if not movies and len(incl) > 1:
        params["with_genres"] = "|".join(map(str, incl))            # pipe = OR
        relaxed.append("genres_and_to_or")
        movies = fetch(params)
    if not movies and kid and incl:
        params.pop("with_genres", None)
        relaxed.append("genres_dropped_for_keyword")
        movies = fetch(params)
    if not movies and kid:
        if incl:
            params["with_genres"] = "|".join(map(str, incl))
        params.pop("with_keywords", None)
        relaxed.append("keyword")
        movies = fetch(params)

    excl_set = set(excl)
    movies = [m for m in movies if not (set(m.get("genre_ids", [])) & excl_set)]
    return json.dumps(_cards_from_tmdb_results(movies[:limit], "tmdb_filter", relaxed or None),
                      ensure_ascii=False)


@mcp.tool()
@ttl_cache(ttl=1800)
def get_similar_movies(movie_title: str = "", year: int = 0, limit: int = DEFAULT_LIMIT,
                       movie_id: int = 0) -> str:
    """
    Verilen filme benzer filmler (hafif aday kartları). İki kaynağı birleştirir:
    TMDB 'recommendations' (izleyici davranışı) + filmin KENDİ embedding'ine en yakın komşular (konu/atmosfer).
    movie_id biliniyorsa ver (başlık araması aynı adlı filmleri karıştırabilir); movie_title yine de gönder.
    """
    mid = resolve_movie_id(movie_title, year, movie_id)
    if not mid:
        return "[]"
    out, seen = [], {str(mid)}

    try:
        recs = tmdb_get(f"/movie/{mid}/recommendations", language="en-US").get("results", [])
        if not recs:
            recs = tmdb_get(f"/movie/{mid}/similar", language="en-US").get("results", [])
        for c in _cards_from_tmdb_results(recs, "tmdb_rec"):
            if c["movie_id"] not in seen:
                seen.add(c["movie_id"])
                out.append(c)
    except Exception:
        pass

    try:   # vektör komşuları (filmin vektörü Neo4j'de varsa)
        cypher = f"""
        MATCH (ref:Movie {{id: $id}}) WHERE ref.embedding IS NOT NULL
        CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $k, ref.embedding) YIELD node AS movie, score
        WHERE movie.id <> $id AND movie.vote_count >= 100
        RETURN {_GRAPH_RETURN}, score
        ORDER BY score DESC LIMIT $limit
        """
        for c in _run_graph(cypher, "vector_neighbor", id=mid, k=60, limit=limit):
            if c["movie_id"] not in seen:
                seen.add(c["movie_id"])
                out.append(c)
    except Exception:
        pass

    return json.dumps(out[: limit * 2], ensure_ascii=False)


@mcp.tool()
def search_movies_in_graph(category: str, search_value: str, limit: int = DEFAULT_LIMIT,
                           min_votes: int = 100) -> str:
    """
    Yerel Neo4j grafında arama (hafif aday kartları; TMDB'ye gitmez, çok hızlıdır).
    category: 'Director' | 'Actor' | 'Genre' | 'Keyword' | 'Similar'.
    search_value: 'Christopher Nolan', 'Brad Pitt', 'Science Fiction,Western' (Genre'da virgül = hepsi), 'time travel'.
    Sonuçlar Bayesian kalite skoruyla sıralanır.
    """
    raw_val = (search_value or "").strip()
    val = norm(raw_val)
    if category == "Keyword":
        val = norm(_TR_EN_THEME_MAP.get(val, _singularize_en(val)))
    heads = {
        "Director": "MATCH (p:Person)-[:DIRECTED]->(movie:Movie) WHERE p.name_norm CONTAINS $val "
                    "AND movie.vote_count >= $min_votes WITH DISTINCT movie",
        "Actor": "MATCH (p:Person)-[:ACTED_IN]->(movie:Movie) WHERE p.name_norm CONTAINS $val "
                 "AND movie.vote_count >= $min_votes WITH DISTINCT movie",
        "Keyword": "MATCH (movie:Movie)-[:HAS_KEYWORD]->(k:Keyword) WHERE k.name_norm CONTAINS $val "
                   "AND movie.vote_count >= $min_votes WITH DISTINCT movie",
        "Genre": "MATCH (movie:Movie)-[:HAS_GENRE]->(g:Genre) WHERE g.id IN $gids "
                 "AND movie.vote_count >= $min_votes "
                 "WITH movie, count(DISTINCT g) AS n WHERE n = size($gids) WITH movie",
        "Similar": "MATCH (ref:Movie) WHERE ref.title_norm = $val OR ref.original_title_norm = $val "
                   "WITH ref ORDER BY ref.vote_count DESC LIMIT 1 "
                   "MATCH (ref)-[:SIMILAR_TO]->(movie:Movie) WHERE movie.overview IS NOT NULL "
                   "WITH DISTINCT movie",
    }
    head = heads.get(category)
    if not head:
        return "[]"
    gids = genre_ids(search_value) if category == "Genre" else []
    if category == "Genre" and not gids:
        return "[]"
    cypher = f"{head} RETURN {_GRAPH_RETURN}, {_WR} AS wr ORDER BY wr DESC LIMIT $limit"
    cards = []
    try:
        cards = _run_graph(cypher, f"graph_{category.lower()}", val=val, gids=gids,
                           min_votes=min_votes, limit=limit)
    except Exception:
        cards = []

    # Neo4j'de Keyword bulunamadıysa özet/başlık veya yerel JSON yedeğine düş
    if not cards and category == "Keyword" and val:
        cards = _semantic_fallback_movies(val, [], 0.0, 10.0, 0, 0, min_votes, limit)
    return json.dumps(cards, ensure_ascii=False)


@mcp.tool()
def search_movies_semantically(
    semantic_query: str,
    exclude_genres: str = None,
    min_rating: float = 0.0,
    max_rating: float = 10.0,
    year_from: int = 0,
    year_to: int = 0,
    min_votes: int = 100,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """
    Anlamsal (vektör) arama: tema, atmosfer, olay örgüsü gibi soyut tarifler için.
    semantic_query'yi İNGİLİZCE yaz (film özetleri İngilizce; Türkçe sorgu eşleşmeyi bozar), sayısal/kadro
    kısıtlarını sorguya KOYMA; bunlar için ilgili parametreleri kullan.
    Örn: "a lonely astronaut drifting in deep space, slow contemplative tone".
    """
    excl_gids = genre_ids(exclude_genres)
    cards = []
    if embeddings_model is not None:
        try:
            vec = embeddings_model.embed_query(QUERY_PREFIX + semantic_query)
            cypher = f"""
            CALL db.index.vector.queryNodes('{VECTOR_INDEX}', $k, $vec) YIELD node AS movie, score
            WHERE movie.vote_count >= $min_votes
              AND movie.vote_average >= $min_rating AND movie.vote_average <= $max_rating
              AND ($year_from = 0 OR movie.year >= $year_from)
              AND ($year_to = 0 OR movie.year <= $year_to)
              AND NOT EXISTS {{ MATCH (movie)-[:HAS_GENRE]->(x:Genre) WHERE x.id IN $excl }}
            RETURN {_GRAPH_RETURN}, score
            ORDER BY score DESC LIMIT $limit
            """
            cards = _run_graph(cypher, "semantic", vec=vec, k=150, min_votes=min_votes,
                               min_rating=min_rating, max_rating=max_rating, year_from=year_from,
                               year_to=year_to, excl=excl_gids, limit=limit)
        except Exception:
            cards = []

    if not cards:
        cards = _semantic_fallback_movies(
            semantic_query=semantic_query,
            excl_gids=excl_gids,
            min_rating=min_rating,
            max_rating=max_rating,
            year_from=year_from,
            year_to=year_to,
            min_votes=min_votes,
            limit=limit,
        )
    return json.dumps(cards, ensure_ascii=False)


@mcp.tool()
def get_collaborative_recommendations(
    user_id: int,
    exclude_genres: str = None,
    min_rating: float = 0.0,
    max_rating: float = 10.0,
    year_from: int = 0,
    year_to: int = 0,
    min_votes: int = 100,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """
    Kullanıcı-film etkileşim grafına dayalı işbirlikçi filtreleme (GraphRAG Collaborative Filtering).
    Neo4j'de :User düğümü yoksa hata fırlatmadan boş liste döner.
    """
    excl_gids = genre_ids(exclude_genres)
    cypher = f"""
    MATCH (u:User {{id: $user_id}})-[:LIKED|FAVORITED]->(fav:Movie)<-[:LIKED|FAVORITED]-(peer:User)-[:LIKED|FAVORITED]->(movie:Movie)
    WHERE movie <> fav AND NOT (u)-[:LIKED|FAVORITED|DISLIKED]->(movie)
      AND movie.vote_count >= $min_votes
      AND movie.vote_average >= $min_rating AND movie.vote_average <= $max_rating
      AND ($year_from = 0 OR movie.year >= $year_from)
      AND ($year_to = 0 OR movie.year <= $year_to)
      AND NOT EXISTS {{ MATCH (movie)-[:HAS_GENRE]->(x:Genre) WHERE x.id IN $excl }}
    WITH movie, count(DISTINCT peer) AS overlap
    RETURN {_GRAPH_RETURN}, (overlap * 1.0 + 0.1 * {_WR}) AS score
    ORDER BY score DESC LIMIT $limit
    """
    try:
        cards = _run_graph(
            cypher, "collaborative",
            user_id=user_id, min_votes=min_votes, min_rating=min_rating, max_rating=max_rating,
            year_from=year_from, year_to=year_to, excl=excl_gids, limit=limit,
        )
    except Exception:
        cards = []
    return json.dumps(cards, ensure_ascii=False)


@mcp.tool()
def get_movie_card(title: str, year: int = 0) -> str:
    """
    Adı belli tek bir filmin TAM kartı (yönetmen, oyuncular, puan, fragman, platform).
    Kullanıcı belirli bir filmi soruyorsa kullan. Varsa yıl ver (aynı adlı filmleri ayırır).
    """
    mid = get_movie_id(title, year)
    card = fetch_movie_card(mid) if mid else None
    if not card:
        return "[]"
    return json.dumps([card], ensure_ascii=False)


@mcp.tool()
def enrich_movies(movie_ids: list[int]) -> str:
    """Seçilmiş filmlerin TAM kartlarını (fragman, platform, kadro) getirir. Verilen sıra korunur."""
    return json.dumps(cards_from_ids(list(movie_ids)), ensure_ascii=False)


# ================================================================ DİZİ (TV) DESTEĞİ
# Film araçlarının dizi karşılıkları. Kartlar film kartlarıyla AYNI anahtarları taşır
# ("movie_id", "Film", "Director"...), ek olarak "media_type": "tv" ve dizi alanları içerir.
SERIES_VECTOR_INDEX = os.getenv("SERIES_VECTOR_INDEX", "series_overview_index")
SERIES_BAYES_M = 100          # dizilerin oy sayıları filmlerden düşük; filmdeki 500 onları haksız cezalandırır

# TMDB'de DİZİ türleri filmlerden farklıdır: Action+Adventure -> 10759, SciFi+Fantasy -> 10765,
# War -> 10768 ve Horror / Thriller / Romance / History / Music dizilerde YOKTUR.
_TV_GENRES = {
    "Action": 10759, "Aksiyon": 10759, "Adventure": 10759, "Macera": 10759,
    "Animation": 16, "Animasyon": 16, "Comedy": 35, "Komedi": 35, "Crime": 80, "Suç": 80,
    "Documentary": 99, "Belgesel": 99, "Drama": 18, "Dram": 18, "Family": 10751, "Aile": 10751,
    "Fantasy": 10765, "Fantastik": 10765, "Science Fiction": 10765, "Sci-Fi": 10765, "Bilim Kurgu": 10765,
    "Mystery": 9648, "Gizem": 9648, "War": 10768, "Savaş": 10768, "Western": 37,
}
TV_GENRE_LOOKUP = {norm(k): v for k, v in _TV_GENRES.items()}
# Dizi türü olmayanlar -> anahtar kelime olarak aranır (örn. "Horror" -> keyword "horror")
_TV_GENRE_AS_KEYWORD = {norm(k): v for k, v in {
    "Horror": "horror", "Korku": "horror", "Thriller": "thriller", "Gerilim": "thriller",
    "Romance": "romance", "Romantik": "romance", "History": "history", "Tarih": "history",
    "Music": "music", "Müzik": "music"}.items()}


def tv_genre_split(csv) -> tuple:
    """'Horror, Science Fiction' -> ([10765], ['horror']): (dizi tür id'leri, anahtar kelimeye düşenler)."""
    ids, kws = [], []
    for g in (csv or "").split(","):
        n = norm(g)
        if n in TV_GENRE_LOOKUP:
            if TV_GENRE_LOOKUP[n] not in ids:
                ids.append(TV_GENRE_LOOKUP[n])
        elif n in _TV_GENRE_AS_KEYWORD and _TV_GENRE_AS_KEYWORD[n] not in kws:
            kws.append(_TV_GENRE_AS_KEYWORD[n])
    return ids, kws


@ttl_cache(ttl=3600)
def get_series_id(title: str, year: int = 0):
    params = {"query": title, "language": "en-US"}
    if year:
        params["first_air_date_year"] = year
    try:
        results = tmdb_get("/search/tv", **params).get("results", [])
    except Exception:
        return None
    if not results:
        return None
    t = norm(title)
    exact = [r for r in results if norm(r.get("name", "")) == t or norm(r.get("original_name", "")) == t]
    return max(exact or results, key=lambda r: r.get("vote_count", 0))["id"]


def _series_card_from_tmdb(r: dict, source: str, relaxed=None) -> dict:
    tv_names = {v: k for k, v in {"Action & Adventure": 10759, "Animation": 16, "Comedy": 35, "Crime": 80,
                                  "Documentary": 99, "Drama": 18, "Family": 10751, "Kids": 10762,
                                  "Mystery": 9648, "News": 10763, "Reality": 10764,
                                  "Sci-Fi & Fantasy": 10765, "Soap": 10766, "Talk": 10767,
                                  "War & Politics": 10768, "Western": 37}.items()}
    return lite_card(
        mid=r["id"], title=r.get("name"), year=(r.get("first_air_date") or "")[:4],
        rating=r.get("vote_average"), votes=r.get("vote_count"),
        genres=[tv_names.get(g) for g in r.get("genre_ids", [])],
        overview=r.get("overview"), poster=r.get("poster_path"),
        source=source, relaxed=relaxed, media_type="tv")


_SERIES_RETURN = """
  movie.id AS id, movie.name AS title, movie.year AS year,
  movie.vote_average AS rating, movie.vote_count AS votes,
  movie.overview AS overview, movie.poster_path AS poster,
  [(movie)-[:HAS_GENRE]->(g:Genre) | g.name] AS genres,
  [(movie)<-[:CREATED]-(d:Person) | d.name] AS director,
  [(movie)<-[:ACTED_IN]-(c:Person) | c.name] AS cast
"""
_WR_SERIES = _WR      # aynı formül; m parametresi çağrıda SERIES_BAYES_M olarak verilir


def _run_series_graph(cypher: str, source: str, **params) -> list:
    params.setdefault("m", SERIES_BAYES_M)
    params.setdefault("c", BAYES_C)
    with neo4j_driver.session(database=NEO4J_DATABASE) as session:
        rows = [dict(r) for r in session.run(cypher, **params)]
    return [
        lite_card(
            mid=r["id"], title=r["title"], year=r.get("year") or "", rating=r.get("rating"),
            votes=r.get("votes"), genres=r.get("genres") or [], overview=r.get("overview"),
            poster=r.get("poster"), director=", ".join(r.get("director") or []),
            cast=", ".join((r.get("cast") or [])[:4]), source=source, media_type="tv",
            score=r.get("score") if r.get("score") is not None else r.get("wr"))
        for r in rows
    ]


def _semantic_fallback_series(
    semantic_query: str,
    excl_gids: list[int],
    min_rating: float,
    max_rating: float,
    year_from: int,
    year_to: int,
    min_votes: int,
    limit: int,
) -> list[dict]:
    """
    DİZİler için hibrit semantik yedek arama (Render'da yerel Ollama olmadığında veya
    vektör index'i boş döndüğünde çalışır):
      1) Neo4j :Series üzerinde Keyword + Overview/Tagline/Name eşleşmesi
      2) TMDB /discover/tv (with_keywords) + /search/tv
      3) Yerel series_data.json yedeği
    """
    terms = _extract_semantic_terms(semantic_query)
    if not terms:
        terms = [norm(semantic_query)[:40]] if norm(semantic_query) else []
    out, seen = [], set()
    excl_set = set(excl_gids or [])

    # 1. Neo4j :Series üzerinde anahtar kelime + özet/başlık araması
    if terms:
        cypher = f"""
        MATCH (movie:Series)
        WHERE movie.vote_count >= $min_votes
          AND movie.vote_average >= $min_rating AND movie.vote_average <= $max_rating
          AND ($year_from = 0 OR movie.year >= $year_from)
          AND ($year_to = 0 OR movie.year <= $year_to)
          AND NOT EXISTS {{ MATCH (movie)-[:HAS_GENRE]->(x:Genre) WHERE x.id IN $excl }}
        OPTIONAL MATCH (movie)-[:HAS_KEYWORD]->(k:Keyword)
        WITH movie, collect(k.name_norm) AS kw_names,
             toLower(coalesce(movie.overview, '')) AS ov,
             toLower(coalesce(movie.tagline, '')) AS tg,
             coalesce(movie.name_norm, '') AS tn
        WITH movie,
             size([t IN $terms WHERE any(kw IN kw_names WHERE kw CONTAINS t)]) AS kw_hits,
             size([t IN $terms WHERE ov CONTAINS t OR tg CONTAINS t OR tn CONTAINS t]) AS txt_hits
        WHERE (kw_hits + txt_hits) > 0
        WITH movie, (kw_hits * 2.5 + txt_hits * 1.0) + 0.15 * {_WR_SERIES} AS score
        RETURN {_SERIES_RETURN}, score
        ORDER BY score DESC LIMIT $limit
        """
        try:
            for c in _run_series_graph(
                cypher, "semantic_series",
                terms=terms, min_votes=min_votes, min_rating=min_rating, max_rating=max_rating,
                year_from=year_from, year_to=year_to, excl=list(excl_set), limit=limit,
            ):
                if c["movie_id"] not in seen:
                    seen.add(c["movie_id"])
                    out.append(c)
        except Exception:
            pass

    # 2. TMDB Keyword Discover (/discover/tv) + TMDB Metin Araması (/search/tv)
    if len(out) < limit and terms:
        kids = []
        for t in terms[:3]:
            kid = get_keyword_id(t)
            if kid is not None and str(kid) not in kids:
                kids.append(str(kid))
        if kids:
            params = {
                "include_adult": "false",
                "language": "en-US",
                "sort_by": "vote_count.desc",
                "vote_count.gte": max(int(min_votes), 1),
                "with_keywords": "|".join(kids),
            }
            if excl_set:
                params["without_genres"] = "|".join(map(str, excl_set))
            if min_rating > 0:
                params["vote_average.gte"] = min_rating
            if max_rating < 10:
                params["vote_average.lte"] = max_rating
            if year_from:
                params["first_air_date.gte"] = f"{year_from}-01-01"
            if year_to:
                params["first_air_date.lte"] = f"{year_to}-12-31"
            try:
                res = tmdb_get("/discover/tv", page=1, **params).get("results", [])
                res = [r for r in res if not (set(r.get("genre_ids", [])) & excl_set)]
                res.sort(
                    key=lambda r: bayes_score(r.get("vote_average"), r.get("vote_count"), m=SERIES_BAYES_M),
                    reverse=True,
                )
                for r in res:
                    c = _series_card_from_tmdb(r, "semantic_series")
                    if c["movie_id"] not in seen:
                        seen.add(c["movie_id"])
                        out.append(c)
            except Exception:
                pass

    if len(out) < limit and terms:
        try:
            q_str = " ".join(terms[:2])
            res = tmdb_get("/search/tv", query=q_str, language="en-US", include_adult="false").get("results", [])
            valid_res = []
            for r in res:
                vc = int(r.get("vote_count") or 0)
                va = float(r.get("vote_average") or 0.0)
                fd = (r.get("first_air_date") or "")[:4]
                yr = int(fd) if fd.isdigit() else 0
                if vc < min_votes or va < min_rating or va > max_rating:
                    continue
                if year_from and (not yr or yr < year_from):
                    continue
                if year_to and (not yr or yr > year_to):
                    continue
                if set(r.get("genre_ids", [])) & excl_set:
                    continue
                valid_res.append(r)
            valid_res.sort(
                key=lambda r: bayes_score(r.get("vote_average"), r.get("vote_count"), m=SERIES_BAYES_M),
                reverse=True,
            )
            for r in valid_res:
                c = _series_card_from_tmdb(r, "semantic_series")
                if c["movie_id"] not in seen:
                    seen.add(c["movie_id"])
                    out.append(c)
        except Exception:
            pass

    # 3. Yerel series_data.json yedeği
    if len(out) < limit and terms:
        for c in _search_local_json(
            True, terms, list(excl_set), min_rating, max_rating,
            year_from, year_to, min_votes, limit, "semantic_series",
        ):
            if c["movie_id"] not in seen:
                seen.add(c["movie_id"])
                out.append(c)

    return out[:limit]


def fetch_series_card(series_id, fallback: dict | None = None) -> dict | None:
    """Dizi için TAM kart (fragman, platform, kadro, sezon/bölüm). Yalnızca son seçilenler için çağrılır."""
    fallback = fallback or {}
    try:
        d = tmdb_get(f"/tv/{series_id}", language="tr-TR",
                     append_to_response="videos,aggregate_credits,watch/providers",
                     include_video_language="en,tr,null")
    except Exception:
        return None
    overview = d.get("overview") or fallback.get("overview")
    if not overview:
        try:
            overview = tmdb_get(f"/tv/{series_id}", language="en-US").get("overview")
        except Exception:
            overview = None
    creators = [c["name"] for c in d.get("created_by", [])]
    cast = [c["name"] for c in d.get("aggregate_credits", {}).get("cast", [])[:4]]
    tr = d.get("watch/providers", {}).get("results", {}).get("TR", {})
    flat = [p["provider_name"] for p in tr.get("flatrate", [])]
    rentbuy = sorted({p["provider_name"] for k in ("rent", "buy") for p in tr.get(k, [])})
    if flat:
        platform = ", ".join(flat)
    elif rentbuy:
        platform = "Kiralık/satın alınabilir: " + ", ".join(rentbuy[:4])
    else:
        platform = "Abonelik platformunda yok"
    poster = d.get("poster_path") or fallback.get("poster_path")
    return {
        "movie_id": str(d.get("id", series_id)),
        "media_type": "tv",
        "Film": d.get("name") or d.get("original_name") or fallback.get("title", ""),
        "Director": ", ".join(creators) or fallback.get("director", ""),     # dizide: yaratıcı(lar)
        "Cast": ", ".join(cast) or fallback.get("cast", ""),
        "Yıl": (d.get("first_air_date") or "")[:4],
        "TMDB Puanı": f"{d.get('vote_average', 0):.1f}",
        "vote_count": int(d.get("vote_count") or 0),
        "Türler": ", ".join(g["name"] for g in d.get("genres", [])),
        "Özet": _clip(overview) or "Özet bulunamadı.",
        "Poster": f"https://image.tmdb.org/t/p/w500{poster}" if poster else "",
        "Fragman": _pick_trailer(d.get("videos", {}).get("results", [])),
        "Şu Anki Platform(lar)": platform,
        "Neden Önerildi": "",
        "Sezon Sayısı": d.get("number_of_seasons"),
        "Bölüm Sayısı": d.get("number_of_episodes"),
        "Durum": d.get("status", ""),
    }


@mcp.tool()
@ttl_cache(ttl=1800)
def search_series_by_filters(
    genre_name: str = None,
    exclude_genres: str = None,
    keyword: str = None,
    min_rating: float = 0.0,
    max_rating: float = 10.0,
    year_from: int = 0,
    year_to: int = 0,
    min_votes: int = 50,
    sort_by: str = "vote_average.desc",
    only_streamable: bool = False,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """
    TMDB canlı DİZİ araması (hafif aday kartları). Oyuncu/yaratıcı filtresi YOKTUR (TMDB dizilerde desteklemez);
    onlar için search_series_in_graph kullan.
    genre_name: virgülle ayrılmış türler. Dizilerde Horror/Thriller/Romance/History/Music türü yoktur;
    bunlar otomatik anahtar kelime aramasına çevrilir. year_from/year_to ilk yayın yılına göredir.
    """
    params = {"include_adult": "false", "language": "en-US", "sort_by": sort_by,
              "vote_count.gte": max(int(min_votes), 1)}
    incl, kw_from_genre = tv_genre_split(genre_name)
    excl, _ = tv_genre_split(exclude_genres)
    if incl:
        params["with_genres"] = ",".join(map(str, incl))
    if excl:
        params["without_genres"] = "|".join(map(str, excl))

    explicit_kid = get_keyword_id(keyword) if keyword else None
    genre_kid = get_keyword_id(kw_from_genre[0]) if (not explicit_kid and kw_from_genre) else None
    kid = explicit_kid or genre_kid
    if kid:
        params["with_keywords"] = kid
    if min_rating > 0:
        params["vote_average.gte"] = min_rating
    if max_rating < 10:
        params["vote_average.lte"] = max_rating
    if year_from:
        params["first_air_date.gte"] = f"{year_from}-01-01"
    if year_to:
        params["first_air_date.lte"] = f"{year_to}-12-31"
    if only_streamable:
        params.update(watch_region="TR", with_watch_monetization_types="flatrate")

    def fetch(p):
        out = []
        seen_ids = set()
        for page in (1, 2):
            if len(out) >= limit:
                break
            try:
                res = tmdb_get("/discover/tv", page=page, **p).get("results", [])
            except Exception:
                break
            for r in res:
                if r.get("id") and r["id"] not in seen_ids:
                    seen_ids.add(r["id"])
                    out.append(r)
            if len(res) < 20:
                break
        # Düşük oylu şişirilmiş animasyon/spin-off kayıtlarının ünlü dizileri ezmesini önle
        if p.get("sort_by") == "vote_average.desc":
            try:
                pop_p = {**p, "sort_by": "popularity.desc"}
                for r in tmdb_get("/discover/tv", page=1, **pop_p).get("results", []):
                    if r.get("id") and r["id"] not in seen_ids:
                        seen_ids.add(r["id"])
                        out.append(r)
            except Exception:
                pass
            out.sort(
                key=lambda r: bayes_score(r.get("vote_average"), r.get("vote_count"), m=SERIES_BAYES_M),
                reverse=True,
            )
        return out

    relaxed = []
    series = fetch(params)

    # Açık keyword ("vampire" vb.) verildiyse ÖNCE türleri gevşet, keyword'ü ASLA ilk adımda silme!
    if not series and explicit_kid:
        if len(incl) > 1:
            params["with_genres"] = "|".join(map(str, incl))
            relaxed.append("genres_and_to_or")
            series = fetch(params)
        if not series and incl:
            params.pop("with_genres", None)
            relaxed.append("genres_dropped_for_keyword")
            series = fetch(params)

    if not series and kid:
        if incl:
            params["with_genres"] = ",".join(map(str, incl))
        params.pop("with_keywords", None)
        relaxed.append("keyword")
        series = fetch(params)
    if not series and len(incl) > 1:
        params["with_genres"] = "|".join(map(str, incl))
        if "genres_and_to_or" not in relaxed:
            relaxed.append("genres_and_to_or")
        series = fetch(params)

    excl_set = set(excl)
    series = [s for s in series if not (set(s.get("genre_ids", [])) & excl_set)]
    return json.dumps([_series_card_from_tmdb(s, "tmdb_filter", relaxed or None) for s in series[:limit]],
                      ensure_ascii=False)


@mcp.tool()
def search_series_in_graph(category: str, search_value: str, limit: int = DEFAULT_LIMIT,
                           min_votes: int = 30) -> str:
    """
    Yerel Neo4j grafında DİZİ araması (hızlı, hafif kartlar).
    category: 'Director' (dizinin YARATICISI) | 'Actor' | 'Genre' | 'Keyword' | 'Similar'.
    """
    raw_val = (search_value or "").strip()
    val = norm(raw_val)
    if category == "Keyword":
        val = norm(_TR_EN_THEME_MAP.get(val, _singularize_en(val)))
    gids = tv_genre_split(search_value)[0] if category == "Genre" else []
    heads = {
        "Director": "MATCH (p:Person)-[:CREATED]->(movie:Series) WHERE p.name_norm CONTAINS $val "
                    "AND movie.vote_count >= $min_votes WITH DISTINCT movie",
        "Actor": "MATCH (p:Person)-[:ACTED_IN]->(movie:Series) WHERE p.name_norm CONTAINS $val "
                 "AND movie.vote_count >= $min_votes WITH DISTINCT movie",
        "Keyword": "MATCH (movie:Series)-[:HAS_KEYWORD]->(k:Keyword) WHERE k.name_norm CONTAINS $val "
                   "AND movie.vote_count >= $min_votes WITH DISTINCT movie",
        "Genre": "MATCH (movie:Series)-[:HAS_GENRE]->(g:Genre) WHERE g.id IN $gids "
                 "AND movie.vote_count >= $min_votes "
                 "WITH movie, count(DISTINCT g) AS n WHERE n = size($gids) WITH movie",
        "Similar": "MATCH (ref:Series) WHERE ref.name_norm = $val OR ref.original_name_norm = $val "
                   "WITH ref ORDER BY ref.vote_count DESC LIMIT 1 "
                   "MATCH (ref)-[:SIMILAR_TO]->(movie:Series) WHERE movie.overview IS NOT NULL "
                   "WITH DISTINCT movie",
    }
    head = heads.get(category)
    if not head or (category == "Genre" and not gids):
        return "[]"
    cypher = f"{head} RETURN {_SERIES_RETURN}, {_WR_SERIES} AS wr ORDER BY wr DESC LIMIT $limit"
    cards = []
    try:
        cards = _run_series_graph(cypher, f"graph_series_{category.lower()}", val=val, gids=gids,
                                  min_votes=min_votes, limit=limit)
    except Exception:
        cards = []

    if not cards and category == "Keyword" and val:
        cards = _semantic_fallback_series(val, [], 0.0, 10.0, 0, 0, min_votes, limit)
    return json.dumps(cards, ensure_ascii=False)


@mcp.tool()
def search_series_semantically(
    semantic_query: str,
    exclude_genres: str = None,
    min_rating: float = 0.0,
    max_rating: float = 10.0,
    year_from: int = 0,
    year_to: int = 0,
    min_votes: int = 30,
    limit: int = DEFAULT_LIMIT,
) -> str:
    """
    DİZİler için anlamsal (vektör) arama: tema, atmosfer, olay örgüsü. semantic_query İNGİLİZCE olmalı.
    """
    excl_gids = tv_genre_split(exclude_genres)[0]
    cards = []
    if embeddings_model is not None:
        try:
            vec = embeddings_model.embed_query(QUERY_PREFIX + semantic_query)
            cypher = f"""
            CALL db.index.vector.queryNodes('{SERIES_VECTOR_INDEX}', $k, $vec) YIELD node AS movie, score
            WHERE movie.vote_count >= $min_votes
              AND movie.vote_average >= $min_rating AND movie.vote_average <= $max_rating
              AND ($year_from = 0 OR movie.year >= $year_from)
              AND ($year_to = 0 OR movie.year <= $year_to)
              AND NOT EXISTS {{ MATCH (movie)-[:HAS_GENRE]->(x:Genre) WHERE x.id IN $excl }}
            RETURN {_SERIES_RETURN}, score
            ORDER BY score DESC LIMIT $limit
            """
            cards = _run_series_graph(cypher, "semantic_series", vec=vec, k=150, min_votes=min_votes,
                                      min_rating=min_rating, max_rating=max_rating, year_from=year_from,
                                      year_to=year_to, excl=excl_gids, limit=limit)
        except Exception:
            cards = []

    if not cards:
        cards = _semantic_fallback_series(
            semantic_query=semantic_query,
            excl_gids=excl_gids,
            min_rating=min_rating,
            max_rating=max_rating,
            year_from=year_from,
            year_to=year_to,
            min_votes=min_votes,
            limit=limit,
        )
    return json.dumps(cards, ensure_ascii=False)


@mcp.tool()
@ttl_cache(ttl=1800)
def get_similar_series(series_title: str, year: int = 0, limit: int = DEFAULT_LIMIT) -> str:
    """Verilen diziye benzer diziler: TMDB 'recommendations' + dizinin kendi embedding komşuları."""
    sid = get_series_id(series_title, year)
    if not sid:
        return "[]"
    out, seen = [], {str(sid)}
    try:
        recs = tmdb_get(f"/tv/{sid}/recommendations", language="en-US").get("results", []) \
            or tmdb_get(f"/tv/{sid}/similar", language="en-US").get("results", [])
        for r in recs:
            c = _series_card_from_tmdb(r, "tmdb_rec")
            if c["movie_id"] not in seen:
                seen.add(c["movie_id"])
                out.append(c)
    except Exception:
        pass
    try:
        cypher = f"""
        MATCH (ref:Series {{id: $id}}) WHERE ref.embedding IS NOT NULL
        CALL db.index.vector.queryNodes('{SERIES_VECTOR_INDEX}', $k, ref.embedding) YIELD node AS movie, score
        WHERE movie.id <> $id AND movie.vote_count >= 30
        RETURN {_SERIES_RETURN}, score
        ORDER BY score DESC LIMIT $limit
        """
        for c in _run_series_graph(cypher, "vector_neighbor_series", id=sid, k=60, limit=limit):
            if c["movie_id"] not in seen:
                seen.add(c["movie_id"])
                out.append(c)
    except Exception:
        pass
    return json.dumps(out[: limit * 2], ensure_ascii=False)


@mcp.tool()
def get_series_card(title: str, year: int = 0) -> str:
    """Adı belli tek bir dizinin TAM kartı (yaratıcı, oyuncular, puan, sezon/bölüm, fragman, platform)."""
    sid = get_series_id(title, year)
    card = fetch_series_card(sid) if sid else None
    return json.dumps([card], ensure_ascii=False) if card else "[]"


@mcp.tool()
def enrich_series(series_ids: list[int]) -> str:
    """Seçilmiş dizilerin TAM kartlarını getirir. Verilen sıra korunur."""
    with ThreadPoolExecutor(max_workers=5) as ex:
        cards = list(ex.map(fetch_series_card, list(series_ids)))
    return json.dumps([c for c in cards if c], ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()