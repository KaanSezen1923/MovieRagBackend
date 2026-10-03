"""
MCP sunucusu: film arama araçları.

MİMARİ DEĞİŞİKLİK — "önce aday, sonra zenginleştir":
  Eski sürümde her arama aracı, döndürdüğü HER filmi TMDB'ye 1 ağır istekle (videos+credits+providers)
  zenginleştiriyordu; LLM sonra bunların sadece 1-5'ini seçiyordu. Şimdi:
    * Arama araçları (search_*, get_similar_movies) HAFİF aday kartları döner: Neo4j'den ya da
      TMDB liste yanıtından, ek istek olmadan.
    * enrich_movies(ids) yalnızca SON seçilen filmler için fragman/platform/kadro getirir.

DÜZELTİLEN HATALAR:
  * Graf araması film adını yeniden TMDB'de aratıyordu (Oldboy remake, 'Piyanist', 'Deniz Feneri'
    çakışmaları). Artık Neo4j'deki TMDB id'si doğrudan kullanılıyor.
  * Arama alanları norm() ile normalize (Inception/ınception hatası).
  * Sıralama: vote_average DESC tek başına 3 oylu 10.0'ları üste çıkarıyordu -> Bayesian skor + min oy.
  * Discover popularity.desc sıralaması yeni çıkanlara kayıyordu -> vote_average.desc + min oy (varsayılan).
  * exclude_genres, max_rating, yıl aralığı destekleniyor (Q12, Q18, Q36 bu yüzden çözülemiyordu).
  * Anlamsal arama: filtreli, k=150 geniş havuz, doğru embedding öneki, film 'kendi vektörü'yle benzer arama.
  * Tüm HTTP isteklerinde timeout; _pick_trailer üzerindeki anlamsız cache kaldırıldı.
  * Sessiz kısıt gevşetme artık kartlarda 'relaxed' alanıyla görünür.
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from neo4j import GraphDatabase
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from movie_common import QUERY_PREFIX, VECTOR_INDEX, bayes_score, norm

load_dotenv()
AUTH_KEY = os.getenv("AUTH_KEY")
BASE_URL = "https://api.themoviedb.org/3"
DEFAULT_LIMIT = 20
BAYES_M, BAYES_C = 500, 6.5

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

try:
    from langchain_ollama import OllamaEmbeddings
    from movie_common import EMBED_MODEL
    embeddings_model = OllamaEmbeddings(model=EMBED_MODEL)
except Exception:
    embeddings_model = None

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


# ---------------------------------------------------------------- id çözümleme
@ttl_cache(ttl=86400)
def get_person_id(name: str):
    res = tmdb_get("/search/person", query=name, language="en-US").get("results", [])
    return res[0]["id"] if res else None


@ttl_cache(ttl=86400)
def get_keyword_id(keyword: str):
    res = tmdb_get("/search/keyword", query=keyword).get("results", [])
    return res[0]["id"] if res else None


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
    # Birebir isim eşleşmelerini bul
    exact = [r for r in results if norm(r.get("title", "")) == t or norm(r.get("original_title", "")) == t]
    pool = exact or results
    
    # Eğer yıl parametresi geldiyse, havuzu o yıla göre daralt
    if year:
        year_matches = [
            r for r in pool 
            if r.get("release_date") and r.get("release_date").startswith(str(year))
        ]
        if year_matches:
            pool = year_matches

    # Daraltılmış havuz içinde en popüler olanı döndür
    return max(pool, key=lambda r: r.get("vote_count", 0))["id"]


# ---------------------------------------------------------------- kartlar
def _clip(text: str, n: int = 400) -> str:
    """Kelime sınırında kes (eski [:200] cümle ortasında kesiyordu)."""
    text = (text or "").strip()
    if len(text) <= n:
        return text
    return text[:n].rsplit(" ", 1)[0].rstrip(",;:- ") + "…"


def lite_card(*, mid, title, year="", rating=None, votes=None, genres=(), overview="", poster=None,
              director="", cast="", source="", score=None, relaxed=None) -> dict:
    """Ek API isteği gerektirmeyen HAFİF aday kartı. Anahtarlar tam kartla uyumlu."""
    card = {
        "movie_id": str(mid),
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
        for page in (1, 2):
            if len(out) >= limit:
                break
            try:
                res = tmdb_get("/discover/movie", page=page, **p).get("results", [])
            except Exception:
                break
            out.extend(res)
            if len(res) < 20:
                break
        return out

    relaxed = []
    movies = fetch(params)
    # Kısıt gevşetme: artık SESSİZ değil, kartlarda 'relaxed' olarak işaretlenir.
    if not movies and kid:
        params.pop("with_keywords", None)
        relaxed.append("keyword")
        movies = fetch(params)
    if not movies and len(incl) > 1:
        params["with_genres"] = "|".join(map(str, incl))            # pipe = OR
        relaxed.append("genres_and_to_or")
        movies = fetch(params)

    excl_set = set(excl)
    movies = [m for m in movies if not (set(m.get("genre_ids", [])) & excl_set)]   # garanti için ayrıca süz
    return json.dumps(_cards_from_tmdb_results(movies[:limit], "tmdb_filter", relaxed or None),
                      ensure_ascii=False)


@mcp.tool()
@ttl_cache(ttl=1800)
def get_similar_movies(movie_title: str, year: int = 0, limit: int = DEFAULT_LIMIT) -> str:
    """
    Verilen filme benzer filmler (hafif aday kartları). İki kaynağı birleştirir:
    TMDB 'recommendations' (izleyici davranışı) + filmin KENDİ embedding'ine en yakın komşular (konu/atmosfer).
    """
    mid = get_movie_id(movie_title, year)
    if not mid:
        return "[]"
    out, seen = [], {mid}

    try:
        recs = tmdb_get(f"/movie/{mid}/recommendations", language="en-US").get("results", [])
        if not recs:
            recs = tmdb_get(f"/movie/{mid}/similar", language="en-US").get("results", [])
        for c in _cards_from_tmdb_results(recs, "tmdb_rec"):
            if c["movie_id"] not in seen:
                seen.add(c["movie_id"]); out.append(c)
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
                seen.add(c["movie_id"]); out.append(c)
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
    val = norm(search_value or "")
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
    try:
        cards = _run_graph(cypher, f"graph_{category.lower()}", val=val, gids=gids,
                           min_votes=min_votes, limit=limit)
    except Exception as e:
        return f"Graf arama hatası: {e}"
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
    if not embeddings_model:
        return "Semantik arama modeli aktif değil."
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
        # Vektör index'i filtreden ÖNCE k kayıt döner; filtre sonrası yeterli kalsın diye k büyük.
        cards = _run_graph(cypher, "semantic", vec=vec, k=150, min_votes=min_votes,
                           min_rating=min_rating, max_rating=max_rating, year_from=year_from,
                           year_to=year_to, excl=genre_ids(exclude_genres), limit=limit)
    except Exception as e:
        return f"Semantik arama hatası: {e}"
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


if __name__ == "__main__":
    mcp.run()