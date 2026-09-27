import os
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from mcp.server.fastmcp import FastMCP
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
import time
from functools import wraps

# Yapılandırma
load_dotenv()
AUTH_KEY = os.getenv("AUTH_KEY")
BASE_URL = "https://api.themoviedb.org/3"
MAX_MOVIES = 5

# Neo4j bağlantı ayarları (ortam değişkenlerinden alınır)
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")
neo4j_driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

mcp = FastMCP("TMDB Movie Explorer")

headers = {
    "accept": "application/json",
    "Authorization": f"Bearer {AUTH_KEY}"
}

# İngilizce, Türkçe ve alternatif kullanımları destekleyen genişletilmiş tür sözlüğü
GENRE_DICT = {
    "Action": 28, "Aksiyon": 28,
    "Adventure": 12, "Macera": 12,
    "Animation": 16, "Animasyon": 16,
    "Comedy": 35, "Komedi": 35,
    "Crime": 80, "Suç": 80,
    "Documentary": 99, "Belgesel": 99,
    "Drama": 18, "Dram": 18,
    "Family": 10751, "Aile": 10751,
    "Fantasy": 14, "Fantastik": 14,
    "History": 36, "Tarih": 36,
    "Horror": 27, "Korku": 27,
    "Music": 10402, "Müzik": 10402,
    "Mystery": 9648, "Gizem": 9648,
    "Romance": 10749, "Romantik": 10749,
    "Science Fiction": 878, "Sci-Fi": 878, "Bilim Kurgu": 878,
    "TV Movie": 10770, "TV Filmi": 10770,
    "Thriller": 53, "Gerilim": 53,
    "War": 10752, "Savaş": 10752,
    "Western": 37
}

REVERSE_GENRE_DICT = {
    28: "Action", 12: "Adventure", 16: "Animation", 35: "Comedy",
    80: "Crime", 99: "Documentary", 18: "Drama", 10751: "Family",
    14: "Fantasy", 36: "History", 27: "Horror", 10402: "Music",
    9648: "Mystery", 10749: "Romance", 878: "Science Fiction",
    10770: "TV Movie", 53: "Thriller", 10752: "War", 37: "Western"
}
def ttl_cache(ttl=3600, maxsize=500):
    """Basit, bağımlılık gerektirmeyen in-memory TTL Cache (Varsayılan: 1 saat)"""
    cache = {}
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            key = str(args) + str(kwargs)
            if key in cache:
                result, timestamp = cache[key]
                if time.time() - timestamp < ttl:
                    return result
                else:
                    del cache[key]
            
            result = func(*args, **kwargs)
            if len(cache) >= maxsize:
                cache.pop(next(iter(cache)))
            cache[key] = (result, time.time())
            return result
        return wrapper
    return decorator
# --- Yardımcı Fonksiyonlar ---
@ttl_cache(ttl=86400)
def get_person_id(name: str):
    search_url = f"{BASE_URL}/search/person"
    params = {"query": name, "language": "en-US"}
    resp = requests.get(search_url, headers=headers, params=params)
    results = resp.json().get('results', [])
    return results[0]['id'] if results else None
@ttl_cache(ttl=86400)
def get_keyword_id(keyword: str):
    search_url = f"{BASE_URL}/search/keyword"
    params = {"query": keyword}
    resp = requests.get(search_url, headers=headers, params=params)
    results = resp.json().get('results', [])
    return results[0]['id'] if results else None
@ttl_cache(ttl=3600)
def get_movie_id(title: str):
    search_url = f"{BASE_URL}/search/movie"
    params = {"query": title, "language": "en-US"}
    try:
        resp = requests.get(search_url, headers=headers, params=params, timeout=8)
        results = resp.json().get('results', [])
    except Exception:
        return None
    return results[0]['id'] if results else None
@ttl_cache(ttl=3600)
def _pick_trailer(videos: list) -> str:
    yt = [v for v in videos if v.get("site") == "YouTube" and v.get("key")]
    # önce resmi olanlar, sonra Trailer > Teaser
    yt.sort(key=lambda v: (not v.get("official", False), v.get("type") != "Trailer"))
    for v in yt:
        if v.get("type") in ("Trailer", "Teaser"):
            return f"https://www.youtube.com/watch?v={v['key']}"
    return ""  # Clip/Featurette/Vimeo asla dönmesin

def fetch_movie_card(movie_id: int, fallback: dict | None = None) -> dict | None:
    """Tek istekle detay + video + kadro + platform bilgisini getirip kart üretir."""
    fallback = fallback or {}
    try:
        r = requests.get(
            f"{BASE_URL}/movie/{movie_id}", headers=headers, timeout=8,
            params={
                "language": "tr-TR",
                "append_to_response": "videos,credits,watch/providers",
                "include_video_language": "en,tr,null",
            },
        )
        r.raise_for_status()
        d = r.json()
    except Exception:
        return None

    credits = d.get("credits", {})
    directors = [c["name"] for c in credits.get("crew", []) if c.get("job") == "Director"]
    cast = [c["name"] for c in credits.get("cast", [])[:4]]
    tr = d.get("watch/providers", {}).get("results", {}).get("TR", {})
    platforms = [p["provider_name"] for p in tr.get("flatrate", [])]
    poster = d.get("poster_path") or fallback.get("poster_path")

    return {
        "movie_id": str(d.get("id", movie_id)),
        "Film": d.get("title") or d.get("original_title") or fallback.get("title", ""),
        "Director": ", ".join(directors) or fallback.get("director", ""),
        "Cast": ", ".join(cast) or fallback.get("cast", ""),
        "Yıl": (d.get("release_date") or "")[:4],
        "TMDB Puanı": f"{d.get('vote_average', 0):.1f}",
        "Türler": ", ".join(REVERSE_GENRE_DICT.get(g["id"], g["name"]) for g in d.get("genres", [])),
        "Özet": (d.get("overview") or fallback.get("overview") or "Özet bulunamadı.")[:200],
        "Poster": f"https://image.tmdb.org/t/p/w500{poster}" if poster else "",
        "Fragman": _pick_trailer(d.get("videos", {}).get("results", [])),
        "Şu Anki Platform(lar)": ", ".join(platforms) or "Abonelik platformunda yok",
        "Neden Önerildi": "",
    }

def cards_from_ids(ids: list) -> list:
    with ThreadPoolExecutor(max_workers=5) as ex:
        cards = list(ex.map(fetch_movie_card, ids))
    return [c for c in cards if c]

# --- MCP Tool Tanımları ---

@mcp.tool()
@ttl_cache(ttl=1800)
def search_movies_by_filters(
    genre_name: str = None,
    actor_name: str = None,
    director_name: str = None,
    keyword: str = None,
    min_rating: float = 0.0
) -> str:
    """
    Belirli kriterlere göre film araması yapar.
    genre_name: Tek tür ("Fantasy") veya virgülle ayrılmış birden fazla tür ("Science Fiction,Fantasy") olabilir.
    """
    discover_url = f"{BASE_URL}/discover/movie"
    params = {
        "include_adult": "false",
        "language": "tr-TR",
        "page": 1,
        "sort_by": "popularity.desc"
    }

    if genre_name:
        genre_ids = []
        for g in genre_name.split(","):
            clean_g = g.strip().title() if not g.strip().isupper() else g.strip()
            gid = GENRE_DICT.get(clean_g) or GENRE_DICT.get(g.strip())
            if gid and str(gid) not in genre_ids:
                genre_ids.append(str(gid))
        if genre_ids:
            # Çoklu türlerde virgül (AND) mantığıyla başla
            params["with_genres"] = ",".join(genre_ids)

    if actor_name:
        aid = get_person_id(actor_name)
        if aid:
            params["with_cast"] = aid

    if director_name:
        did = get_person_id(director_name)
        if did:
            params["with_crew"] = did

    kid = None
    if keyword:
        kid = get_keyword_id(keyword)
        if kid:
            params["with_keywords"] = kid

    if min_rating > 0:
        params["vote_average.gte"] = min_rating
        params["vote_count.gte"] = 30

    resp = requests.get(discover_url, headers=headers, params=params)
    movies = resp.json().get('results', []) if resp.status_code == 200 else []

    # --- AKILLI YENİDEN DENEME (FALLBACK MEKANİZMASI) ---
    # 1. Keyword yüzünden film bulunamadıysa keyword'ü çıkarıp tekrar ara
    if not movies and kid and "with_keywords" in params:
        del params["with_keywords"]
        resp = requests.get(discover_url, headers=headers, params=params)
        movies = resp.json().get('results', []) if resp.status_code == 200 else []

    # 2. Hala yoksa ve türler çok sıkıysa (AND), OR (|) mantığına çevirip ara
    if not movies and "with_genres" in params and "," in params["with_genres"]:
        params["with_genres"] = params["with_genres"].replace(",", "|")
        resp = requests.get(discover_url, headers=headers, params=params)
        movies = resp.json().get('results', []) if resp.status_code == 200 else []

    if not movies:
        return "Aradığınız kriterlere uygun film bulunamadı."

    return json.dumps(cards_from_ids([m["id"] for m in movies[:MAX_MOVIES]]), ensure_ascii=False)


@mcp.tool()
@ttl_cache(ttl=1800)
def get_similar_movies(movie_title: str) -> str:
    """
    Kullanıcının belirttiği bir film adına benzer veya önerilen diğer filmleri getirir.
    """
    search_url = f"{BASE_URL}/search/movie"
    params = {"query": movie_title, "language": "tr-TR"}
    resp = requests.get(search_url, headers=headers, params=params)

    if resp.status_code != 200 or not resp.json().get('results'):
        params["language"] = "en-US"
        resp = requests.get(search_url, headers=headers, params=params)

    if resp.status_code != 200 or not resp.json().get('results'):
        return f"'{movie_title}' adında bir film bulunamadı."

    movie = resp.json()['results'][0]
    movie_id = movie['id']
    orig_title = movie['title']

    recs_url = f"{BASE_URL}/movie/{movie_id}/recommendations"
    params_recs = {"language": "tr-TR", "page": 1}
    recs_resp = requests.get(recs_url, headers=headers, params=params_recs)

    recs = []
    if recs_resp.status_code == 200:
        recs = recs_resp.json().get('results', [])[:MAX_MOVIES]

    if not recs:
        similar_url = f"{BASE_URL}/movie/{movie_id}/similar"
        sim_resp = requests.get(similar_url, headers=headers, params=params_recs)
        if sim_resp.status_code == 200:
            recs = sim_resp.json().get('results', [])[:MAX_MOVIES]

    if not recs:
        return f"'{orig_title}' filmi için benzer film önerileri bulunamadı."

    return json.dumps(cards_from_ids([m["id"] for m in recs]), ensure_ascii=False)


@mcp.tool()
def search_movies_in_graph(category: str, search_value: str) -> str:
    """
    Yerel Neo4j graf veritabanında esnek arama yapar.
    Category değerleri: 'Director', 'Actor', 'Genre', 'Keyword', 'Similar'.
    search_value aranacak terimdir (örn: 'Christopher Nolan', 'Action').
    """
    queries = {
        "Director": """
            MATCH (p:Person)-[:DIRECTED]->(m:Movie)
            WHERE toLower(p.name) CONTAINS toLower($val)
            RETURN m.title AS title, m.overview AS overview, m.poster_path AS poster_path,
                   [(m)<-[:DIRECTED]-(d:Person) | d.name] AS director,
                   [(m)<-[r:ACTED_IN]-(c:Person) | c.name] AS cast
            ORDER BY m.vote_average DESC LIMIT 5
        """,
        "Actor": """
            MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
            WHERE toLower(p.name) CONTAINS toLower($val)
            RETURN m.title AS title, m.overview AS overview, m.poster_path AS poster_path,
                   [(m)<-[:DIRECTED]-(d:Person) | d.name] AS director,
                   [(m)<-[r2:ACTED_IN]-(c:Person) | c.name] AS cast
            ORDER BY m.popularity DESC LIMIT 5
        """,
        "Genre": """
            MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
            WHERE toLower(g.name) CONTAINS toLower($val)
            RETURN m.title AS title, m.overview AS overview, m.poster_path AS poster_path,
                   [(m)<-[:DIRECTED]-(d:Person) | d.name] AS director,
                   [(m)<-[r:ACTED_IN]-(c:Person) | c.name] AS cast
            ORDER BY m.popularity DESC LIMIT 5
        """,
        "Keyword": """
            MATCH (m:Movie)-[:HAS_KEYWORD]->(k:Keyword)
            WHERE toLower(k.name) CONTAINS toLower($val)
            RETURN m.title AS title, m.overview AS overview, m.poster_path AS poster_path,
                   [(m)<-[:DIRECTED]-(d:Person) | d.name] AS director,
                   [(m)<-[r:ACTED_IN]-(c:Person) | c.name] AS cast
            ORDER BY m.vote_average DESC LIMIT 5
        """,
        "Similar": """
            MATCH (m:Movie)
            WHERE toLower(m.title) CONTAINS toLower($val)
            MATCH (m)-[:SIMILAR_TO]->(similar:Movie)
            RETURN similar.title AS title, similar.overview AS overview, similar.poster_path AS poster_path,
                   [(similar)<-[:DIRECTED]-(d:Person) | d.name] AS director,
                   [(similar)<-[r:ACTED_IN]-(c:Person) | c.name] AS cast
            LIMIT 5
        """
    }

    cypher = queries.get(category)
    if not cypher:
        return "Geçersiz kategori."

    try:
        with neo4j_driver.session() as session:
            result = session.run(cypher, val=search_value)
            records = []
            for record in result:
                rec_dict = dict(record)
                if isinstance(rec_dict.get("director"), list):
                    rec_dict["director"] = ", ".join(rec_dict["director"])
                if isinstance(rec_dict.get("cast"), list):
                    rec_dict["cast"] = ", ".join(rec_dict["cast"][:4])  # İlk 4 oyuncu
                records.append(rec_dict)
    except Exception as e:
        return f"Graf arama hatası: {str(e)}"

    if not records:
        return "[]"

    # Graf kayıtlarını TMDB'den zenginleştir (gerçek fragman, yıl, puan, platform)
    def build(rec):
        mid = get_movie_id(rec["title"])
        card = fetch_movie_card(mid, fallback=rec) if mid else None
        return card or {
            "movie_id": "", "Film": rec["title"], "Director": rec.get("director", ""),
            "Cast": rec.get("cast", ""), "Yıl": "", "TMDB Puanı": "", "Türler": "",
            "Özet": (rec.get("overview") or "")[:200],
            "Poster": f"https://image.tmdb.org/t/p/w500{rec['poster_path']}" if rec.get("poster_path") else "",
            "Fragman": "", "Şu Anki Platform(lar)": "", "Neden Önerildi": "",
        }

    with ThreadPoolExecutor(max_workers=5) as ex:
        cards = list(ex.map(build, records))
    return json.dumps(cards, ensure_ascii=False)

@mcp.tool()
def search_movies_semantically(semantic_query: str) -> str:
    """
    Film özetleri üzerinde anlamsal (semantic) vektör araması yapar.
    Klasik kelime eşleşmesi yetersiz olduğunda (duygular, temalar, soyut konular, örn: 'yalnız bir astronot') kullanılır.
    """
    try:
        # 1. Kullanıcının arama metnini vektöre dönüştür
        query_vector = embeddings_model.embed_query(semantic_query)
        
        # 2. Neo4j Vektör Arama Cypher sorgusu
        cypher = """
        CALL db.index.vector.queryNodes('movie_overview_index', $top_k, $query_vector)
        YIELD node AS movie, score
        RETURN movie.id AS id, movie.title AS title, movie.overview AS overview, movie.poster_path AS poster_path, score
        """
        
        session_kwargs = {"database": NEO4J_DATABASE} if NEO4J_DATABASE else {}
        with neo4j_driver.session(**session_kwargs) as session:
            result = session.run(cypher, query_vector=query_vector, top_k=MAX_MOVIES)
            records = [dict(record) for record in result]
            
    except Exception as e:
        return f"Semantik arama hatası: {str(e)}"

    if not records:
        return "Anlamsal olarak benzer bir film bulunamadı."

    # Kayıtları TMDB kart formatına zenginleştirerek dön
    movie_ids = [r["id"] for r in records if r.get("id")]
    cards = cards_from_ids(movie_ids)
    
    if not cards:
        # Fallback: Eğer TMDB ID eşleşmezse veritabanındaki ham verilerle kart oluştur
        fallback_cards = []
        for r in records:
            fallback_cards.append({
                "movie_id": "",
                "Film": r.get("title", ""),
                "Director": "",
                "Cast": "",
                "Yıl": "",
                "TMDB Puanı": "",
                "Türler": "",
                "Özet": (r.get("overview") or "")[:200],
                "Poster": f"https://image.tmdb.org/t/p/w500{r['poster_path']}" if r.get("poster_path") else "",
                "Fragman": "",
                "Şu Anki Platform(lar)": "",
                "Neden Önerildi": f"Anlamsal Benzerlik Skoru: {r.get('score', 0):.2f}"
            })
        return json.dumps(fallback_cards, ensure_ascii=False)

    return json.dumps(cards, ensure_ascii=False)


@mcp.tool()
def get_movie_card(title: str) -> str:
    """
    Adı belli tek bir filmin detay kartını getirir (yönetmen, oyuncular, puan, fragman, platform).
    Kullanıcı belirli bir filmi soruyorsa veya film adı biliniyorsa kullan.
    """
    mid = get_movie_id(title)
    card = fetch_movie_card(mid) if mid else None
    if not card:
        return f"'{title}' adında bir film bulunamadı."
    return json.dumps([card], ensure_ascii=False)


if __name__ == "__main__":
    mcp.run()