"""
TMDB'den film verisi çeker -> movies_data.json

DEĞİŞİKLİKLER (eski sürüme göre):
  * Sadece /movie/popular (en çok POPÜLER 10k) yerine 3 kaynak birleştirilir:
    popular + top_rated + 5'er yıllık dilimlerde en çok oylananlar (vote_count.desc).
    Eski yöntem yeni çıkanlara (2026) ve güncel popülerliğe kayıyor, klasikleri (Memento, Se7en...)
    kaçırıyordu.
  * Kalite kapısı: yetişkin içerik, özeti olmayan ve çok az oy almış (MIN_VOTES altı) kayıtlar
    veritabanına hiç girmez (puanı 0.0 / 10.0 olan gürültü filmlerin kaynağı).
  * 'similar' yerine 'recommendations' (kullanıcı davranışına dayalı, daha kaliteli).
  * tagline, original_language, runtime gibi alanlar da kaydedilir.
  * 429/5xx için otomatik yeniden deneme (backoff); ara kayıt hatası düzeltildi.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from tqdm import tqdm
from urllib3.util.retry import Retry

load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
AUTH_KEY = os.getenv("AUTH_KEY")
if not TMDB_API_KEY and not AUTH_KEY:
    raise ValueError("TMDB_API_KEY veya AUTH_KEY .env dosyasında bulunamadı!")

BASE_URL = "https://api.themoviedb.org/3"
MIN_VOTES = int(os.getenv("MIN_VOTES", "25"))

http = requests.Session()
http.headers.update({"accept": "application/json"})
if AUTH_KEY:
    http.headers["Authorization"] = f"Bearer {AUTH_KEY}"
http.mount("https://", HTTPAdapter(
    max_retries=Retry(total=5, backoff_factor=0.6, status_forcelist=[429, 500, 502, 503, 504],
                      allowed_methods=["GET"]),
    pool_maxsize=32))


def tmdb_get(path, **params):
    if not AUTH_KEY:
        params["api_key"] = TMDB_API_KEY
    r = http.get(f"{BASE_URL}{path}", params=params, timeout=15)
    r.raise_for_status()
    return r.json()


def fetch_ids(path, pages, extra=None, max_workers=10):
    """Bir liste endpoint'inin ilk `pages` sayfasındaki film ID'lerini paralel toplar."""
    extra = extra or {}

    def one(page):
        try:
            data = tmdb_get(path, page=page, language="en-US", include_adult="false", **extra)
            return [m["id"] for m in data.get("results", [])]
        except Exception:
            return []

    ids = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(one, range(1, min(pages, 500) + 1)):
            ids.extend(res)
    return ids


def collect_movie_ids(pop_pages=500, top_pages=500, bucket_pages=15, bucket_years=5):
    ids = set()
    print("popular...");   ids.update(fetch_ids("/movie/popular", pop_pages))
    print("top_rated..."); ids.update(fetch_ids("/movie/top_rated", top_pages))
    for start in range(1930, 2030, bucket_years):
        print(f"discover {start}-{start + bucket_years - 1}...")
        ids.update(fetch_ids("/discover/movie", bucket_pages, extra={
            "sort_by": "vote_count.desc",
            "primary_release_date.gte": f"{start}-01-01",
            "primary_release_date.lte": f"{start + bucket_years - 1}-12-31",
            "vote_count.gte": MIN_VOTES,
        }))
    return sorted(ids)


def get_movie_details(movie_id):
    data = tmdb_get(f"/movie/{movie_id}", language="en-US",
                    append_to_response="credits,keywords,recommendations")

    # Kalite kapısı
    if data.get("adult") or not data.get("overview"):
        return None
    if (data.get("vote_count") or 0) < MIN_VOTES:
        return None

    credits = data.get("credits", {})
    return {
        "id": data.get("id"),
        "title": data.get("title"),
        "original_title": data.get("original_title"),
        "overview": data.get("overview"),
        "tagline": data.get("tagline"),
        "release_date": data.get("release_date"),
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "popularity": data.get("popularity"),
        "poster_path": data.get("poster_path"),
        "original_language": data.get("original_language"),
        "runtime": data.get("runtime"),
        "genres": [{"id": g["id"], "name": g["name"]} for g in data.get("genres", [])],
        "directors": [{"id": c["id"], "name": c["name"]}
                      for c in credits.get("crew", []) if c.get("job") == "Director"],
        "cast": [{"id": a["id"], "name": a["name"], "character": a.get("character", "")}
                 for a in credits.get("cast", [])[:7]],
        "keywords": [{"id": k["id"], "name": k["name"]}
                     for k in data.get("keywords", {}).get("keywords", [])],
        # db.py bu anahtarı okuyor; içerik artık 'recommendations'
        "similar_movies": [{"id": s["id"], "title": s.get("title")}
                           for s in data.get("recommendations", {}).get("results", [])[:10]],
    }


def save(path, movies_dict):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(list(movies_dict.values()), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)   # yarım yazılmış dosya bırakma


def fetch_and_save_movies(output_file="movies_data.json", max_workers=15, **id_kwargs):
    movie_ids = collect_movie_ids(**id_kwargs)
    print(f"{len(movie_ids)} benzersiz ID toplandı.")

    existing = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing = {m["id"]: m for m in json.load(f) if "id" in m}
            print(f"Mevcut dosyada {len(existing)} film var; yalnızca yenileri çekilecek.")
        except Exception:
            existing = {}

    todo = [i for i in movie_ids if i not in existing]
    print(f"Çekilecek: {len(todo)}")
    all_movies, saved = dict(existing), 0

    def safe(mid):
        try:
            return get_movie_details(mid)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(safe, mid) for mid in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Filmler"):
            res = fut.result()
            if res:
                all_movies[res["id"]] = res
                saved += 1
                if saved % 200 == 0:          # ara kayıt: sadece başarılı kayıtta, tekrar tekrar yazmaz
                    save(output_file, all_movies)

    save(output_file, all_movies)
    print(f"Bitti: {len(all_movies)} film -> {output_file}")


if __name__ == "__main__":
    fetch_and_save_movies()