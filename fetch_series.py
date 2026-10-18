"""
TMDB'den DİZİ verisi çeker -> series_data.json   (fetch_movies.py'nin dizi karşılığı)

Film hattından farkları:
  * Uç noktalar: /tv/popular, /tv/top_rated, /discover/tv (first_air_date ile 5'er yıllık dilimler)
  * Alan adları: name / original_name / first_air_date / number_of_seasons / number_of_episodes
  * Dizide 'yönetmen' yoktur -> created_by (yaratıcılar) kaydedilir.
  * Kadro için aggregate_credits (tüm sezonların toplam kadrosu) kullanılır.
  * Dizi oy sayıları filmlerden çok düşüktür -> MIN_VOTES varsayılanı 15.
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
MIN_VOTES = int(os.getenv("SERIES_MIN_VOTES", "15"))

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
    extra = extra or {}

    def one(page):
        try:
            data = tmdb_get(path, page=page, language="en-US", include_adult="false", **extra)
            return [s["id"] for s in data.get("results", [])]
        except Exception:
            return []

    ids = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for res in ex.map(one, range(1, min(pages, 500) + 1)):
            ids.extend(res)
    return ids


def collect_series_ids(pop_pages=300, top_pages=300, bucket_pages=10, bucket_years=5):
    ids = set()
    print("tv/popular...");   ids.update(fetch_ids("/tv/popular", pop_pages))
    print("tv/top_rated..."); ids.update(fetch_ids("/tv/top_rated", top_pages))
    for start in range(1950, 2030, bucket_years):
        print(f"discover tv {start}-{start + bucket_years - 1}...")
        ids.update(fetch_ids("/discover/tv", bucket_pages, extra={
            "sort_by": "vote_count.desc",
            "first_air_date.gte": f"{start}-01-01",
            "first_air_date.lte": f"{start + bucket_years - 1}-12-31",
            "vote_count.gte": MIN_VOTES,
        }))
    return sorted(ids)


def get_series_details(series_id):
    data = tmdb_get(f"/tv/{series_id}", language="en-US",
                    append_to_response="aggregate_credits,keywords,recommendations")

    if not data.get("overview"):
        return None
    if (data.get("vote_count") or 0) < MIN_VOTES:
        return None

    cast = data.get("aggregate_credits", {}).get("cast", [])[:7]
    runtimes = data.get("episode_run_time") or []
    return {
        "id": data.get("id"),
        "name": data.get("name"),
        "original_name": data.get("original_name"),
        "overview": data.get("overview"),
        "tagline": data.get("tagline"),
        "first_air_date": data.get("first_air_date"),
        "last_air_date": data.get("last_air_date"),
        "status": data.get("status"),                       # Returning Series / Ended / Canceled
        "number_of_seasons": data.get("number_of_seasons"),
        "number_of_episodes": data.get("number_of_episodes"),
        "episode_runtime": runtimes[0] if runtimes else None,
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "popularity": data.get("popularity"),
        "poster_path": data.get("poster_path"),
        "original_language": data.get("original_language"),
        "networks": [n["name"] for n in data.get("networks", [])],
        "genres": [{"id": g["id"], "name": g["name"]} for g in data.get("genres", [])],
        "creators": [{"id": c["id"], "name": c["name"]} for c in data.get("created_by", [])],
        "cast": [{"id": a["id"], "name": a["name"],
                  "character": ((a.get("roles") or [{}])[0]).get("character", "")} for a in cast],
        "keywords": [{"id": k["id"], "name": k["name"]}
                     for k in data.get("keywords", {}).get("results", [])],   # TV'de anahtar 'results'
        "similar_series": [{"id": s["id"], "name": s.get("name")}
                           for s in data.get("recommendations", {}).get("results", [])[:10]],
    }


def save(path, items):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(list(items.values()), f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def fetch_and_save_series(output_file="series_data.json", max_workers=15, **id_kwargs):
    ids = collect_series_ids(**id_kwargs)
    print(f"{len(ids)} benzersiz dizi ID'si toplandı.")

    existing = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing = {s["id"]: s for s in json.load(f) if "id" in s}
            print(f"Mevcut dosyada {len(existing)} dizi var; yalnızca yenileri çekilecek.")
        except Exception:
            existing = {}

    todo = [i for i in ids if i not in existing]
    all_series, saved = dict(existing), 0

    def safe(sid):
        try:
            return get_series_details(sid)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(safe, sid) for sid in todo]
        for fut in tqdm(as_completed(futs), total=len(futs), desc="Diziler"):
            res = fut.result()
            if res:
                all_series[res["id"]] = res
                saved += 1
                if saved % 200 == 0:
                    save(output_file, all_series)

    save(output_file, all_series)
    print(f"Bitti: {len(all_series)} dizi -> {output_file}")


if __name__ == "__main__":
    fetch_and_save_series()
