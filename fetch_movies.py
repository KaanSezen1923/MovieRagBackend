import os
import json
import time
import requests
from dotenv import load_dotenv
from tqdm import tqdm

# .env dosyasındaki anahtarları yükle
load_dotenv()

TMDB_API_KEY = os.getenv("TMDB_API_KEY")
AUTH_KEY = os.getenv("AUTH_KEY")

if not TMDB_API_KEY and not AUTH_KEY:
    raise ValueError("TMDB_API_KEY veya AUTH_KEY .env dosyasında bulunamadı!")

BASE_URL = "https://api.themoviedb.org/3"
HEADERS = {
    "accept": "application/json",
}
if AUTH_KEY:
    HEADERS["Authorization"] = f"Bearer {AUTH_KEY}"


def get_popular_movie_ids(total_pages=500, max_workers=10):
    """
    Popüler filmlerin ID listesini paralel olarak çeker.
    Her sayfada 20 film bulunur (500 sayfa = 10.000 film).
    """
    if total_pages > 500:
        total_pages = 500

    movie_ids = []
    print(f"[{total_pages} sayfa] Popüler film listesi paralel olarak çekiliyor...")

    def fetch_page(page):
        params = {
            "page": page,
            "language": "en-US"
        }
        if not AUTH_KEY:
            params["api_key"] = TMDB_API_KEY

        url = f"{BASE_URL}/movie/popular"
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=10)
            if resp.status_code == 200:
                data = resp.json()
                return [movie["id"] for movie in data.get("results", [])]
        except Exception:
            pass
        return []

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(fetch_page, p) for p in range(1, total_pages + 1)]
        for f in tqdm(as_completed(futures), total=total_pages, desc="Sayfalar Taranıyor"):
            movie_ids.extend(f.result())

    return list(set(movie_ids))


def get_movie_details(movie_id):
    """
    Tek bir filmin detaylarını, oyuncularını (credits), anahtar kelimelerini (keywords) 
    ve benzer filmlerini (similar) tek bir API isteğinde çeker.
    """
    params = {
        "append_to_response": "credits,keywords,similar",
        "language": "en-US"
    }
    if not AUTH_KEY:
        params["api_key"] = TMDB_API_KEY

    url = f"{BASE_URL}/movie/{movie_id}"
    resp = requests.get(url, headers=HEADERS, params=params, timeout=10)

    if resp.status_code != 200:
        return None

    data = resp.json()

    # Yönetmenleri bul
    directors = [
        {"id": c["id"], "name": c["name"]}
        for c in data.get("credits", {}).get("crew", [])
        if c.get("job") == "Director"
    ]

    # Başrol oyuncuları (ilk 5-8 oyuncu tavsiye sistemi için yeterli ve kalitelidir)
    cast = [
        {"id": a["id"], "name": a["name"], "character": a.get("character", "")}
        for a in data.get("credits", {}).get("cast", [])[:7]
    ]

    # Türler (Genres)
    genres = [
        {"id": g["id"], "name": g["name"]}
        for g in data.get("genres", [])
    ]

    # Anahtar kelimeler (Keywords)
    keywords = [
        {"id": k["id"], "name": k["name"]}
        for k in data.get("keywords", {}).get("keywords", [])
    ]

    # Benzer filmler (Similar Movies - ilk 5'i)
    similar_movies = [
        {"id": s["id"], "title": s.get("title")}
        for s in data.get("similar", {}).get("results", [])[:5]
    ]

    # Temizlenmiş ve Graph DB için optimize edilmiş JSON formatı
    cleaned_movie = {
        "id": data.get("id"),
        "title": data.get("title"),
        "original_title": data.get("original_title"),
        "overview": data.get("overview"),
        "release_date": data.get("release_date"),
        "vote_average": data.get("vote_average"),
        "vote_count": data.get("vote_count"),
        "popularity": data.get("popularity"),
        "poster_path": data.get("poster_path"),
        "genres": genres,
        "directors": directors,
        "cast": cast,
        "keywords": keywords,
        "similar_movies": similar_movies
    }

    return cleaned_movie


def fetch_and_save_movies(total_pages=500, output_file="movies_data.json", max_workers=15):
    """
    TMDB API'nin izin verdiği maksimum sayfa sayısı 500'dür (en fazla 10.000 film).
    ThreadPoolExecutor ile istekleri paralel yaparak süreci hızlandırır ve
    her 200 filmde bir ara kayıt alarak veriyi korur.
    """
    # TMDB API üst limiti 500 sayfadır
    if total_pages > 500:
        print(f"Uyarı: TMDB API en fazla 500 sayfaya izin verir. Sayfa sayısı 500 olarak ayarlandı.")
        total_pages = 500

    movie_ids = get_popular_movie_ids(total_pages=total_pages)
    print(f"Toplam {len(movie_ids)} benzersiz film ID'si toplandı. Detaylar paralel olarak çekiliyor...")

    # Mevcut veriyi kontrol et (varsa kaldığı yerden devam edebilsin)
    existing_movies = {}
    if os.path.exists(output_file):
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                saved_data = json.load(f)
                for item in saved_data:
                    if "id" in item:
                        existing_movies[item["id"]] = item
            print(f"Mevcut '{output_file}' dosyasında {len(existing_movies)} kayıtlı film bulundu. Yeni olanlar indirilecek.")
        except Exception:
            existing_movies = {}

    from concurrent.futures import ThreadPoolExecutor, as_completed

    # Henüz çekilmemiş olan ID'leri filtrele
    ids_to_fetch = [mid for mid in movie_ids if mid not in existing_movies]
    print(f"Çekilecek yeni film sayısı: {len(ids_to_fetch)}")

    all_movies_dict = dict(existing_movies)
    saved_counter = 0

    def safe_get_details(mid):
        try:
            return get_movie_details(mid)
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {executor.submit(safe_get_details, mid): mid for mid in ids_to_fetch}
        for future in tqdm(as_completed(future_to_id), total=len(ids_to_fetch), desc="Filmler Çekiliyor"):
            result = future.result()
            if result:
                all_movies_dict[result["id"]] = result
                saved_counter += 1

            # Her 200 filmde bir ara kayıt (veri kaybını önlemek için)
            if saved_counter > 0 and saved_counter % 200 == 0:
                with open(output_file, "w", encoding="utf-8") as f:
                    json.dump(list(all_movies_dict.values()), f, ensure_ascii=False, indent=2)

    # Nihai kayıt
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(list(all_movies_dict.values()), f, ensure_ascii=False, indent=2)

    print(f"\nİşlem tamamlandı! Toplam {len(all_movies_dict)} film '{output_file}' dosyasına kaydedildi.")


if __name__ == "__main__":
    # TMDB API'nin izin verdiği mutlak tavan: 500 sayfa (10.000 film)
    fetch_and_save_movies(total_pages=500, output_file="movies_data.json", max_workers=15)

