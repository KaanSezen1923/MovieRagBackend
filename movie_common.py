"""
Ortak yardımcılar: db.py, embeddings.py ve server.py aynı normalizasyonu,
aynı embedding ayarlarını ve aynı kalite skorunu buradan kullanır.
(Üçünün ayrı ayrı tanımlaması, 'ınception' hatasının kaynağıydı.)
"""
import os
import unicodedata
from dotenv import load_dotenv

load_dotenv()


# ---------- Metin normalizasyonu (SADECE arama alanları için) ----------
def norm(text):
    """
    Arama için normalize eder: küçük harf + aksan temizliği + Türkçe ı/İ -> i.
    Görüntülenen alanlara (title, overview, name) ASLA uygulanmaz; onlar ham kalır.
    norm("Inception") == norm("ınception") == norm("İNCEPTİON") == "inception"
    """
    if not isinstance(text, str):
        return text
    t = text.replace("İ", "i").replace("ı", "i")
    t = unicodedata.normalize("NFKD", t.casefold())
    return "".join(c for c in t if not unicodedata.combining(c)).strip()


# ---------- Embedding ayarları ----------
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))
_nomic = "nomic" in EMBED_MODEL
# nomic-embed-text görev öneki ister; bge-m3 vb. modellerde env ile boş bırak.
DOC_PREFIX = os.getenv("EMBED_DOC_PREFIX", "search_document: " if _nomic else "")
QUERY_PREFIX = os.getenv("EMBED_QUERY_PREFIX", "search_query: " if _nomic else "")
VECTOR_INDEX = "movie_overview_index"

EMBED_PROVIDER = os.getenv("EMBED_PROVIDER", "ollama")


def get_embedder():
    provider = (os.getenv("EMBED_PROVIDER") or EMBED_PROVIDER).strip().lower()
    model = os.getenv("EMBED_MODEL") or EMBED_MODEL
    dim = int(os.getenv("EMBED_DIM") or EMBED_DIM)

    if provider == "openai" or (os.getenv("OPENAI_API_KEY") and model.startswith("text-embedding")):
        from langchain_openai import OpenAIEmbeddings   # OPENAI_API_KEY env'den okunur
        return OpenAIEmbeddings(model=model, dimensions=dim)

    from langchain_ollama import OllamaEmbeddings
    base_url = (
        os.getenv("OLLAMA_EMBED_HOST")
        or os.getenv("OLLAMA_HOST")
        or "http://localhost:11434"
    ).rstrip("/")
    client_kwargs = {
        "trust_env": False,
        "timeout": float(os.getenv("EMBED_TIMEOUT", "5.0")),
    }
    ollama_key = os.getenv("OLLAMA_API_KEY")
    if ollama_key:
        client_kwargs["headers"] = {"Authorization": f"Bearer {ollama_key}"}
    return OllamaEmbeddings(model=model, base_url=base_url, client_kwargs=client_kwargs)


def movie_doc(m: dict) -> str:
    """Embedding'e giren metin: sadece özet değil; başlık, tür, tagline, anahtar kelimeler de."""
    year = (m.get("release_date") or "")[:4]
    genres = ", ".join(g["name"] for g in m.get("genres") or [])
    kws = ", ".join(k["name"] for k in (m.get("keywords") or [])[:15])
    parts = [f"{m.get('title', '')} ({year})"]
    if genres:
        parts.append(f"Genres: {genres}")
    if m.get("tagline"):
        parts.append(m["tagline"])
    parts.append(m.get("overview") or "")
    if kws:
        parts.append(f"Keywords: {kws}")
    return DOC_PREFIX + ". ".join(p for p in parts if p)


# ---------- Kalite skoru ----------
def bayes_score(avg, votes, m=500, c=6.5):
    """
    Bayesian ağırlıklı puan (IMDb formülü). 3 oylu '10.0' film, 50 bin oylu '8.4' filmi geçemez.
    m: güvenilir sayılmak için gereken oy sayısı, c: genel ortalama.
    """
    try:
        avg, votes = float(avg), float(votes)
    except (TypeError, ValueError):
        return 0.0
    return (votes * avg + m * c) / (votes + m)