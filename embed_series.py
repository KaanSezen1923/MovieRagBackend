"""
Dizi vektörlerini üretip :Series düğümlerine yazar ve AYRI bir vektör index'i kurar.

DİKKAT: embeddings.py her çalıştığında film index'ini DROP edip yeniden kurar.
Bu script yalnızca SERIES_VECTOR_INDEX ile ilgilenir; film index'ine dokunmaz.
Film ile AYNI embedding modeli (EMBED_MODEL / EMBED_DIM) kullanılır, böylece tek sorgu vektörü
iki index'te de aranabilir.
"""
import json
import os
from itertools import islice

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from neo4j import GraphDatabase

from movie_common import EMBED_DIM, EMBED_MODEL

load_dotenv()
SERIES_VECTOR_INDEX = os.getenv("SERIES_VECTOR_INDEX", "series_overview_index")
EMBED_HOST = os.getenv("OLLAMA_EMBED_HOST", "http://localhost:11434")
BATCH_SIZE = 50
embeddings_model = OllamaEmbeddings(model=EMBED_MODEL, base_url=EMBED_HOST)

from movie_common import movie_doc


def series_doc(s: dict) -> str:
    """movie_doc'u YENİDEN KULLANIR (aynı önek/biçim -> aynı vektör uzayı). Dizi alanlarını film şekline eşler."""
    return movie_doc({**s, "title": s.get("name"), "release_date": s.get("first_air_date")})


def chunked(it, n):
    it = iter(it)
    while b := list(islice(it, n)):
        yield b


def embed_with_retry(docs):
    try:
        return embeddings_model.embed_documents(docs)
    except Exception as e:
        print(f"  batch hatası ({e}); tek tek deneniyor...")
        out = []
        for d in docs:
            try:
                out.append(embeddings_model.embed_documents([d])[0])
            except Exception:
                out.append(None)
        return out


def run(driver, json_path="series_data.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        items = [s for s in json.load(f) if s.get("overview")]
    print(f"{len(items)} dizi için embedding (model={EMBED_MODEL}, dim={EMBED_DIM}, host={EMBED_HOST})")
    q = "UNWIND $rows AS r MATCH (s:Series {id: r.id}) SET s.embedding = r.vector"
    done = failed = 0
    with driver.session() as session:
        for batch in chunked(items, BATCH_SIZE):
            vecs = embed_with_retry([series_doc(s) for s in batch])
            rows = [{"id": s["id"], "vector": v} for s, v in zip(batch, vecs) if v]
            failed += len(batch) - len(rows)
            if rows:
                session.run(q, rows=rows)
            done += len(rows)
            print(f"İlerleme: {done}/{len(items)} (başarısız: {failed})")
        session.run(f"""
        CREATE VECTOR INDEX {SERIES_VECTOR_INDEX} IF NOT EXISTS
        FOR (s:Series) ON (s.embedding)
        OPTIONS {{indexConfig: {{`vector.dimensions`: {EMBED_DIM}, `vector.similarity_function`: 'cosine'}}}}
        """)
    print(f"{SERIES_VECTOR_INDEX} hazır (dim={EMBED_DIM}).")


if __name__ == "__main__":
    drv = GraphDatabase.driver(os.getenv("NEO4J_URI"),
                               auth=(os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j"),
                                     os.getenv("NEO4J_PASSWORD")))
    try:
        run(drv)
    finally:
        drv.close()
