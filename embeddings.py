"""
Film vektörlerini üretip Neo4j'ye yazar.

DEĞİŞİKLİKLER:
  * nomic-embed-text için zorunlu 'search_document:' / 'search_query:' önekleri eklendi
    (önek yoksa model kalitesi belirgin düşer). Ayar: movie_common.py / .env
  * Embedding metni sadece özet değil: başlık + yıl + tür + tagline + özet + anahtar kelimeler.
  * Modeli/boyutu değiştirirsen (örn. EMBED_MODEL=bge-m3 EMBED_DIM=1024) eski index silinip yeniden kurulur.
  * Hata alan batch artık sessizce atlanmaz; tek tek yeniden denenir.
"""
import json
import os
from itertools import islice

from dotenv import load_dotenv
from langchain_ollama import OllamaEmbeddings
from neo4j import GraphDatabase

from movie_common import EMBED_DIM, EMBED_MODEL, VECTOR_INDEX, movie_doc

load_dotenv()
BATCH_SIZE = 50
embeddings_model = OllamaEmbeddings(model=EMBED_MODEL)


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch


def embed_with_retry(docs):
    """Önce toplu dener; olmazsa tek tek. Başarısız olanlar için None döner."""
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


def update_movies_with_embeddings(driver, json_path="movies_data.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        movies = json.load(f)
    valid = [m for m in movies if m.get("overview")]
    print(f"{len(valid)} film için embedding üretiliyor (model={EMBED_MODEL}, dim={EMBED_DIM})...")

    query = """
    UNWIND $rows AS row
    MATCH (m:Movie {id: row.movie_id})
    SET m.embedding = row.vector
    """
    done = failed = 0
    with driver.session() as session:
        for batch in chunked(valid, BATCH_SIZE):
            vectors = embed_with_retry([movie_doc(m) for m in batch])
            rows = [{"movie_id": m["id"], "vector": v} for m, v in zip(batch, vectors) if v]
            failed += len(batch) - len(rows)
            if rows:
                session.run(query, rows=rows)
            done += len(rows)
            print(f"İlerleme: {done}/{len(valid)} (başarısız: {failed})")


def create_vector_index(driver):
    with driver.session() as session:
        # Boyut/model değiştiyse eski index geçersizdir.
        session.run(f"DROP INDEX {VECTOR_INDEX} IF EXISTS")
        session.run(f"""
        CREATE VECTOR INDEX {VECTOR_INDEX} IF NOT EXISTS
        FOR (m:Movie) ON (m.embedding)
        OPTIONS {{indexConfig: {{
          `vector.dimensions`: {EMBED_DIM},
          `vector.similarity_function`: 'cosine'
        }}}}
        """)
    print(f"{VECTOR_INDEX} oluşturuldu (dim={EMBED_DIM}).")


if __name__ == "__main__":
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    drv = GraphDatabase.driver(uri, auth=(user, password))
    try:
        update_movies_with_embeddings(drv, "movies_data.json")
        create_vector_index(drv)
    finally:
        drv.close()