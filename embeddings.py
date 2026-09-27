import os
import json
from neo4j import GraphDatabase
from langchain_ollama import OllamaEmbeddings
from dotenv import load_dotenv
from itertools import islice

load_dotenv()

# Performans ve bellek dengesi için ideal bir batch boyutu (örn: 50 veya 100)
BATCH_SIZE = 50 

embeddings_model = OllamaEmbeddings(model="nomic-embed-text") 

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch

def update_movies_with_embeddings(uri, user, password, json_path="movies_data.json"):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    with open(json_path, "r", encoding="utf-8") as f:
        movies = json.load(f)

    # Özet alanı boş olmayan filmleri filtrele
    valid_movies = [m for m in movies if m.get("overview")]
    total_movies = len(valid_movies)
    print(f"Toplam {total_movies} film için batch tabanlı embedding aktarımı başlatılıyor...")

    with driver.session() as session:
        counter = 0
        for batch in chunked(valid_movies, BATCH_SIZE):
            movie_ids = [m["id"] for m in batch]
            overviews = [m["overview"] for m in batch]
            
            try:
                # 1. Toplu (Batch) Embedding Üretimi (Çok daha hızlıdır)
                vectors = embeddings_model.embed_documents(overviews)
                
                # 2. Neo4j'ye gönderilecek satırları hazırla
                rows = [
                    {"movie_id": mid, "vector": vec}
                    for mid, vec in zip(movie_ids, vectors)
                ]
                
                # 3. UNWIND ile Toplu Cypher Güncellemesi
                query = """
                UNWIND $rows AS row
                MATCH (m:Movie {id: row.movie_id})
                SET m.embedding = row.vector
                """
                session.run(query, rows=rows)
                
                counter += len(batch)
                print(f"İlerleme: {counter}/{total_movies} film güncellendi.")
                
            except Exception as e:
                print(f"Batch işlenirken hata oluştu: {e}")
                
    driver.close()
    print("Tüm filmlerin embedding güncellemeleri tamamlandı!")

def create_vector_index(uri, user, password):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    query = """
    CREATE VECTOR INDEX movie_overview_index IF NOT EXISTS
    FOR (m:Movie)
    ON (m.embedding)
    OPTIONS {indexConfig: {
     `vector.dimensions`: 768,
     `vector.similarity_function`: 'cosine'
    }}
    """
    with driver.session() as session:
        session.run(query)
        print("movie_overview_index vektör indeksi kontrol edildi / oluşturuldu.")
    driver.close()

if __name__ == "__main__":
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")

    update_movies_with_embeddings(
        uri=uri,
        user=user,
        password=password,
        json_path="movies_data.json",
    )
    create_vector_index(uri=uri, user=user, password=password)