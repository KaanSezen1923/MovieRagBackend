from langchain_ollama import OllamaEmbeddings
from neo4j import GraphDatabase
import os
from dotenv import load_dotenv

load_dotenv()
# Embedding modelini tanımlayın (İndeksi oluştururken kullandığınız model olmalı)
embeddings_model = OllamaEmbeddings(model="nomic-embed-text")

def semantic_search(user_query, top_k=5):
    # 1. Kullanıcının arama metnini vektöre dönüştür
    query_vector = embeddings_model.embed_query(user_query)
    
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    
    database = os.getenv("NEO4J_DATABASE")
    
    driver = GraphDatabase.driver(uri, auth=(user, password))
    
    cypher = """
    CALL db.index.vector.queryNodes('movie_overview_index', $top_k, $query_vector)
    YIELD node AS movie, score
    RETURN movie.title AS title, movie.overview AS overview, score
    """
    
    session_kwargs = {"database": database} if database else {}
    with driver.session(**session_kwargs) as session:
        result = session.run(cypher, query_vector=query_vector, top_k=top_k)
        records = [dict(record) for record in result]
        
    driver.close()
    return records

# Örnek kullanım:
# results = semantic_search("yalnız bir astronotun uzay macerası")
# for r in results:
#     print(f"Film: {r['title']} (Benzerlik Skoru: {r['score']:.4f})\nÖzet: {r['overview']}\n---")

if __name__ == "__main__":
    search_query = input("Aramak istediğiniz film veya anahtar kelime: ")
    results = semantic_search(search_query, top_k=5)
    for r in results:
        print(f"Film: {r['title']} (Benzerlik Skoru: {r['score']:.4f})\nÖzet: {r['overview']}\n---")
    