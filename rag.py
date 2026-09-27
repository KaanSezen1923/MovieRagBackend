import os
import sys
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_ollama import ChatOllama
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
import json 
load_dotenv()

# ==========================================
# 1. VERİTABANI VE LLM KURULUMU
# ==========================================
class MovieRecommender:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def get_recommendations(self, query, param_value):
        with self.driver.session(database=self.database) as session:
            result = session.run(query, val=param_value)
            records = list(result)
            return records

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD")
database = os.getenv("NEO4J_DATABASE", "neo4j")

if not password:
    print("Hata: NEO4J_PASSWORD ortam değişkeni tanımlı değil.")
    sys.exit(1)

recommender = MovieRecommender(uri, user, password, database)
model = ChatOllama(model="gemma4:31b-cloud", temperature=0.8)

# ==========================================
# ESNEK CYPHER SORGULARI (CONTAINS & toLower İle Optimize Edildi)
# ==========================================
queries = {
    "Director": {
        "cypher": """
            MATCH (p:Person)-[:DIRECTED]->(m:Movie)
            WHERE toLower(p.name) CONTAINS toLower($val)
            RETURN m AS FilmDetaylari,
                   [(m)<-[:DIRECTED]-(d:Person) | d.name] AS Yonetmenler,
                   [(m)<-[r:ACTED_IN]-(c:Person) | {oyuncu: c.name, karakter: r.character}] AS Oyuncular,
                   [(m)-[:HAS_GENRE]->(g:Genre) | g.name] AS Turler
            ORDER BY m.vote_average DESC LIMIT 5
        """
    },
    "Actor": {
        "cypher": """
            MATCH (p:Person)-[r:ACTED_IN]->(m:Movie)
            WHERE toLower(p.name) CONTAINS toLower($val)
            RETURN m AS FilmDetaylari,
                   [(m)<-[:DIRECTED]-(d:Person) | d.name] AS Yonetmenler,
                   [(m)<-[r2:ACTED_IN]-(c:Person) | {oyuncu: c.name, karakter: r2.character}] AS Oyuncular,
                   [(m)-[:HAS_GENRE]->(g:Genre) | g.name] AS Turler
            ORDER BY m.popularity DESC LIMIT 5
        """
    },
    "Genre": {
        "cypher": """
            MATCH (m:Movie)-[:HAS_GENRE]->(g:Genre)
            WHERE toLower(g.name) CONTAINS toLower($val)
            RETURN m AS FilmDetaylari,
                   [(m)<-[:DIRECTED]-(d:Person) | d.name] AS Yonetmenler,
                   [(m)<-[r:ACTED_IN]-(c:Person) | {oyuncu: c.name, karakter: r.character}] AS Oyuncular,
                   [(m)-[:HAS_GENRE]->(gen:Genre) | gen.name] AS Turler
            ORDER BY m.popularity DESC LIMIT 5
        """
    },
    "Keyword": {
        "cypher": """
            MATCH (m:Movie)-[:HAS_KEYWORD]->(k:Keyword)
            WHERE toLower(k.name) CONTAINS toLower($val)
            RETURN m AS FilmDetaylari,
                   [(m)<-[:DIRECTED]-(d:Person) | d.name] AS Yonetmenler,
                   [(m)<-[r:ACTED_IN]-(c:Person) | {oyuncu: c.name, karakter: r.character}] AS Oyuncular,
                   [(m)-[:HAS_GENRE]->(g:Genre) | g.name] AS Turler
            ORDER BY m.vote_average DESC LIMIT 5
        """
    },
    "Similar": {
        "cypher": """
            MATCH (m:Movie)
            WHERE toLower(m.title) CONTAINS toLower($val)
            MATCH (m)-[:SIMILAR_TO]->(similar:Movie)
            RETURN similar AS FilmDetaylari,
                   [(similar)<-[:DIRECTED]-(d:Person) | d.name] AS Yonetmenler,
                   [(similar)<-[r:ACTED_IN]-(c:Person) | {oyuncu: c.name, karakter: r.character}] AS Oyuncular,
                   [(similar)-[:HAS_GENRE]->(g:Genre) | g.name] AS Turler
            LIMIT 5
        """
    }
}

# ==========================================
# 2. YARDIMCI FONKSİYONLAR
# ==========================================
def custom_lower(text):
    if not isinstance(text, str):
        return text
    return text.replace('İ', 'i').replace('I', 'ı').lower()

def format_db_records(records, search_value):
    if not records:
        return "[]"
    
    movies = []
    for record in records:
        film_props = dict(record["FilmDetaylari"])
        
        movies.append({
            "title": film_props.get("title", "Bilinmiyor"),
            "overview": film_props.get("overview", "Özet bulunmuyor."),
            "director": ", ".join(record["Yonetmenler"]) if record["Yonetmenler"] else "Bilinmiyor",
            "cast": [o['oyuncu'] for o in record["Oyuncular"]],
            "genre": ", ".join(record["Turler"]) if record["Turler"] else "Bilinmiyor",
            "poster_path": film_props.get("poster_path", "")
        })
        
    return json.dumps(movies, ensure_ascii=False)

# ==========================================
# 3. LANGGRAPH YAPISI VE DÜĞÜMLER
# ==========================================
class GraphState(TypedDict):
    user_input: str
    category: Optional[str]
    search_value: Optional[str]
    db_results: Optional[str]
    final_answer: Optional[str]

def classify_input_node(state: GraphState) -> GraphState:
    user_input = state["user_input"]
    
    messages = [
        ("system", "Sen bir niyet sınıflandırıcısın. Girdiyi analiz et ve SADECE 'Kategori|Değer' formatında tek satır çıktı ver. "
                   "Kategoriler: Director, Actor, Genre, Keyword, Similar. "
                   "Örnek 1: 'aksiyon filmi öner' -> Genre|Action "
                   "Örnek 2: 'christopher nolan filmleri' -> Director|Christopher Nolan "
                   "Örnek 3: 'matrix benzeri filmler' -> Similar|The Matrix "
                   "Örnek 4: 'brad pitt oynadığı filmler' -> Actor|Brad Pitt "
                   "Örnek 5: 'zaman yolculuğu konulu filmler' -> Keyword|time travel"),
        ("human", user_input)
    ]
    
    response = model.invoke(messages)
    content = response.content.strip()
    
    try:
        category, search_value = content.split("|")
        return {"category": category.strip(), "search_value": search_value.strip()}
    except ValueError:
        return {"category": "Unknown", "search_value": user_input}

def director_tool_node(state: GraphState) -> GraphState:
    val = state.get("search_value", "")
    records = recommender.get_recommendations(queries["Director"]["cypher"], custom_lower(val))
    return {"db_results": format_db_records(records, val)}

def actor_tool_node(state: GraphState) -> GraphState:
    val = state.get("search_value", "")
    records = recommender.get_recommendations(queries["Actor"]["cypher"], custom_lower(val))
    return {"db_results": format_db_records(records, val)}

def genre_tool_node(state: GraphState) -> GraphState:
    val = state.get("search_value", "")
    records = recommender.get_recommendations(queries["Genre"]["cypher"], custom_lower(val))
    return {"db_results": format_db_records(records, val)}

def keyword_tool_node(state: GraphState) -> GraphState:
    val = state.get("search_value", "")
    records = recommender.get_recommendations(queries["Keyword"]["cypher"], custom_lower(val))
    return {"db_results": format_db_records(records, val)}

def similar_tool_node(state: GraphState) -> GraphState:
    val = state.get("search_value", "")
    records = recommender.get_recommendations(queries["Similar"]["cypher"], val)
    return {"db_results": format_db_records(records, val)}

def generate_response_node(state: GraphState) -> GraphState:
    db_results = state.get("db_results", "[]")
    
    messages = [
        ("system", "Sen film kartları için veri sağlayan bir API'sin. Kullanıcının sorusunu görmezden gel ve SADECE sana verilen veritabanı sonuçlarını JSON dizisi (array) olarak döndür. Hiçbir selamlama, açıklama veya düz metin yazma. Markdown kod blokları (```json) KULLANMA. Çıktın doğrudan parse edilebilir saf JSON olmalıdır. Beklenen anahtarlar: title, overview, director, cast, genre, poster_path."),
        ("human", f"Veritabanı Sonuçları:\n{db_results}")
    ]
    
    response = model.invoke(messages)
    clean_json = response.content.replace("```json", "").replace("```", "").strip()
    
    return {"final_answer": clean_json}

def route_to_tool(state: GraphState) -> str:
    category = state.get("category")
    
    if category == "Director": return "director_tool"
    elif category == "Actor": return "actor_tool"
    elif category == "Genre": return "genre_tool"
    elif category == "Keyword": return "keyword_tool"
    elif category == "Similar": return "similar_tool"
    else: return "generate_response"

# ==========================================
# 4. GRAF İNŞASI VE DERLEME
# ==========================================
workflow = StateGraph(GraphState)

workflow.add_node("classifier", classify_input_node)
workflow.add_node("director_tool", director_tool_node)
workflow.add_node("actor_tool", actor_tool_node)
workflow.add_node("genre_tool", genre_tool_node)
workflow.add_node("keyword_tool", keyword_tool_node)
workflow.add_node("similar_tool", similar_tool_node)
workflow.add_node("generate_response", generate_response_node)

workflow.add_edge(START, "classifier")

workflow.add_conditional_edges(
    "classifier",
    route_to_tool,
    {
        "director_tool": "director_tool",
        "actor_tool": "actor_tool",
        "genre_tool": "genre_tool",
        "keyword_tool": "keyword_tool",
        "similar_tool": "similar_tool",
        "generate_response": "generate_response"
    }
)

tool_nodes = ["director_tool", "actor_tool", "genre_tool", "keyword_tool", "similar_tool"]
for tool_node in tool_nodes:
    workflow.add_edge(tool_node, "generate_response")

workflow.add_edge("generate_response", END)

rag_app = workflow.compile()

# ==========================================
# 5. UYGULAMA DÖNGÜSÜ
# ==========================================
if __name__ == "__main__":
    print("Sinema Asistanı (LangGraph): Merhaba! Size nasıl film önerilerinde bulunabilirim? (Çıkmak için 'q' yazın)")
    
    try:
        while True:
            try:
                user_input = input("\nSen: ")
                
                if user_input.lower() in ['q', 'quit', 'çıkış']:
                    print("Görüşmek üzere!")
                    break
                
                if not user_input.strip():
                    continue

                initial_state = {
                    "user_input": user_input,
                    "category": None,
                    "search_value": None,
                    "db_results": None,
                    "final_answer": None
                }
                
                print("\n--- LangGraph Çalışma Adımları ---")
                final_answer = "Üzgünüm, bir yanıt üretemedim veya akışta bir hata oluştu."
                
                for event in rag_app.stream(initial_state):
                    for node_name, node_state in event.items():
                        print(f"✅ Çalışan Düğüm: {node_name}")
                        if node_name == "classifier":
                            print(f"   ↳ Tespit Edilen Kategori: {node_state.get('category')}")
                            print(f"   ↳ Aranan Değer: {node_state.get('search_value')}")
                        elif node_name.endswith("_tool"):
                            db_res = node_state.get('db_results', '')
                            if "[]" in db_res or not db_res:
                                print("   ↳ Neo4j: Sonuç bulunamadı.")
                            else:
                                print(f"   ↳ Neo4j: Veriler başarıyla çekildi (Metin Uzunluğu: {len(db_res)} karakter)")
                        elif node_name == "generate_response":
                            final_answer = node_state.get("final_answer", final_answer)
                
                print("----------------------------------\n")
                print("Asistan:\n", final_answer)
                
            except KeyboardInterrupt:
                print("\nGörüşmek üzere!")
                break
            except Exception as e:
                print(f"\nSistem Hatası: {e}")
    finally:
        recommender.close()