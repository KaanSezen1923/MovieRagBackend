import os
import time
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

uri = os.getenv("NEO4J_URI")
user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
password = os.getenv("NEO4J_PASSWORD")

print(f"Connecting to Neo4j: {uri} as {user}...")
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

try:
    with driver.session() as session:
        print("Executing CREATE VECTOR INDEX...")
        session.run(query)
        print("Index command executed. Checking status...")
        
        while True:
            result = session.run("SHOW INDEXES YIELD name, type, state, populationPercent WHERE name = 'movie_overview_index'")
            rec = result.single()
            if rec:
                print(f"Index: {rec['name']} | State: {rec['state']} | Population: {rec.get('populationPercent', 'N/A')}%")
                if rec['state'] == 'ONLINE':
                    print("Vector index is ONLINE and ready!")
                    break
            else:
                print("Index not found in SHOW INDEXES yet...")
            time.sleep(2)
finally:
    driver.close()
