"""
movies_data.json -> Neo4j graph aktarma script'i

Graf semasi:
  (:Movie {id, title, original_title, overview, release_date, vote_average,
            vote_count, popularity, poster_path})
  (:Genre {id, name})
  (:Person {id, name})
  (:Keyword {id, name})

  (:Movie)-[:HAS_GENRE]->(:Genre)
  (:Person)-[:DIRECTED]->(:Movie)
  (:Person)-[:ACTED_IN {character}]->(:Movie)
  (:Movie)-[:HAS_KEYWORD]->(:Keyword)
  (:Movie)-[:SIMILAR_TO]->(:Movie)

Kullanim:
  export NEO4J_URI="bolt://localhost:7687"
  export NEO4J_USER="neo4j"
  export NEO4J_PASSWORD="sifreniz"
  python import_movies_to_neo4j.py movies_data.json

Gereksinim:
  pip install neo4j
"""

import json
import os
import sys
from itertools import islice
from dotenv import load_dotenv
from neo4j import GraphDatabase

BATCH_SIZE = 500

load_dotenv()


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch


def custom_lower(text):
    if not isinstance(text, str):
        return text
    return text.replace('İ', 'i').replace('I', 'ı').lower()


class MovieImporter:
    def __init__(self, uri, user, password, database="neo4j"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database

    def close(self):
        self.driver.close()

    def run(self, query, **params):
        with self.driver.session(database=self.database) as session:
            return session.run(query, **params).consume()

    # ---------- temizlik ----------
    def clear_database(self):
        query = "MATCH (n) DETACH DELETE n"
        self.run(query)
        print("Mevcut veriler veritabanindan tamamen silindi.")

    # ---------- kurulum ----------
    def create_constraints(self):
        constraints = [
            "CREATE CONSTRAINT movie_id IF NOT EXISTS "
            "FOR (m:Movie) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT genre_id IF NOT EXISTS "
            "FOR (g:Genre) REQUIRE g.id IS UNIQUE",
            "CREATE CONSTRAINT person_id IF NOT EXISTS "
            "FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT keyword_id IF NOT EXISTS "
            "FOR (k:Keyword) REQUIRE k.id IS UNIQUE",
        ]
        for c in constraints:
            self.run(c)
        print("Constraints/index'ler hazir.")

    # ---------- yukleme adimlari ----------
    def load_movies(self, movies):
        query = """
        UNWIND $rows AS row
        MERGE (m:Movie {id: row.id})
        SET m.title = row.title,
            m.original_title = row.original_title,
            m.overview = row.overview,
            m.release_date = row.release_date,
            m.vote_average = row.vote_average,
            m.vote_count = row.vote_count,
            m.popularity = row.popularity,
            m.poster_path = row.poster_path
        """
        rows = [
            {
                "id": m["id"],
                "title": custom_lower(m.get("title")),
                "original_title": custom_lower(m.get("original_title")),
                "overview": custom_lower(m.get("overview")),
                "release_date": m.get("release_date"),
                "vote_average": m.get("vote_average"),
                "vote_count": m.get("vote_count"),
                "popularity": m.get("popularity"),
                "poster_path": m.get("poster_path"),
            }
            for m in movies
        ]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} film node'u yuklendi.")

    def load_genres(self, movies):
        query = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        MERGE (g:Genre {id: row.genre_id})
        ON CREATE SET g.name = row.genre_name
        MERGE (m)-[:HAS_GENRE]->(g)
        """
        rows = [
            {"movie_id": m["id"], "genre_id": g["id"], "genre_name": custom_lower(g["name"])}
            for m in movies
            for g in m.get("genres") or []
        ]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} HAS_GENRE iliskisi yuklendi.")

    def load_directors(self, movies):
        query = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        MERGE (p:Person {id: row.person_id})
        ON CREATE SET p.name = row.person_name
        MERGE (p)-[:DIRECTED]->(m)
        """
        rows = [
            {"movie_id": m["id"], "person_id": d["id"], "person_name": custom_lower(d["name"])}
            for m in movies
            for d in m.get("directors") or []
        ]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} DIRECTED iliskisi yuklendi.")

    def load_cast(self, movies):
        query = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        MERGE (p:Person {id: row.person_id})
        ON CREATE SET p.name = row.person_name
        MERGE (p)-[r:ACTED_IN]->(m)
        SET r.character = row.character
        """
        rows = [
            {
                "movie_id": m["id"],
                "person_id": c["id"],
                "person_name": custom_lower(c["name"]),
                "character": custom_lower(c.get("character")),
            }
            for m in movies
            for c in m.get("cast") or []
        ]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} ACTED_IN iliskisi yuklendi.")

    def load_keywords(self, movies):
        query = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        MERGE (k:Keyword {id: row.keyword_id})
        ON CREATE SET k.name = row.keyword_name
        MERGE (m)-[:HAS_KEYWORD]->(k)
        """
        rows = [
            {"movie_id": m["id"], "keyword_id": k["id"], "keyword_name": custom_lower(k["name"])}
            for m in movies
            for k in m.get("keywords") or []
        ]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} HAS_KEYWORD iliskisi yuklendi.")

    def load_similar_movies(self, movies):
        query = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        MERGE (s:Movie {id: row.similar_id})
        ON CREATE SET s.title = row.similar_title
        MERGE (m)-[:SIMILAR_TO]->(s)
        """
        rows = [
            {
                "movie_id": m["id"],
                "similar_id": s["id"],
                "similar_title": custom_lower(s.get("title")),
            }
            for m in movies
            for s in m.get("similar_movies") or []
        ]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} SIMILAR_TO iliskisi yuklendi.")


def main():
    json_path = "movies_data.json"
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE")


    if not password:
        print("Hata: NEO4J_PASSWORD ortam degiskeni tanimli degil.")
        sys.exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        movies = json.load(f)

    print(f"{len(movies)} film JSON'dan okundu. Neo4j'ye baglaniliyor: {uri}")

    importer = MovieImporter(uri, user, password, database)
    try:
        # Önce eski verileri siliyoruz
        importer.clear_database()
        
        # Sonra kuralları koyup yeni verileri yüklüyoruz
        importer.create_constraints()
        importer.load_movies(movies)
        importer.load_genres(movies)
        importer.load_directors(movies)
        importer.load_cast(movies)
        importer.load_keywords(movies)
        importer.load_similar_movies(movies)
        print("Aktarim tamamlandi.")
    finally:
        importer.close()


if __name__ == "__main__":
    main()