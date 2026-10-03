"""
movies_data.json -> Neo4j graph aktarma script'i

Graf şeması:
  (:Movie {id, title, original_title, title_norm, original_title_norm, overview, tagline,
           release_date, year, vote_average, vote_count, popularity, poster_path,
           original_language, runtime})
  (:Genre {id, name})   (:Person {id, name, name_norm})   (:Keyword {id, name, name_norm})
  (:Movie)-[:HAS_GENRE]->(:Genre)       (:Person)-[:DIRECTED]->(:Movie)
  (:Person)-[:ACTED_IN {character}]->(:Movie)
  (:Movie)-[:HAS_KEYWORD]->(:Keyword)   (:Movie)-[:SIMILAR_TO]->(:Movie)

Kullanım:
  python db.py movies_data.json           # mevcut veriyi günceller (MERGE)
  python db.py movies_data.json --reset   # önce her şeyi siler (şema değişikliğinde bir kez)

DEĞİŞİKLİKLER (eski sürüme göre):
  * Görüntülenen alanlar (title/overview/name) artık ham saklanır. Eski custom_lower,
    'Inception' -> 'ınception' yapıp Cypher'daki toLower('Inception') ile eşleşmiyordu.
  * Arama için ayrı *_norm alanları + index'ler.
  * SIMILAR_TO artık hayalet (verisiz) Movie node'u yaratmaz; sadece var olan filmlere bağlar.
  * year (int) alanı: yıl aralığı filtreleri için.
  * Silme işlemi parça parça yapılır ve sadece --reset ile çalışır.
"""
import argparse
import json
import os
import sys
from itertools import islice

from dotenv import load_dotenv
from neo4j import GraphDatabase

from movie_common import norm

BATCH_SIZE = 500
load_dotenv()


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch


def year_of(release_date):
    y = (release_date or "")[:4]
    return int(y) if y.isdigit() else None


# ---- satır üreticiler (saf fonksiyonlar; Neo4j olmadan test edilebilir) ----
def movie_rows(movies):
    return [
        {
            "id": m["id"],
            "title": m.get("title"),
            "original_title": m.get("original_title"),
            "title_norm": norm(m.get("title")),
            "original_title_norm": norm(m.get("original_title")),
            "overview": m.get("overview"),
            "tagline": m.get("tagline"),
            "release_date": m.get("release_date"),
            "year": year_of(m.get("release_date")),
            "vote_average": m.get("vote_average"),
            "vote_count": m.get("vote_count"),
            "popularity": m.get("popularity"),
            "poster_path": m.get("poster_path"),
            "original_language": m.get("original_language"),
            "runtime": m.get("runtime"),
        }
        for m in movies
    ]


class MovieImporter:
    def __init__(self, uri, user, password, database=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database or "neo4j"

    def close(self):
        self.driver.close()

    def run(self, query, **params):
        with self.driver.session(database=self.database) as session:
            return session.run(query, **params).consume()

    def clear_database(self):
        # Tek transaction'da MATCH (n) DETACH DELETE n büyük grafta belleği şişirir.
        self.run("CALL { MATCH (n) DETACH DELETE n } IN TRANSACTIONS OF 10000 ROWS")
        print("Mevcut veriler silindi.")

    def create_constraints(self):
        stmts = [
            "CREATE CONSTRAINT movie_id IF NOT EXISTS FOR (m:Movie) REQUIRE m.id IS UNIQUE",
            "CREATE CONSTRAINT genre_id IF NOT EXISTS FOR (g:Genre) REQUIRE g.id IS UNIQUE",
            "CREATE CONSTRAINT person_id IF NOT EXISTS FOR (p:Person) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.id IS UNIQUE",
            # sorgularda kullanılan alanlar
            "CREATE INDEX movie_vote_count IF NOT EXISTS FOR (m:Movie) ON (m.vote_count)",
            "CREATE INDEX movie_year IF NOT EXISTS FOR (m:Movie) ON (m.year)",
            "CREATE INDEX movie_title_norm IF NOT EXISTS FOR (m:Movie) ON (m.title_norm)",
            "CREATE INDEX person_name_norm IF NOT EXISTS FOR (p:Person) ON (p.name_norm)",
        ]
        for s in stmts:
            self.run(s)
        print("Constraint/index'ler hazır.")

    def load_movies(self, movies):
        query = """
        UNWIND $rows AS row
        MERGE (m:Movie {id: row.id})
        SET m.title = row.title, m.original_title = row.original_title,
            m.title_norm = row.title_norm, m.original_title_norm = row.original_title_norm,
            m.overview = row.overview, m.tagline = row.tagline,
            m.release_date = row.release_date, m.year = row.year,
            m.vote_average = row.vote_average, m.vote_count = row.vote_count,
            m.popularity = row.popularity, m.poster_path = row.poster_path,
            m.original_language = row.original_language, m.runtime = row.runtime
        """
        rows = movie_rows(movies)
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} film node'u yüklendi.")

    def load_genres(self, movies):
        query = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        MERGE (g:Genre {id: row.genre_id})
        ON CREATE SET g.name = row.genre_name
        MERGE (m)-[:HAS_GENRE]->(g)
        """
        rows = [{"movie_id": m["id"], "genre_id": g["id"], "genre_name": g["name"]}
                for m in movies for g in m.get("genres") or []]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} HAS_GENRE ilişkisi yüklendi.")

    def load_directors(self, movies):
        query = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        MERGE (p:Person {id: row.person_id})
        ON CREATE SET p.name = row.person_name, p.name_norm = row.name_norm
        MERGE (p)-[:DIRECTED]->(m)
        """
        rows = [{"movie_id": m["id"], "person_id": d["id"], "person_name": d["name"],
                 "name_norm": norm(d["name"])}
                for m in movies for d in m.get("directors") or []]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} DIRECTED ilişkisi yüklendi.")

    def load_cast(self, movies):
        query = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        MERGE (p:Person {id: row.person_id})
        ON CREATE SET p.name = row.person_name, p.name_norm = row.name_norm
        MERGE (p)-[r:ACTED_IN]->(m)
        SET r.character = row.character
        """
        rows = [{"movie_id": m["id"], "person_id": c["id"], "person_name": c["name"],
                 "name_norm": norm(c["name"]), "character": c.get("character")}
                for m in movies for c in m.get("cast") or []]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} ACTED_IN ilişkisi yüklendi.")

    def load_keywords(self, movies):
        query = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        MERGE (k:Keyword {id: row.keyword_id})
        ON CREATE SET k.name = row.keyword_name, k.name_norm = row.name_norm
        MERGE (m)-[:HAS_KEYWORD]->(k)
        """
        rows = [{"movie_id": m["id"], "keyword_id": k["id"], "keyword_name": k["name"],
                 "name_norm": norm(k["name"])}
                for m in movies for k in m.get("keywords") or []]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} HAS_KEYWORD ilişkisi yüklendi.")

    def load_similar_movies(self, movies):
        # MATCH (MERGE değil): veritabanında olmayan filmler için verisiz hayalet node yaratma.
        query = """
        UNWIND $rows AS row
        MATCH (m:Movie {id: row.movie_id})
        MATCH (s:Movie {id: row.similar_id})
        MERGE (m)-[:SIMILAR_TO]->(s)
        """
        rows = [{"movie_id": m["id"], "similar_id": s["id"]}
                for m in movies for s in m.get("similar_movies") or []]
        for batch in chunked(rows, BATCH_SIZE):
            self.run(query, rows=batch)
        print(f"{len(rows)} SIMILAR_TO ilişkisi yüklendi.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default="movies_data.json")
    ap.add_argument("--reset", action="store_true", help="önce tüm grafı sil")
    args = ap.parse_args()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    database = os.getenv("NEO4J_DATABASE")
    if not password:
        print("Hata: NEO4J_PASSWORD tanımlı değil.")
        sys.exit(1)

    with open(args.json_path, "r", encoding="utf-8") as f:
        movies = json.load(f)
    print(f"{len(movies)} film okundu. Neo4j: {uri}")

    imp = MovieImporter(uri, user, password, database)
    try:
        if args.reset:
            imp.clear_database()
        imp.create_constraints()
        imp.load_movies(movies)       # similar'dan ÖNCE: tüm film node'ları var olmalı
        imp.load_genres(movies)
        imp.load_directors(movies)
        imp.load_cast(movies)
        imp.load_keywords(movies)
        imp.load_similar_movies(movies)
        print("Aktarım tamamlandı.")
    finally:
        imp.close()


if __name__ == "__main__":
    main()