"""
series_data.json -> Neo4j   (db.py'nin dizi karşılığı; filmlere DOKUNMAZ)

Şema (filmlerle ortak: Genre, Person, Keyword düğümleri):
  (:Series {id, name, original_name, name_norm, original_name_norm, overview, tagline,
            first_air_date, year, status, seasons, episodes, episode_runtime, networks,
            vote_average, vote_count, popularity, poster_path, original_language})
  (:Series)-[:HAS_GENRE]->(:Genre)     (:Person)-[:CREATED]->(:Series)
  (:Person)-[:ACTED_IN {character}]->(:Series)
  (:Series)-[:HAS_KEYWORD]->(:Keyword)   (:Series)-[:SIMILAR_TO]->(:Series)

NEDEN AYRI ETİKET?  TMDB'de film ve dizi ID'leri ayrı sayaçlardır (film 1399 ile dizi 1399 farklı
şeylerdir). Aynı :Movie etiketine yazsaydık `movie_id` UNIQUE kısıtı çakışırdı.

Kullanım:  python import_series.py series_data.json
"""
import argparse
import json
import os
import sys
from itertools import islice

from dotenv import load_dotenv
from neo4j import GraphDatabase

from movie_common import norm

load_dotenv()
BATCH_SIZE = 500


def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            return
        yield batch


def year_of(d):
    y = (d or "")[:4]
    return int(y) if y.isdigit() else None


class SeriesImporter:
    def __init__(self, uri, user, password, database=None):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.database = database or "neo4j"

    def close(self):
        self.driver.close()

    def run(self, query, **params):
        with self.driver.session(database=self.database) as s:
            return s.run(query, **params).consume()

    def create_constraints(self):
        for stmt in [
            "CREATE CONSTRAINT series_id IF NOT EXISTS FOR (s:Series) REQUIRE s.id IS UNIQUE",
            "CREATE INDEX series_vote_count IF NOT EXISTS FOR (s:Series) ON (s.vote_count)",
            "CREATE INDEX series_year IF NOT EXISTS FOR (s:Series) ON (s.year)",
            "CREATE INDEX series_name_norm IF NOT EXISTS FOR (s:Series) ON (s.name_norm)",
            # Genre/Person/Keyword kısıtları db.py ile zaten var.
        ]:
            self.run(stmt)
        print("Dizi constraint/index'leri hazır.")

    def _batches(self, query, rows, label):
        for b in chunked(rows, BATCH_SIZE):
            self.run(query, rows=b)
        print(f"{len(rows)} {label} yüklendi.")

    def load_series(self, items):
        rows = [{
            "id": s["id"], "name": s.get("name"), "original_name": s.get("original_name"),
            "name_norm": norm(s.get("name")), "original_name_norm": norm(s.get("original_name")),
            "overview": s.get("overview"), "tagline": s.get("tagline"),
            "first_air_date": s.get("first_air_date"), "year": year_of(s.get("first_air_date")),
            "status": s.get("status"), "seasons": s.get("number_of_seasons"),
            "episodes": s.get("number_of_episodes"), "episode_runtime": s.get("episode_runtime"),
            "networks": s.get("networks") or [],
            "vote_average": s.get("vote_average"), "vote_count": s.get("vote_count"),
            "popularity": s.get("popularity"), "poster_path": s.get("poster_path"),
            "original_language": s.get("original_language"),
        } for s in items]
        self._batches("""
        UNWIND $rows AS row
        MERGE (s:Series {id: row.id})
        SET s.name = row.name, s.original_name = row.original_name,
            s.name_norm = row.name_norm, s.original_name_norm = row.original_name_norm,
            s.overview = row.overview, s.tagline = row.tagline,
            s.first_air_date = row.first_air_date, s.year = row.year, s.status = row.status,
            s.seasons = row.seasons, s.episodes = row.episodes, s.episode_runtime = row.episode_runtime,
            s.networks = row.networks, s.vote_average = row.vote_average, s.vote_count = row.vote_count,
            s.popularity = row.popularity, s.poster_path = row.poster_path,
            s.original_language = row.original_language
        """, rows, "dizi düğümü")

    def load_genres(self, items):
        rows = [{"sid": s["id"], "gid": g["id"], "gname": g["name"]}
                for s in items for g in s.get("genres") or []]
        self._batches("""
        UNWIND $rows AS row
        MATCH (s:Series {id: row.sid})
        MERGE (g:Genre {id: row.gid}) ON CREATE SET g.name = row.gname
        MERGE (s)-[:HAS_GENRE]->(g)
        """, rows, "HAS_GENRE")

    def load_creators(self, items):
        rows = [{"sid": s["id"], "pid": c["id"], "pname": c["name"], "pn": norm(c["name"])}
                for s in items for c in s.get("creators") or []]
        self._batches("""
        UNWIND $rows AS row
        MATCH (s:Series {id: row.sid})
        MERGE (p:Person {id: row.pid}) ON CREATE SET p.name = row.pname, p.name_norm = row.pn
        MERGE (p)-[:CREATED]->(s)
        """, rows, "CREATED")

    def load_cast(self, items):
        rows = [{"sid": s["id"], "pid": c["id"], "pname": c["name"], "pn": norm(c["name"]),
                 "character": c.get("character")}
                for s in items for c in s.get("cast") or []]
        self._batches("""
        UNWIND $rows AS row
        MATCH (s:Series {id: row.sid})
        MERGE (p:Person {id: row.pid}) ON CREATE SET p.name = row.pname, p.name_norm = row.pn
        MERGE (p)-[r:ACTED_IN]->(s) SET r.character = row.character
        """, rows, "ACTED_IN")

    def load_keywords(self, items):
        rows = [{"sid": s["id"], "kid": k["id"], "kname": k["name"], "kn": norm(k["name"])}
                for s in items for k in s.get("keywords") or []]
        self._batches("""
        UNWIND $rows AS row
        MATCH (s:Series {id: row.sid})
        MERGE (k:Keyword {id: row.kid}) ON CREATE SET k.name = row.kname, k.name_norm = row.kn
        MERGE (s)-[:HAS_KEYWORD]->(k)
        """, rows, "HAS_KEYWORD")

    def load_similar(self, items):
        rows = [{"a": s["id"], "b": x["id"]} for s in items for x in s.get("similar_series") or []]
        self._batches("""
        UNWIND $rows AS row
        MATCH (a:Series {id: row.a}) MATCH (b:Series {id: row.b})
        MERGE (a)-[:SIMILAR_TO]->(b)
        """, rows, "SIMILAR_TO")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", nargs="?", default="series_data.json")
    args = ap.parse_args()

    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME") or os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD")
    if not password:
        print("Hata: NEO4J_PASSWORD tanımlı değil.")
        sys.exit(1)

    with open(args.json_path, "r", encoding="utf-8") as f:
        items = json.load(f)
    print(f"{len(items)} dizi okundu. Neo4j: {uri}")

    imp = SeriesImporter(uri, user, password, os.getenv("NEO4J_DATABASE"))
    try:
        imp.create_constraints()
        imp.load_series(items)          # ilişkilerden ÖNCE
        imp.load_genres(items)
        imp.load_creators(items)
        imp.load_cast(items)
        imp.load_keywords(items)
        imp.load_similar(items)
        print("Dizi aktarımı tamamlandı.")
    finally:
        imp.close()


if __name__ == "__main__":
    main()
