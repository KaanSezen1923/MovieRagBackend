"""
Harness'ın (run_eval -> judge toplama -> report) SAHTE LLM ve SAHTE MCP oturumuyla uçtan uca duman testi.
Gerçek Ollama/TMDB/Neo4j gerekmez; yalnızca borunun (şemaların, anahtarların) bozulmadığını doğrular.
"""
import asyncio
import json
from types import SimpleNamespace

import client
import judge
import report
import run_eval
from common import load_jsonl, write_json

CARDS = {
    "semantic": [
        {"movie_id": "10", "Film": "Uzay Yolu", "Yıl": "2014", "TMDB Puanı": "8.4", "vote_count": 900,
         "Türler": "Science Fiction, Drama", "Özet": "uzayda yalnız", "source": "semantic"},
        {"movie_id": "11", "Film": "Korku Evi", "Yıl": "2010", "TMDB Puanı": "6.0", "vote_count": 900,
         "Türler": "Horror", "Özet": "ev", "source": "semantic"},
    ],
}


class FakeLLM:
    """Prompt'a bakıp sabit JSON döndürür."""
    temperature = 0.7

    async def ainvoke(self, msgs):
        sys_, user = msgs[0].content, msgs[-1].content
        if "sorgu analiz bileşenisin" in sys_:
            spec = {"intent": "recommendation", "exclude_genres": ["Horror"], "semantic_query_en": "lonely astronaut"}
            if "Merhaba" in user:
                spec = {"intent": "general"}
            return SimpleNamespace(content=json.dumps(spec))
        if "SON SEÇİM" in sys_:
            return SimpleNamespace(content=json.dumps({"text": "işte", "no_exact_match": False,
                                                        "ranked": [{"id": "10", "reason": "uzayda geçiyor"}]}))
        return SimpleNamespace(content="Merhaba!")


class FakeSession:
    async def call_tool(self, name, args):
        if name == "enrich_movies":
            full = [{**CARDS["semantic"][0], "Director": "D", "Cast": "A, B", "Fragman": "", "Neden Önerildi": ""}]
            return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(full))])
        return SimpleNamespace(content=[SimpleNamespace(text=json.dumps(CARDS["semantic"]))])


def test_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(client, "aclient", FakeLLM())
    client.ctx.session = FakeSession()
    items = [
        {"qid": "T1", "type": "genre_exclude", "prompt": "korku olmasın uzay filmi",
         "gold_spec": {"intent": "recommendation", "exclude_genres": ["Horror"]}},
        {"qid": "T2", "type": "general", "prompt": "Merhaba", "gold_spec": {"intent": "general"}},
    ]
    runs = []
    for it in items:
        for rep in (0, 1):
            runs.append(asyncio.run(run_eval.run_one(client, it, rep, 30, True)))
    assert not any(r["error"] for r in runs), [r["error"] for r in runs]

    r1 = runs[0]
    assert r1["final_ids"] == ["10"]
    assert r1["spec"]["exclude_genres"] == ["Horror"]
    assert r1["candidates_top15"] == ["10"]                       # sert filtre Horror'u eledi
    assert set(r1["variants"]) == {"score_only", "no_filter", "semantic_only"}
    assert "11" in r1["variants"]["no_filter"]                    # filtresiz varyant ihlali yakalar
    assert runs[2]["final_ids"] == [] and runs[2]["final"]["type"] == "text"

    # pool + rapor
    gold = {it["qid"]: it for it in items}
    pool = judge.collect_pool(runs, gold)
    assert set(pool["T1"]) == {"10", "11"} and "T2" not in pool
    labels = {"T1": {"10": {"score": 2, "reason": "", "by": "llm"}, "11": {"score": 0, "reason": "", "by": "llm"}}}
    rel = report.merge_labels(labels, {})
    faith = {"T1|10": {"supported": True, "unsupported_claims": [], "reason": "x"}}
    C, lat, nc, drop = report.build(gold, runs, rel, faith)
    text = report.render(C, lat, nc, drop, len(runs))

    assert C.ci("final/hard_ok")[0] == 1.0
    assert C.ci("no_filter/hard_ok")[0] == 0.0                    # ablation: filtresiz sistem kısıt ihlal ediyor
    assert C.ci("final/ndcg@5")[0] == 1.0
    assert C.ci("faith/supported")[0] == 1.0
    assert C.ci("stability/final_jaccard")[0] == 1.0
    assert C.ci("general/no_movies")[0] == 1.0
    assert "MovieMCP eval raporu" in text
