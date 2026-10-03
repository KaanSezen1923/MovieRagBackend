"""
Uçtan uca eval koşucusu.

Her soru için grafı (client.app) bir kez çalıştırır ve şunları kaydeder:
  spec (LLM çıkarımı), retrieval çıktıları, sert filtre/skor sonrası aday havuzu,
  LLM'in son seçimi, gecikme. Ek olarak AYNI retrieval çıktısından LLM'siz ablation
  varyantları türetir (ekstra LLM/TMDB maliyeti yok):
    score_only    : LLM seçimi yok, skor sırasıyla ilk N
    no_filter     : sert filtre + LLM seçimi yok
    semantic_only : yalnızca semantik arama kaynağı, LLM seçimi yok

Kullanım (proje kökünden):
  python eval/run_eval.py --data eval/data/dev.jsonl --out eval/runs/dev_full.jsonl \
      --repeats 3 --temperature 0
"""
import argparse
import asyncio
import copy
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))   # client.py, shared.py
sys.path.insert(0, str(ROOT))

from common import append_jsonl, load_jsonl  # noqa: E402

CARD_KEYS = ("movie_id", "Film", "Yıl", "TMDB Puanı", "vote_count", "Türler", "Director", "Cast")


def min_card(c: dict) -> dict:
    d = {k: c.get(k) for k in CARD_KEYS}
    d["Özet"] = (c.get("Özet") or "")[:500]
    if c.get("Neden Önerildi"):
        d["Neden Önerildi"] = c["Neden Önerildi"]
    return d


def parse_cards(raw: str) -> list:
    """client.run_tool ile aynı ayrıştırma."""
    try:
        data = json.loads(raw)
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
    except Exception:
        return []


def derive_variants(client, spec, retrieved: list) -> dict:
    """retrieved: [(tool_name, [cards])]. client'ın kendi fonksiyonlarıyla aşamaları yeniden uygular."""
    limit = client.MAX_MOVIES_PEOPLE if (spec.director or spec.actor) else client.MAX_MOVIES

    def pipeline(pairs, use_filter=True):
        merged = client.merge_candidates(copy.deepcopy([cards for _, cards in pairs]))
        n_merged = len(merged)
        if use_filter:
            merged = client.hard_filter(merged, spec)
        return client.score_candidates(merged, spec), n_merged

    ids = lambda cs: [str(c["movie_id"]) for c in cs]   # noqa: E731
    full, n_merged = pipeline(retrieved)
    nofilter, _ = pipeline(retrieved, use_filter=False)
    sem, _ = pipeline([p for p in retrieved if p[0] == "search_movies_semantically"])
    return {
        "n_merged": n_merged,
        "n_filtered": len(full),
        "candidates_top15": ids(full)[:client.CANDIDATES_TO_LLM],
        "variants": {
            "score_only": ids(full)[:limit],
            "no_filter": ids(nofilter)[:limit],
            "semantic_only": ids(sem)[:limit],
        },
        "limit": limit,
    }


async def run_one(client, item: dict, rep: int, timeout: float, include_prompt: bool) -> dict:
    msgs = list(item.get("messages", []))
    if include_prompt:
        msgs.append({"role": "user", "content": item["prompt"]})
    state = {"prompt": item["prompt"], "messages": msgs, "session_id": None,
             "persona": item.get("persona", "")}
    rec = {"qid": item["qid"], "rep": rep, "prompt": item["prompt"], "error": None}
    t0 = time.perf_counter()
    try:
        out = await asyncio.wait_for(client.app.ainvoke(state), timeout)
    except Exception as e:  # noqa: BLE001
        rec.update(error=repr(e), latency_s=time.perf_counter() - t0)
        return rec
    rec["latency_s"] = round(time.perf_counter() - t0, 3)
    rec["intent"] = out.get("intent")
    rec["spec"] = out.get("intent_data")
    rec["n_tool_calls"] = len(out.get("tool_calls") or [])

    # final çıktı: öneri/bilgi -> JSON; sohbet -> düz metin
    raw_final = out.get("final_output") or ""
    try:
        final = json.loads(raw_final)
        assert isinstance(final, dict)
    except Exception:  # noqa: BLE001
        final = {"type": "text", "text": str(raw_final), "movies": []}
    movies = final.get("movies") or []
    rec["final"] = {"type": final.get("type"), "text": final.get("text", ""),
                    "mood_response": final.get("mood_response", "")}
    rec["final_ids"] = [str(m.get("movie_id")) for m in movies]

    cards = {}
    retrieved = []
    for tr in out.get("tool_results") or []:
        name = tr.get("tool_name")
        if name in ("enrich_movies", "get_movie_card"):
            continue
        lst = parse_cards(tr.get("raw_result", ""))
        retrieved.append((name, lst))
        for c in lst:
            cards.setdefault(str(c.get("movie_id")), min_card(c))
    for m in movies:                       # zenginleştirilmiş kartlar lite kartların üstüne yazar
        cards[str(m.get("movie_id"))] = min_card(m)
    rec["cards"] = cards

    if retrieved:                          # yalnızca öneri hattı retrieval yapar
        try:
            from client import QuerySpec, _clean_spec
            spec = QuerySpec.model_validate(_clean_spec(rec["spec"] or {}))
            rec.update(derive_variants(client, spec, retrieved))
        except Exception as e:  # noqa: BLE001
            rec["derive_error"] = repr(e)
    return rec


async def main_async(args):
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    import client

    if args.temperature is not None:
        client.aclient.temperature = args.temperature

    items = load_jsonl(args.data)
    if args.only:
        items = [i for i in items if i["qid"] in set(args.only.split(","))]
    out_path = Path(args.out)
    if out_path.exists() and not args.append:
        out_path.unlink()

    params = StdioServerParameters(command=sys.executable, args=[args.server], env=dict(os.environ))
    async with stdio_client(params) as (r, w):
        async with ClientSession(r, w) as session:
            await session.initialize()
            client.ctx.session = session
            total = len(items) * args.repeats
            n = 0
            for rep in range(args.repeats):
                for item in items:
                    n += 1
                    rec = await run_one(client, item, rep, args.timeout, not args.history_only)
                    append_jsonl(out_path, rec)
                    status = "HATA " + rec["error"][:60] if rec["error"] else f"{rec['latency_s']:.1f}s"
                    print(f"[{n}/{total}] {item['qid']} rep{rep} {status}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dev.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "dev_full.jsonl"))
    ap.add_argument("--server", default=str(ROOT.parent / "server.py"))
    ap.add_argument("--repeats", type=int, default=1,
                    help="Her soruyu N kez çalıştır (tutarlılık ölçümü). Gecikme için yalnızca rep0 sayılır (cache).")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Verilirse client.aclient.temperature ezilir (varsayılan 0.7 kalır).")
    ap.add_argument("--timeout", type=float, default=180)
    ap.add_argument("--only", default="", help="Virgülle ayrılmış qid listesi")
    ap.add_argument("--append", action="store_true")
    ap.add_argument("--history-only", action="store_true",
                    help="state['messages'] içine güncel mesajı EKLEME (uygulamanız eklemiyorsa).")
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
