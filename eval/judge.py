"""
LLM-judge araçları.

  pool   : tüm koşulardaki (final + ablation + aday havuzu) filmleri birleştirip 0/1/2 relevance etiketler
  faith  : 'Neden Önerildi' gerekçelerinin kartla desteklenip desteklenmediğini kontrol eder
  sample : insan doğrulaması için rastgele örnek CSV'si üretir
  agree  : insan etiketleri ile judge arasında uyum (yüzde + Cohen's kappa)

ÖNEMLİ: JUDGE_MODEL, sistemin kendi modelinden (OLLAMA_MODEL) FARKLI ve tercihen daha güçlü olmalı.
  JUDGE_MODEL ver (biri yeter):
    .env dosyasına:   JUDGE_MODEL=...
    PowerShell:       $env:JUDGE_MODEL="..."
    cmd:              set JUDGE_MODEL=...
    Linux/macOS:      export JUDGE_MODEL=...
    ya da komutta:    --judge-model ...
  JUDGE_BASE_URL (isteğe bağlı; varsayılan OLLAMA_HOST ya da http://127.0.0.1:11434)

Başka bir sağlayıcı kullanacaksan yalnızca `call_judge` fonksiyonunu değiştir.
"""
import argparse
import asyncio
import csv
import json
import os
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)   # client.py ile aynı: .env, sistem ortam değişkenini ezer
except ImportError:
    pass

from common import DATA, load_jsonl, read_json, write_json  # noqa: E402
from metrics import cohen_kappa  # noqa: E402

REL_SYSTEM = """Sen bir film öneri sisteminin DEĞERLENDİRİCİSİSİN. Kullanıcı isteği, kriterler ve tek bir film verilecek.
Filmin bilgisini hem verilen kartlardan hem kendi film bilginden kullan.

Puanla:
2 = Tüm açık kriterleri (tür, yıl, puan aralığı, 'sert kriterler') KARŞILIYOR ve konu/atmosfer olarak isteğe belirgin uyuyor.
1 = Sayısal/tür kriterlerini karşılıyor ama konu/atmosfer olarak kısmen uyuyor; ya da sert kriterlerden birini doğrulayamıyorsun.
0 = Açık bir kriteri İHLAL ediyor (hariç tutulan tür, yıl/puan aralığı dışı, sert kriter ihlali) ya da konu olarak alakasız.

Emin değilsen 1 ver. Yalnızca JSON döndür: {"score": 0|1|2, "reason": "1 cümle"}"""

FAITH_SYSTEM = """Bir film önerisinin GEREKÇE cümlesini denetliyorsun. Sana filmin kartı (tür, yıl, puan, özet, yönetmen, oyuncular)
ve gerekçe verilecek. Gerekçedeki her olgusal iddianın kartta desteklenip desteklenmediğine bak.
Kartta olmayan ya da karttan çıkarılamayan iddia (besteci, sahne detayı, renk paleti, ödül, bütçe vb.) 'desteksiz'dir.
Genel yargılar ("sürükleyici bir film") iddia sayılmaz. Yalnızca JSON döndür:
{"supported": true|false, "unsupported_claims": ["..."]}"""

def _normalize_ollama_host(raw):
    """client.py ile birebir aynı: boş/0.0.0.0/localhost -> 127.0.0.1 (Windows'ta 'All connection attempts failed' sebebi)."""
    from urllib.parse import urlparse
    raw = (raw or "").strip() or "http://127.0.0.1:11434"
    if "://" not in raw:
        raw = "http://" + raw
    u = urlparse(raw)
    if (u.hostname or "") in ("0.0.0.0", "localhost", "::", ""):
        return f"{u.scheme}://127.0.0.1:{u.port or 11434}"
    return raw


_llm = None
_model_override = None
_base_url = None


def _get_llm():
    global _llm
    if _llm is None:
        from langchain_ollama import ChatOllama
        model = _model_override or os.environ.get("JUDGE_MODEL")
        if not model:
            sys.exit("JUDGE_MODEL ayarlı değil. `--judge-model <ad>` ver ya da .env'e JUDGE_MODEL=<ad> yaz "
                     "(sistem modelinden farklı bir model seç; PowerShell: $env:JUDGE_MODEL=\"<ad>\").")
        if model == os.environ.get("OLLAMA_MODEL"):
            print("UYARI: judge modeli sistem modeliyle aynı; kendi çıktısını kayırabilir.", file=sys.stderr)
        global _base_url
        base = _base_url = _normalize_ollama_host(os.environ.get("JUDGE_BASE_URL") or os.environ.get("OLLAMA_HOST"))
        kw = {"trust_env": False}
        if os.environ.get("OLLAMA_API_KEY"):
            kw["headers"] = {"Authorization": f"Bearer {os.environ['OLLAMA_API_KEY']}"}
        _llm = ChatOllama(model=model, temperature=0, base_url=base, client_kwargs=kw)
    return _llm


def _parse(raw: str) -> dict:
    clean = re.sub(r"```json\s?|```", "", raw or "").strip()
    m = re.search(r"(\{.*\})", clean, re.DOTALL)
    return json.loads(m.group(1) if m else clean)


async def call_judge(system: str, user: str, retries: int = 2) -> dict:
    from langchain_core.messages import HumanMessage, SystemMessage
    last = None
    for _ in range(retries + 1):
        try:
            resp = await _get_llm().ainvoke([SystemMessage(content=system), HumanMessage(content=user)])
            return _parse(resp.content if isinstance(resp.content, str) else str(resp.content))
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"judge başarısız ({_base_url}): {last!r}")


def card_text(c: dict) -> str:
    return (f"{c.get('Film')} ({c.get('Yıl')}) | puan {c.get('TMDB Puanı')} ({c.get('vote_count')} oy) | "
            f"{c.get('Türler')} | Yön: {c.get('Director') or '?'} | Oyuncular: {c.get('Cast') or '?'}\n"
            f"Özet: {c.get('Özet')}")


def gold_criteria(g: dict) -> str:
    spec = g.get("gold_spec", {})
    parts = []
    if spec.get("include_genres"):
        parts.append("İstenen türler: " + ", ".join(spec["include_genres"]))
    if spec.get("exclude_genres"):
        parts.append("İSTENMEYEN türler: " + ", ".join(spec["exclude_genres"]))
    for k, label in (("min_rating", "en az puan"), ("max_rating", "en çok puan"),
                     ("year_from", "yıl >="), ("year_to", "yıl <=")):
        if spec.get(k) is not None:
            parts.append(f"{label}: {spec[k]}")
    if spec.get("reference_movie"):
        parts.append("Benzeri istenen film: " + str(spec["reference_movie"]).split("|")[0])
    if spec.get("director"):
        parts.append("Yönetmen: " + str(spec["director"]))
    if spec.get("actor"):
        parts.append("Oyuncu: " + str(spec["actor"]))
    if g.get("must_have"):
        parts.append("Sert kriterler: " + "; ".join(g["must_have"]))
    return "\n".join(parts) or "(ek kriter yok)"


def collect_pool(runs: list, gold: dict) -> dict:
    """qid -> {movie_id: card}. Yalnızca öneri sorguları (cevabı boş olması beklenenler hariç)."""
    pool = {}
    for r in runs:
        g = gold.get(r["qid"])
        if not g or r.get("error") or g.get("expect_empty"):
            continue
        if g.get("gold_spec", {}).get("intent", "recommendation") != "recommendation":
            continue
        ids = set(r.get("final_ids", [])) | set(r.get("candidates_top15", []))
        for v in (r.get("variants") or {}).values():
            ids |= set(v)
        for i in ids:
            if i in r.get("cards", {}):
                pool.setdefault(r["qid"], {}).setdefault(i, r["cards"][i])
    return pool


async def _gather(coros, conc):
    sem = asyncio.Semaphore(conc)

    async def guard(c):
        async with sem:
            return await c
    return await asyncio.gather(*[guard(c) for c in coros], return_exceptions=True)


def _init_judge(a):
    """Model ayarını coroutine'ler oluşturulmadan ÖNCE doğrula (hızlı ve temiz hata)."""
    global _model_override
    _model_override = getattr(a, "judge_model", None) or None
    llm = _get_llm()
    try:   # ön kontrol: 400 çağrı patlamadan önce bağlantıyı/modeli doğrula
        llm.invoke("ping")
    except Exception as e:  # noqa: BLE001
        msg = str(e)
        hint = ("Ollama çalışmıyor ya da adres yanlış. Ollama uygulaması açık mı? "
                "Tarayıcıda şu adres açılıyor mu: " + str(_base_url)) if "connect" in type(e).__name__.lower() or "connect" in msg.lower() \
            else ("Model bulunamadı. `ollama list` ile kontrol et; bulut modeli için `ollama signin` yap ve "
                  "`ollama pull <model>` / `ollama run <model>` ile bir kez çalıştır.") if "not found" in msg.lower() or "404" in msg \
            else ("Yetki hatası: `ollama signin` yap ya da OLLAMA_API_KEY'i kontrol et.") if "401" in msg or "403" in msg or "unauthor" in msg.lower() \
            else "Ayrıntı için aşağıdaki hataya bak."
        sys.exit(f"Judge ön kontrolü BAŞARISIZ\n  adres : {_base_url}\n  model : {llm.model}\n  hata  : {type(e).__name__}: {msg[:300]}\n  ipucu : {hint}")
    print(f"judge hazır: {llm.model} @ {_base_url}")


def cmd_pool(a):
    _init_judge(a)
    gold = {g["qid"]: g for g in load_jsonl(a.data)}
    runs = [r for p in a.runs for r in load_jsonl(p)]
    labels = read_json(a.out, {})
    todo = []
    for qid, cards in collect_pool(runs, gold).items():
        for mid, card in cards.items():
            if mid not in labels.get(qid, {}):
                todo.append((qid, mid, card))
    print(f"{len(todo)} film etiketlenecek (zaten etiketli: {sum(len(v) for v in labels.values())})")

    async def one(qid, mid, card):
        g = gold[qid]
        user = (f"Kullanıcı isteği: {g['prompt']}\nKriterler:\n{gold_criteria(g)}\n\nFilm:\n{card_text(card)}")
        j = await call_judge(REL_SYSTEM, user)
        return qid, mid, {"score": int(j["score"]), "reason": str(j.get("reason", ""))[:300], "by": "llm"}

    async def go():
        res = await _gather([one(*t) for t in todo], a.concurrency)
        for r in res:
            if isinstance(r, Exception):
                print("  hata:", r, file=sys.stderr)
                continue
            qid, mid, lab = r
            labels.setdefault(qid, {})[mid] = lab
        write_json(a.out, labels)
    asyncio.run(go())
    print("yazıldı:", a.out)


def cmd_faith(a):
    _init_judge(a)
    gold = {g["qid"]: g for g in load_jsonl(a.data)}
    runs = [r for p in a.runs for r in load_jsonl(p) if r.get("rep", 0) == 0 and not r.get("error")]
    out = read_json(a.out, {})
    todo = []
    for r in runs:
        for mid in r.get("final_ids", []):
            card = r["cards"].get(mid, {})
            reason = card.get("Neden Önerildi")
            key = f"{r['qid']}|{mid}"
            if reason and key not in out:
                todo.append((key, card, reason))
    print(f"{len(todo)} gerekçe denetlenecek")

    async def one(key, card, reason):
        j = await call_judge(FAITH_SYSTEM, f"Kart:\n{card_text(card)}\n\nGerekçe: {reason}")
        return key, {"supported": bool(j.get("supported")),
                     "unsupported_claims": j.get("unsupported_claims", []), "reason": reason}

    async def go():
        for r in await _gather([one(*t) for t in todo], a.concurrency):
            if isinstance(r, Exception):
                print("  hata:", r, file=sys.stderr)
            else:
                out[r[0]] = r[1]
        write_json(a.out, out)
    asyncio.run(go())
    print("yazıldı:", a.out)


def cmd_sample(a):
    labels = read_json(a.labels, {})
    gold = {g["qid"]: g for g in load_jsonl(a.data)}
    runs = [r for p in a.runs for r in load_jsonl(p)]
    pool = collect_pool(runs, gold)
    rows = [(q, m, pool[q][m], lab["score"]) for q, d in labels.items() for m, lab in d.items()
            if q in pool and m in pool[q]]
    random.Random(a.seed).shuffle(rows)
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["qid", "movie_id", "prompt", "criteria", "film", "year", "genres", "overview",
                    "llm_score", "human_score"])
        for q, m, c, s in rows[:a.n]:
            w.writerow([q, m, gold[q]["prompt"], gold_criteria(gold[q]).replace("\n", " | "),
                        c.get("Film"), c.get("Yıl"), c.get("Türler"), c.get("Özet"), s, ""])
    print(f"{min(a.n, len(rows))} örnek -> {a.out}. 'human_score' sütununu 0/1/2 ile doldur, sonra `agree` çalıştır.")


def load_human(path) -> dict:
    out = {}
    if not Path(path).exists():
        return out
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("human_score", "").strip() in ("0", "1", "2"):
                out.setdefault(row["qid"], {})[row["movie_id"]] = int(row["human_score"])
    return out


def cmd_agree(a):
    labels = read_json(a.labels, {})
    human = load_human(a.human)
    h, l = [], []
    for q, d in human.items():
        for m, s in d.items():
            if m in labels.get(q, {}):
                h.append(s)
                l.append(labels[q][m]["score"])
    if not h:
        sys.exit("İnsan etiketi bulunamadı.")
    agree = sum(x == y for x, y in zip(h, l)) / len(h)
    near = sum(abs(x - y) <= 1 for x, y in zip(h, l)) / len(h)
    bin_h, bin_l = [int(x >= 1) for x in h], [int(x >= 1) for x in l]
    print(f"n={len(h)} | birebir uyum={agree:.2%} | kappa={cohen_kappa(h, l):.3f} | "
          f"ilgili/ilgisiz (>=1) kappa={cohen_kappa(bin_h, bin_l):.3f}")
    print("Kabaca: kappa < 0.4 zayıf (judge'ı/rubriği düzelt), 0.4-0.6 orta, > 0.6 iyi.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("pool", "faith", "sample"):
        p = sub.add_parser(name)
        p.add_argument("--runs", nargs="+", required=True)
        p.add_argument("--data", default=str(DATA / "dev.jsonl"))
        p.add_argument("--concurrency", type=int, default=4)
        p.add_argument("--judge-model", default=None, help="JUDGE_MODEL ortam değişkenini ezer")
    sub.choices["pool"].add_argument("--out", default=str(DATA / "labels_dev.json"))
    sub.choices["faith"].add_argument("--out", default=str(DATA / "faith_dev.json"))
    s = sub.choices["sample"]
    s.add_argument("--labels", default=str(DATA / "labels_dev.json"))
    s.add_argument("--out", default=str(DATA / "human_sample_dev.csv"))
    s.add_argument("--n", type=int, default=40)
    s.add_argument("--seed", type=int, default=0)
    g = sub.add_parser("agree")
    g.add_argument("--labels", default=str(DATA / "labels_dev.json"))
    g.add_argument("--human", default=str(DATA / "human_sample_dev.csv"))
    a = ap.parse_args()
    {"pool": cmd_pool, "faith": cmd_faith, "sample": cmd_sample, "agree": cmd_agree}[a.cmd](a)


if __name__ == "__main__":
    main()