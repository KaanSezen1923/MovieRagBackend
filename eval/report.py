"""
Koşu çıktılarından rapor üretir (markdown).

  python eval/report.py --runs eval/runs/dev_full.jsonl --data eval/data/dev.jsonl \
      --labels eval/data/labels_dev.json --faith eval/data/faith_dev.json --out eval/runs/report_dev.md

Tüm ortalamalar SORGU bazındadır (tekrarlar önce sorgu içinde ortalanır); güven aralığı sorgular
üzerinden bootstrap ile hesaplanır. Küçük n'de aralıklara bak, tek bir ortalamaya değil.
"""
import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import DATA, load_jsonl, norm_text, read_json  # noqa: E402
from judge import load_human  # noqa: E402
from metrics import (NUM_FIELDS, _alts, bootstrap_ci, check_constraints, eval_spec, has_hard_constraints,  # noqa: E402
                     jaccard, label_coverage, ndcg_at_k, oracle_ndcg, percentile, precision_at_k,
                     spec_exact)

VARIANTS = ("final", "score_only", "no_filter", "semantic_only")


class Coll:
    def __init__(self):
        self.d = defaultdict(lambda: defaultdict(list))   # metrik -> qid -> [değerler]
        self.types = {}

    def add(self, metric, qid, val):
        if val is not None:
            self.d[metric][qid].append(float(val))

    def per_query(self, metric, qids=None):
        return [statistics.fmean(v) for q, v in self.d[metric].items() if qids is None or q in qids]

    def ci(self, metric, qids=None):
        return bootstrap_ci(self.per_query(metric, qids))


def fmt(t):
    mean, lo, hi, n = t
    return "—" if n == 0 else f"{mean:.3f} [{lo:.2f}, {hi:.2f}] (n={n})"


def merge_labels(llm: dict, human: dict) -> dict:
    """qid -> {movie_id: 0/1/2}. İnsan etiketi varsa judge'ı ezer."""
    rel = {q: {m: lab["score"] for m, lab in d.items()} for q, d in llm.items()}
    for q, d in human.items():
        rel.setdefault(q, {}).update(d)
    return rel


def build(gold: dict, runs: list, rel_all: dict, faith: dict):
    C = Coll()
    lat, ncalls, drop = [], [], []
    finals = defaultdict(list)     # qid -> [final_ids per rep]
    specs = defaultdict(list)
    for r in runs:
        g = gold.get(r["qid"])
        if not g:
            continue
        qid, gs, loose = r["qid"], g.get("gold_spec", {}), g.get("loose", False)
        C.types[qid] = g.get("type", "?")
        C.add("run/error", qid, bool(r.get("error")))
        if r.get("error"):
            continue                       # hatalı koşular diğer metriklerden hariç, hata oranında görünür
        spec = r.get("spec") or {}
        sc = eval_spec(gs, spec, loose)
        for f, v in sc.items():
            C.add(f"spec/{f}", qid, v)
        C.add("spec/exact", qid, spec_exact(sc))
        if not loose:
            C.add("spec/spurious_numeric", qid, any(
                f not in gs and spec.get(f) not in (None, 0, 0.0) for f in NUM_FIELDS))
        if gs.get("mood_null") is True:
            C.add("spec/mood_hallucinated", qid, spec.get("mood") not in (None, ""))
        if r.get("rep", 0) == 0:
            lat.append(r["latency_s"])
            ncalls.append(r.get("n_tool_calls", 0))
        if r.get("n_merged"):
            drop.append(1 - r["n_filtered"] / r["n_merged"])

        ids, cards = r.get("final_ids", []), r.get("cards", {})
        finals[qid].append(ids)
        specs[qid].append({k: spec.get(k) for k in (
            "intent", "include_genres", "exclude_genres", "reference_movie", "director", "actor",
            "min_rating", "max_rating", "year_from", "year_to", "obscure")})
        final_cards = [cards[i] for i in ids if i in cards]
        intent = gs.get("intent", "recommendation")

        if g.get("forbid_titles"):
            bad = {norm_text(t) for t in g["forbid_titles"]}
            C.add("robust/forbidden_returned", qid, any(norm_text(c.get("Film")) in bad for c in final_cards))

        if intent == "general":
            C.add("general/no_movies", qid, not ids)
            continue
        if intent == "movie_info":
            first = final_cards[0] if final_cards else None
            C.add("info/found", qid, bool(first))
            if g.get("expect_title"):
                C.add("info/title_match", qid, bool(first) and norm_text(first.get("Film")) in _alts(g["expect_title"]))
            if g.get("expect_year"):
                C.add("info/year_match", qid, bool(first) and str(first.get("Yıl")) == str(g["expect_year"]))
            C.add("info/has_director_cast", qid, bool(first) and bool(first.get("Director")) and bool(first.get("Cast")))
            continue

        # ---- öneri sorguları
        if g.get("expect_empty"):
            C.add("empty/correct", qid, not ids)       # cevap olmayan soruda boş dönmeli
            continue
        C.add("empty/false_empty", qid, not ids)
        rel = rel_all.get(qid, {})
        vlists = {"final": ids, **(r.get("variants") or {})}
        for vname, vids in vlists.items():
            vc = [cards[i] for i in vids if i in cards]
            if has_hard_constraints(gs) and vc:
                chk = check_constraints(gs, vc)
                C.add(f"{vname}/hard_ok", qid, all(not x["viol"] for x in chk))
                C.add(f"{vname}/viol_rate", qid, sum(bool(x["viol"]) for x in chk) / len(chk))
            if gs.get("include_genres") and vc:
                C.add(f"{vname}/include_all", qid, 1 - sum(x["soft_miss"] for x in check_constraints(gs, vc)) / len(vc))
            if rel:
                C.add(f"{vname}/ndcg@5", qid, ndcg_at_k(vids, rel))
                C.add(f"{vname}/p@5", qid, precision_at_k(vids, rel, 5, 1))
                C.add(f"{vname}/strict_p@5", qid, precision_at_k(vids, rel, 5, 2))
                C.add(f"{vname}/label_coverage", qid, label_coverage(vids, rel))
        cands = r.get("candidates_top15") or []
        if rel and cands:
            C.add("cand/ceiling_ndcg@5", qid, oracle_ndcg(cands, rel))
            C.add("cand/any_relevant", qid, any(rel.get(i, 0) >= 1 for i in cands))
            C.add("cand/any_great", qid, any(rel.get(i, 0) == 2 for i in cands))

    # tutarlılık (tekrarlar arası)
    for qid, lists in finals.items():
        if len(lists) > 1:
            pairs = [(i, j) for i in range(len(lists)) for j in range(i + 1, len(lists))]
            C.add("stability/final_jaccard", qid, statistics.fmean(jaccard(lists[i], lists[j]) for i, j in pairs))
            sp = specs[qid]
            C.add("stability/spec_equal", qid, statistics.fmean(float(sp[i] == sp[j]) for i, j in pairs))

    # sadakat
    by_q = defaultdict(list)
    for key, v in faith.items():
        by_q[key.split("|")[0]].append(float(v["supported"]))
    for qid, vals in by_q.items():
        if qid in gold:
            C.add("faith/supported", qid, statistics.fmean(vals))
    return C, lat, ncalls, drop


def table(C, title, rows, cols):
    """rows: [(etiket, metrik_öneki)], cols: [(başlık, metrik_soneki)]"""
    out = [f"### {title}", "", "| | " + " | ".join(c for c, _ in cols) + " |",
           "|---|" + "---|" * len(cols)]
    for label, pre in rows:
        out.append(f"| {label} | " + " | ".join(fmt(C.ci(f"{pre}{suf}")) for _, suf in cols) + " |")
    return "\n".join(out) + "\n"


def render(C, lat, ncalls, drop, n_runs) -> str:
    L = ["# MovieMCP eval raporu", "", f"Koşu sayısı: {n_runs}; sorgu sayısı: "
         f"{len({q for m in C.d.values() for q in m})}. Hücre biçimi: ortalama [%95 bootstrap GA] (n=sorgu).", ""]
    err = C.ci("run/error")
    L.append(f"Hata oranı (timeout/exception): {fmt(err)}\n")

    spec_cols = [(f, f"spec/{f}") for f in ("intent", "include_genres", "exclude_genres", "reference_movie",
                                            "movie_title", "director", "actor", "min_rating", "max_rating",
                                            "year_from", "year_to", "mood_null", "obscure", "exact")
                 if C.per_query(f"spec/{f}")]
    L += ["## 1. Sorgu çıkarımı (QuerySpec)", "",
          "| alan | skor |", "|---|---|"]
    for name, m in spec_cols:
        L.append(f"| {name} | {fmt(C.ci(m))} |")
    L.append(f"| **uydurulan sayısal kısıt oranı** (düşük iyi) | {fmt(C.ci('spec/spurious_numeric'))} |")
    L.append(f"| **uydurulan mood oranı** (düşük iyi) | {fmt(C.ci('spec/mood_hallucinated'))} |\n")

    L.append("## 2. Ablation: hangi bileşen ne katıyor?\n")
    L.append("`final` = tam sistem. `score_only` = LLM seçimi yok. `no_filter` = sert filtre + LLM yok. "
             "`semantic_only` = yalnızca vektör arama. nDCG/P için etiket (judge/insan) gerekir.\n")
    rows = [(v, f"{v}/") for v in VARIANTS]
    L.append(table(C, "Kalite (sıralama)", rows, [("nDCG@5", "ndcg@5"), ("P@5 (≥1)", "p@5"),
                                                   ("P@5 (=2)", "strict_p@5"), ("etiket kapsaması", "label_coverage")]))
    L.append(table(C, "Kısıt uyumu (sert kısıtlı sorgularda)", rows, [
        ("tüm filmler kısıta uyuyor", "hard_ok"), ("ihlalli film oranı (düşük iyi)", "viol_rate"),
        ("istenen türlerin tamamı", "include_all")]))

    L += ["## 3. Retrieval tavanı vs seçim", "",
          f"- Aday havuzunun (ilk 15) en iyi 5'iyle ulaşılabilecek nDCG@5 (tavan): {fmt(C.ci('cand/ceiling_ndcg@5'))}",
          f"- Gerçek `final` nDCG@5: {fmt(C.ci('final/ndcg@5'))}",
          f"- Havuzda en az 1 ilgili film olan sorgular: {fmt(C.ci('cand/any_relevant'))}; "
          f"en az 1 'çok uygun' (=2): {fmt(C.ci('cand/any_great'))}",
          "- Yorum: tavan düşükse sorun retrieval/filtrede (QuerySpec, plan, semantik arama); tavan yüksek ama "
          "final düşükse sorun LLM seçim aşamasında.",
          f"- Sert filtre ortalama aday eleme oranı: "
          f"{(statistics.fmean(drop) if drop else float('nan')):.2%}\n"]

    L += ["## 4. Dürüstlük, sadakat, dayanıklılık", "",
          "| metrik | skor |", "|---|---|",
          f"| Cevabı olmayan sorularda doğru şekilde boş dönme (yüksek iyi) | {fmt(C.ci('empty/correct'))} |",
          f"| Cevabı olan sorularda yanlışlıkla boş dönme (düşük iyi) | {fmt(C.ci('empty/false_empty'))} |",
          f"| Gerekçelerin karttan desteklenme oranı (judge) | {fmt(C.ci('faith/supported'))} |",
          f"| Persona enjeksiyonuna kanma (düşük iyi) | {fmt(C.ci('robust/forbidden_returned'))} |",
          f"| Film bilgisi: bulundu | {fmt(C.ci('info/found'))} |",
          f"| Film bilgisi: doğru film | {fmt(C.ci('info/title_match'))} |",
          f"| Film bilgisi: doğru yıl (aynı adlı film ayrımı) | {fmt(C.ci('info/year_match'))} |",
          f"| Film bilgisi: yönetmen+oyuncu dolu | {fmt(C.ci('info/has_director_cast'))} |",
          f"| Sohbette film listesi dönmedi | {fmt(C.ci('general/no_movies'))} |\n"]

    L += ["## 5. Tutarlılık (aynı soru, birden fazla koşu)", "",
          f"- Sonuç kümesi Jaccard: {fmt(C.ci('stability/final_jaccard'))}",
          f"- Aynı QuerySpec üretme oranı: {fmt(C.ci('stability/spec_equal'))}\n"]

    L += ["## 6. Gecikme / maliyet (yalnızca ilk tekrar; cache ısındıktan sonrakiler hariç)", "",
          f"- p50: {percentile(lat, .5):.1f} sn, p95: {percentile(lat, .95):.1f} sn" if lat else "- veri yok",
          f"- Ortalama araç çağrısı/sorgu: {statistics.fmean(ncalls):.1f}" if ncalls else "", ""]

    L += ["## 7. Sorgu tipine göre kırılım", "", "| tip | n | spec exact | final nDCG@5 | final hard_ok |", "|---|---|---|---|---|"]
    types = defaultdict(set)
    for q, t in C.types.items():
        types[t].add(q)
    for t, qs in sorted(types.items()):
        def cell(m):
            v = C.per_query(m, qs)
            return f"{statistics.fmean(v):.2f} (n={len(v)})" if v else "—"
        L.append(f"| {t} | {len(qs)} | {cell('spec/exact')} | {cell('final/ndcg@5')} | {cell('final/hard_ok')} |")
    return "\n".join(x for x in L if x is not None) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default=str(DATA / "dev.jsonl"))
    ap.add_argument("--labels", default=str(DATA / "labels_dev.json"))
    ap.add_argument("--human", default=str(DATA / "human_sample_dev.csv"))
    ap.add_argument("--faith", default=str(DATA / "faith_dev.json"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    gold = {g["qid"]: g for g in load_jsonl(a.data)}
    runs = [r for p in a.runs for r in load_jsonl(p)]
    rel = merge_labels(read_json(a.labels, {}), load_human(a.human))
    faith = read_json(a.faith, {})
    if not rel:
        print("UYARI: etiket yok -> nDCG/P@5 hesaplanamaz. Önce `judge.py pool` çalıştır.", file=sys.stderr)
    C, lat, ncalls, drop = build(gold, runs, rel, faith)
    text = render(C, lat, ncalls, drop, len(runs))
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print("yazıldı:", a.out, file=sys.stderr)


if __name__ == "__main__":
    main()
