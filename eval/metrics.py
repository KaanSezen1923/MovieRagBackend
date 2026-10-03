"""Saf metrik fonksiyonları (LLM/ağ yok, birim test edilebilir)."""
import math
import random

from common import norm_text

SET_FIELDS = ("include_genres", "exclude_genres")
STR_FIELDS = ("reference_movie", "movie_title", "director", "actor")
NUM_FIELDS = ("min_rating", "max_rating", "year_from", "year_to")


# ----------------------------------------------------------------- QuerySpec
def f1_sets(g: set, p: set) -> float:
    if not g and not p:
        return 1.0
    if not g or not p:
        return 0.0
    tp = len(g & p)
    if tp == 0:
        return 0.0
    pr, rc = tp / len(p), tp / len(g)
    return 2 * pr * rc / (pr + rc)


def _num_eq(pred, exp) -> bool:
    if exp is None:
        return pred is None
    if pred is None:
        return False
    return abs(float(pred) - float(exp)) < 1e-6


def _alts(v) -> set:
    """'The Dark Knight|Dark Knight' -> kabul edilen alternatifler."""
    return {norm_text(x) for x in str(v).split("|")}


def eval_spec(gold: dict, pred: dict, loose: bool = False) -> dict:
    """
    gold'da yazan alanlar kontrol edilir (kısmi etiket). İstisna: sayısal alanlar
    (min/max_rating, year_from/to) gold'da YOKSA 'None olmalı' beklenir (LLM'in sayı uydurmasını yakalar);
    belirsiz sorgular için gold'a "loose": true koy.
    Dönüş: alan -> 0..1
    """
    pred = pred or {}
    res = {}
    if "intent" in gold:
        res["intent"] = float(norm_text(pred.get("intent")) == norm_text(gold["intent"]))
    for f in SET_FIELDS:
        if f in gold:
            g = {norm_text(x) for x in gold[f]}
            p = {norm_text(x) for x in (pred.get(f) or [])}
            res[f] = f1_sets(g, p)
    for f in STR_FIELDS:
        if f in gold:
            if gold[f] is None:
                res[f] = float(not pred.get(f))
            else:
                res[f] = float(norm_text(pred.get(f)) in _alts(gold[f]))
    for f in NUM_FIELDS:
        if f in gold:
            res[f] = float(_num_eq(pred.get(f), gold[f]))
        elif not loose:
            res[f] = float(pred.get(f) in (None, 0, 0.0))   # beklenen: sınır yok
    if "mood_null" in gold:
        res["mood_null"] = float((pred.get("mood") in (None, "")) == bool(gold["mood_null"]))
    if "obscure" in gold:
        res["obscure"] = float(bool(pred.get("obscure")) == bool(gold["obscure"]))
    return res


def spec_exact(scores: dict) -> float:
    return float(all(v == 1.0 for v in scores.values())) if scores else 0.0


# ------------------------------------------------------------- kısıt kontrolü
def _rating(c):
    try:
        return float(c.get("TMDB Puanı"))
    except (TypeError, ValueError):
        return None


def _genres(c) -> set:
    return {norm_text(g) for g in (c.get("Türler") or "").split(",") if g.strip()}


def has_hard_constraints(gold_spec: dict) -> bool:
    return bool(gold_spec.get("exclude_genres")) or any(
        gold_spec.get(k) is not None for k in NUM_FIELDS)


def check_constraints(gold_spec: dict, cards: list) -> list:
    """
    Her kart için {'id', 'viol': [...], 'soft_miss': bool}.
    SERT: exclude_genres, min/max_rating, year_from/to. YUMUŞAK: include_genres'in tamamı.
    """
    ex = {norm_text(g) for g in (gold_spec.get("exclude_genres") or [])}
    inc = {norm_text(g) for g in (gold_spec.get("include_genres") or [])}
    mn, mx = gold_spec.get("min_rating"), gold_spec.get("max_rating")
    yf, yt = gold_spec.get("year_from"), gold_spec.get("year_to")
    out = []
    for c in cards:
        viol, g, r = [], _genres(c), _rating(c)
        y = str(c.get("Yıl") or "")
        y = int(y) if y.isdigit() else None
        if ex & g:
            viol.append("exclude_genre")
        if mn is not None and (r is None or r < mn):
            viol.append("min_rating")
        if mx is not None and (r is None or r > mx):
            viol.append("max_rating")
        if yf and (y is None or y < yf):
            viol.append("year_from")
        if yt and (y is None or y > yt):
            viol.append("year_to")
        out.append({"id": str(c.get("movie_id")), "viol": viol,
                    "soft_miss": bool(inc) and not inc <= g})
    return out


# ------------------------------------------------------------ sıralama metrikleri
def dcg(gains: list) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranked_ids: list, rel: dict, k: int = 5):
    """rel: {id: 0/1/2}. Etiketli havuzda hiç ilgili film yoksa None."""
    ideal = sorted(rel.values(), reverse=True)[:k]
    idcg = dcg(ideal)
    if idcg == 0:
        return None
    return dcg([rel.get(i, 0) for i in ranked_ids[:k]]) / idcg


def precision_at_k(ranked_ids: list, rel: dict, k: int = 5, min_rel: int = 1):
    top = ranked_ids[:k]
    if not top:
        return 0.0
    return sum(rel.get(i, 0) >= min_rel for i in top) / len(top)


def label_coverage(ids: list, rel: dict):
    return None if not ids else sum(i in rel for i in ids) / len(ids)


def oracle_ndcg(candidate_ids: list, rel: dict, k: int = 5):
    """Aday havuzundan EN İYİ k filmi seçebilseydik elde edilecek nDCG (retrieval tavanı)."""
    best = sorted(candidate_ids, key=lambda i: -rel.get(i, 0))
    return ndcg_at_k(best, rel, k)


# ------------------------------------------------------------------ istatistik
def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def bootstrap_ci(values: list, n: int = 2000, seed: int = 0, alpha: float = 0.05):
    """Sorgular üzerinden yüzdelik bootstrap. -> (ortalama, alt, üst, n)"""
    vals = [v for v in values if v is not None]
    if not vals:
        return (float("nan"),) * 3 + (0,)
    mean = sum(vals) / len(vals)
    if len(vals) == 1:
        return mean, mean, mean, 1
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(vals, k=len(vals))) / len(vals) for _ in range(n))
    lo = means[int(alpha / 2 * n)]
    hi = means[min(int((1 - alpha / 2) * n), n - 1)]
    return mean, lo, hi, len(vals)


def cohen_kappa(a: list, b: list) -> float:
    assert len(a) == len(b) and a, "boş ya da eşit uzunlukta olmayan etiket listeleri"
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    pe = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def percentile(values: list, q: float):
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return float("nan")
    k = (len(vals) - 1) * q
    f, c = math.floor(k), math.ceil(k)
    return vals[f] if f == c else vals[f] + (vals[c] - vals[f]) * (k - f)
