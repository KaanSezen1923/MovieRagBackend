"""Eval metriklerinin kendi doğruluğu (metriği ölçen kod da test edilmeli)."""
import math

import pytest

from metrics import (bootstrap_ci, check_constraints, cohen_kappa, eval_spec, f1_sets, has_hard_constraints,
                     jaccard, label_coverage, ndcg_at_k, oracle_ndcg, percentile, precision_at_k, spec_exact)


class TestSpec:
    def test_perfect(self):
        gold = {"intent": "recommendation", "include_genres": ["Comedy"], "exclude_genres": ["Romance"], "director": "Nolan"}
        pred = {"intent": "recommendation", "include_genres": ["comedy"], "exclude_genres": ["Romance"],
                "director": "nolan", "min_rating": None, "max_rating": None, "year_from": None, "year_to": None}
        sc = eval_spec(gold, pred)
        assert spec_exact(sc) == 1.0

    def test_genre_f1_partial(self):
        sc = eval_spec({"include_genres": ["Comedy", "Drama"]}, {"include_genres": ["Comedy"]}, loose=True)
        assert sc["include_genres"] == pytest.approx(2 * 1 * 0.5 / 1.5)

    def test_unlisted_numeric_must_be_none_unless_loose(self):
        assert eval_spec({}, {"min_rating": 7.0})["min_rating"] == 0.0
        assert "min_rating" not in eval_spec({}, {"min_rating": 7.0}, loose=True)

    def test_explicit_numeric(self):
        assert eval_spec({"year_from": 1920}, {"year_from": 1920})["year_from"] == 1.0
        assert eval_spec({"year_from": 1920}, {"year_from": None})["year_from"] == 0.0

    def test_alternatives_and_accents(self):
        sc = eval_spec({"reference_movie": "Amelie|Le Fabuleux Destin d'Amélie Poulain"}, {"reference_movie": "AMÉLIE"}, loose=True)
        assert sc["reference_movie"] == 1.0

    def test_mood_null(self):
        assert eval_spec({"mood_null": True}, {"mood": "üzgün"}, loose=True)["mood_null"] == 0.0
        assert eval_spec({"mood_null": False}, {"mood": "üzgün"}, loose=True)["mood_null"] == 1.0

    def test_f1_both_empty(self):
        assert f1_sets(set(), set()) == 1.0 and f1_sets({"a"}, set()) == 0.0


class TestConstraints:
    def c(self, mid, genres="Drama", rating="7.0", year="2000"):
        return {"movie_id": mid, "Türler": genres, "TMDB Puanı": rating, "Yıl": year}

    def test_violations_detected(self):
        gold = {"exclude_genres": ["Horror"], "min_rating": 7, "max_rating": 9, "year_from": 1990, "year_to": 1999}
        res = check_constraints(gold, [self.c("1", "Horror", "6", "2005")])
        assert set(res[0]["viol"]) == {"exclude_genre", "min_rating", "year_to"}

    def test_clean_card(self):
        gold = {"exclude_genres": ["Horror"], "min_rating": 7}
        assert check_constraints(gold, [self.c("1")])[0]["viol"] == []

    def test_soft_include_requires_all_genres(self):
        gold = {"include_genres": ["Comedy", "Drama"]}
        assert check_constraints(gold, [self.c("1", "Comedy")])[0]["soft_miss"] is True
        assert check_constraints(gold, [self.c("1", "Comedy, Drama")])[0]["soft_miss"] is False

    def test_missing_rating_violates_min(self):
        assert "min_rating" in check_constraints({"min_rating": 5}, [self.c("1", rating="")])[0]["viol"]

    def test_has_hard_constraints(self):
        assert not has_hard_constraints({"include_genres": ["Comedy"]})
        assert has_hard_constraints({"year_from": 1990})


class TestRanking:
    REL = {"a": 2, "b": 1, "c": 0, "d": 2}

    def test_ndcg_perfect_and_worse(self):
        assert ndcg_at_k(["a", "d", "b"], self.REL) == pytest.approx(1.0)
        assert ndcg_at_k(["c", "b", "a"], self.REL) < 1.0

    def test_ndcg_none_when_no_relevant(self):
        assert ndcg_at_k(["x"], {"x": 0}) is None

    def test_unjudged_counts_as_zero(self):
        assert ndcg_at_k(["zzz"], self.REL) == 0.0

    def test_precision(self):
        assert precision_at_k(["a", "c", "b", "x"], self.REL, 4) == pytest.approx(0.5)
        assert precision_at_k(["a", "c", "b", "x"], self.REL, 4, min_rel=2) == pytest.approx(0.25)
        assert precision_at_k([], self.REL) == 0.0

    def test_oracle_is_upper_bound(self):
        cands = ["c", "b", "a", "d"]
        assert oracle_ndcg(cands, self.REL) >= ndcg_at_k(cands, self.REL)
        assert oracle_ndcg(cands, self.REL) == pytest.approx(1.0)

    def test_coverage(self):
        assert label_coverage(["a", "x"], self.REL) == 0.5 and label_coverage([], self.REL) is None


class TestStats:
    def test_jaccard(self):
        assert jaccard(["a", "b"], ["b", "c"]) == pytest.approx(1 / 3)
        assert jaccard([], []) == 1.0

    def test_bootstrap_ci_brackets_mean(self):
        m, lo, hi, n = bootstrap_ci([0, 1, 1, 1, 0, 1, 1, 0, 1, 1])
        assert lo <= m <= hi and n == 10

    def test_bootstrap_empty(self):
        assert math.isnan(bootstrap_ci([])[0])

    def test_kappa(self):
        assert cohen_kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1.0)
        assert cohen_kappa([0, 0, 1, 1], [1, 1, 0, 0]) == pytest.approx(-1.0)

    def test_percentile(self):
        assert percentile([1, 2, 3, 4, 5], 0.5) == 3 and percentile([1, 2], 0.95) == pytest.approx(1.95)
