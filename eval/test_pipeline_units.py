"""
client.py içindeki DETERMİNİSTİK mantığın birim testleri (LLM/TMDB/Neo4j gerekmez).
Çalıştır:  pytest eval/test_pipeline_units.py -q
"""
import json

import pytest

import client
from client import (QuerySpec, _clean_spec, _parse_json, _strip_internal, bayes, build_plan, hard_filter,
                    merge_candidates, parse_selection, parse_spec, route_by_intent, score_candidates)


def card(mid="1", genres="Drama", rating="7.0", votes=500, year="2000", overview="bir özet", **kw):
    c = {"movie_id": str(mid), "Film": f"Film{mid}", "Türler": genres, "TMDB Puanı": rating,
         "vote_count": votes, "Yıl": year, "Özet": overview, "source": "t"}
    c.update(kw)
    return c


def ids(cards):
    return [c["movie_id"] for c in cards]


# ------------------------------------------------------------ _clean_spec / parse_spec
class TestCleanSpec:
    def test_nullish_strings_become_none(self):
        d = _clean_spec({"mood": "null", "director": "Yok", "actor": " N/A "})
        assert d["mood"] is None and d["director"] is None and d["actor"] is None

    def test_genres_canonicalized_tr_en(self):
        d = _clean_spec({"include_genres": ["korku", "Sci-Fi", "bilim kurgu", "nonsense"],
                         "exclude_genres": "Aksiyon, komedi"})
        assert d["include_genres"] == ["Horror", "Science Fiction"]      # tekrarsız, geçersiz atıldı
        assert d["exclude_genres"] == ["Action", "Comedy"]

    def test_invalid_intent_falls_back_to_recommendation(self):
        assert _clean_spec({"intent": "banana"})["intent"] == "recommendation"
        assert _clean_spec({})["intent"] == "recommendation"
        assert _clean_spec({"intent": "GENERAL"})["intent"] == "general"

    @pytest.mark.parametrize("val,exp", [(True, True), ("true", True), ("false", False), (False, False), (None, False)])
    def test_obscure_flag(self, val, exp):
        assert _clean_spec({"obscure": val})["obscure"] is exp

    def test_none_input(self):
        assert _clean_spec(None)["intent"] == "recommendation"


class TestParseSpec:
    def test_fenced_json(self):
        s = parse_spec('```json\n{"intent": "general"}\n```')
        assert s.intent == "general"

    def test_json_with_leading_text(self):
        s = parse_spec('Tabii! {"intent": "movie_info", "movie_title": "Heat"} umarım yardımcı olur')
        assert s.intent == "movie_info" and s.movie_title == "Heat"

    def test_garbage_returns_default_spec(self):
        s = parse_spec("bu hiç JSON değil")
        assert s == QuerySpec()

    def test_wrong_types_fall_back(self):
        # min_rating sayıya çevrilemiyor -> doğrulama hatası -> güvenli varsayılan
        assert parse_spec('{"min_rating": "çok yüksek"}') == QuerySpec()

    def test_parse_json_handles_nested(self):
        assert _parse_json('x {"a": {"b": 1}} y') == {"a": {"b": 1}}


# ------------------------------------------------------------------ hard_filter
class TestHardFilter:
    def test_excluded_genre_removed(self):
        spec = QuerySpec(exclude_genres=["Horror"])
        out = hard_filter([card("1", "Horror, Thriller"), card("2", "Comedy")], spec)
        assert ids(out) == ["2"]

    def test_exclude_is_case_insensitive(self):
        spec = QuerySpec(exclude_genres=["Horror"])
        assert hard_filter([card("1", "horror")], spec) == []

    def test_rating_bounds(self):
        spec = QuerySpec(min_rating=6.0, max_rating=8.0)
        cs = [card("1", rating="5.9"), card("2", rating="6.0"), card("3", rating="8.0"), card("4", rating="8.1")]
        assert ids(hard_filter(cs, spec)) == ["2", "3"]

    def test_missing_rating_dropped_only_when_rating_filter_active(self):
        c = card("1", rating="")
        assert hard_filter([c], QuerySpec(min_rating=5)) == []
        assert ids(hard_filter([c], QuerySpec())) == ["1"]

    def test_year_range(self):
        spec = QuerySpec(year_from=1990, year_to=1999)
        cs = [card("1", year="1989"), card("2", year="1990"), card("3", year="1999"), card("4", year="2000")]
        assert ids(hard_filter(cs, spec)) == ["2", "3"]

    def test_vote_threshold_normal_vs_obscure(self):
        c = card("1", votes=50)
        assert hard_filter([c], QuerySpec()) == []                                  # eşik 100
        assert ids(hard_filter([c], QuerySpec(obscure=True))) == ["1"]              # eşik 20
        assert hard_filter([card("2", votes=10)], QuerySpec(obscure=True)) == []

    def test_empty_overview_dropped(self):
        assert hard_filter([card("1", overview="  ")], QuerySpec()) == []


# ------------------------------------------------------------ merge / score
class TestMergeAndScore:
    def test_rrf_scores_and_sources(self):
        a = [card("1", source="a"), card("2", source="a")]
        b = [card("2", source="b"), card("3", source="b")]
        m = {c["movie_id"]: c for c in merge_candidates([a, b])}
        assert set(m) == {"1", "2", "3"}
        assert m["2"]["_rrf"] == pytest.approx(1 / 61 + 1 / 60)
        assert m["2"]["_sources"] == ["a", "b"]
        assert m["1"]["_rrf"] == pytest.approx(1 / 60)

    def test_missing_fields_backfilled_and_max_votes(self):
        a = [card("1", overview="", votes=10)]
        b = [card("1", overview="dolu özet", votes=900)]
        m = merge_candidates([a, b])[0]
        assert m["Özet"] == "dolu özet" and m["vote_count"] == 900

    def test_cards_without_id_skipped(self):
        assert merge_candidates([[{"Film": "x"}]]) == []

    def test_bayes_penalizes_few_votes(self):
        assert bayes(10.0, 3) < bayes(8.0, 5000)
        assert bayes("x", 5) == 0.0

    def test_multi_source_beats_single_source(self):
        a = [card("1"), card("2")]
        b = [card("2")]
        ranked = score_candidates(merge_candidates([a, b]), QuerySpec())
        assert ids(ranked)[0] == "2"

    def test_genre_fit_breaks_ties(self):
        # sıra farkını yok etmek için ikisini ayrı listelerde, aynı sırada ver
        cs = merge_candidates([[card("1", genres="Drama")], [card("2", genres="Comedy")]])
        ranked = score_candidates(cs, QuerySpec(include_genres=["Comedy"]))
        assert ids(ranked)[0] == "2"


# ---------------------------------------------------------------- build_plan
class TestBuildPlan:
    def names(self, plan):
        return [n for n, _ in plan]

    def test_plain_query_is_semantic_only(self):
        plan = build_plan(QuerySpec(), "bir şey öner")
        assert self.names(plan) == ["search_movies_semantically"]
        assert plan[0][1]["semantic_query"] == "bir şey öner"

    def test_reference_movie_adds_similar_tool_first(self):
        plan = build_plan(QuerySpec(reference_movie="Inception"), "x")
        assert plan[0] == ("get_similar_movies", {"movie_title": "Inception"})
        assert "search_movies_semantically" in self.names(plan)

    def test_director_only_has_graph_and_filter_but_no_semantic(self):
        plan = build_plan(QuerySpec(director="Christopher Nolan"), "x")
        assert self.names(plan) == ["search_movies_in_graph", "search_movies_by_filters"]
        assert plan[0][1]["category"] == "Director"
        assert plan[1][1]["director_name"] == "Christopher Nolan"

    def test_director_with_semantic_query_adds_semantic(self):
        plan = build_plan(QuerySpec(director="X", semantic_query_en="mind-bending heist"), "x")
        assert "search_movies_semantically" in self.names(plan)

    def test_exclude_only_builds_filter_args_without_default_noise(self):
        plan = dict((n, a) for n, a in build_plan(QuerySpec(exclude_genres=["Horror"]), "x"))
        f = plan["search_movies_by_filters"]
        assert f["exclude_genres"] == "Horror" and f["max_rating"] == 10.0
        assert "min_rating" not in f and "year_from" not in f
        assert plan["search_movies_semantically"]["exclude_genres"] == "Horror"

    def test_numeric_constraints_forwarded(self):
        spec = QuerySpec(min_rating=8.0, year_from=2000, year_to=2009, include_genres=["Drama", "War"])
        plan = dict(build_plan(spec, "x"))
        f = plan["search_movies_by_filters"]
        assert f["min_rating"] == 8.0 and f["year_from"] == 2000 and f["year_to"] == 2009
        assert f["genre_name"] == "Drama,War"

    def test_obscure_lowers_vote_threshold(self):
        plan = dict(build_plan(QuerySpec(obscure=True, include_genres=["Horror"]), "x"))
        assert plan["search_movies_by_filters"]["min_votes"] == client.MIN_VOTES_OBSCURE
        assert plan["search_movies_semantically"]["min_votes"] == client.MIN_VOTES_OBSCURE

    def test_no_none_values_in_tool_args(self):
        for spec in (QuerySpec(), QuerySpec(director="A", actor="B", exclude_genres=["Action"], reference_movie="C")):
            for _, args in build_plan(spec, "x"):
                assert all(v is not None for v in args.values())


# ----------------------------------------------------------- parse_selection
class TestParseSelection:
    VALID = {"1", "2", "3", "4"}

    def sel(self, ranked, **extra):
        return json.dumps({"text": "giriş", "ranked": ranked, **extra})

    def test_happy_path_keeps_llm_order(self):
        text, mood, no_exact, picks = parse_selection(
            self.sel([{"id": "3", "reason": "a"}, {"id": "1", "reason": "b"}]), self.VALID, 5)
        assert [p[0] for p in picks] == ["3", "1"] and text == "giriş" and mood == "" and not no_exact

    def test_hallucinated_and_duplicate_ids_dropped(self):
        _, _, _, picks = parse_selection(
            self.sel([{"id": "99", "reason": "x"}, {"id": "2", "reason": "a"}, {"id": "2", "reason": "dup"}]),
            self.VALID, 5)
        assert [p[0] for p in picks] == ["2"]

    def test_limit_enforced(self):
        ranked = [{"id": str(i), "reason": ""} for i in (1, 2, 3, 4)]
        _, _, _, picks = parse_selection(self.sel(ranked), self.VALID, 2)
        assert len(picks) == 2

    def test_id_prefix_tolerated(self):
        _, _, _, picks = parse_selection(self.sel([{"id": "ID:4", "reason": "r"}]), self.VALID, 5)
        assert picks == [("4", "r")]

    def test_dict_format_tolerated(self):
        _, _, _, picks = parse_selection(self.sel({"1": "neden"}), self.VALID, 5)
        assert picks == [("1", "neden")]

    def test_empty_ranked_is_valid_honest_empty(self):
        out = parse_selection(self.sel([], no_exact_match=True), self.VALID, 5)
        assert out is not None and out[3] == [] and out[2] is True

    @pytest.mark.parametrize("raw", ["json değil", '{"text": "x"}', '{"ranked": "yanlış tip"}'])
    def test_unparseable_returns_none(self, raw):
        assert parse_selection(raw, self.VALID, 5) is None


# --------------------------------------------------------------- yönlendirme
class TestRouting:
    def test_general(self):
        assert route_by_intent({"intent": "general"}) == "general_chatter"

    def test_movie_info_needs_title(self):
        assert route_by_intent({"intent": "movie_info", "intent_data": {}}) == "recommendation_engine"
        assert route_by_intent({"intent": "movie_info", "intent_data": {"movie_title": "Heat"}}) == "movie_info"

    def test_default_is_recommendation(self):
        assert route_by_intent({"intent": "recommendation"}) == "recommendation_engine"
        assert route_by_intent({}) == "recommendation_engine"


def test_strip_internal_removes_private_and_debug_keys():
    c = {"Film": "x", "_score": 1, "_rrf": 2, "source": "s", "score": 3, "relaxed": ["k"], "Yıl": "2000"}
    assert _strip_internal(c) == {"Film": "x", "Yıl": "2000"}
