"""
LangGraph istemcisi: niyet analizi -> (öneri hattı | film bilgisi | sohbet).

MİMARİ DEĞİŞİKLİK — "LLM araç çağırır" yerine "kod planlar, LLM sadece anlar ve seçer":

  ESKİ:  intent LLM -> reasoning LLM (aynı iş, ikinci kez) -> ajan döngüsü (en çok 3 LLM turu,
         araçlar SIRAYLA) -> her adayı TMDB'de zenginleştirme -> writer LLM
         => 4-6 seri LLM çağrısı + ~15 TMDB isteği (ort. 8.9 sn)
  YENİ:  1 LLM (yapısal çıkarım) -> kod: paralel retrieval -> kod: sert filtre + skor
         -> 1 LLM (sıralı seçim + gerekçe) -> sadece seçilen 5 film için zenginleştirme

Düzeltilen hatalar:
  * GraphState'te intent_data yoktu; LangGraph bilinmeyen anahtarı sessizce düşürdüğü için
    recommendation_node çıkarımı ikinci kez yapmak zorunda kalıyordu. Artık shared.GraphState kullanılır.
  * Kartlar LLM'in sıralamasıyla değil, araçların dönüş sırasıyla dizilmişti
    (Memento 3., Godzilla Resurgence son sıradaydı). Artık LLM'in sırası korunur.
  * 'avoid', 'min_rating' sadece prompt metnindeydi; LLM'in uymasına bırakılmıştı.
    Artık exclude_genres / min_rating / max_rating / yıl KODDA uygulanır (Q12, Q18, Q36).
  * mood şemada 'null olabilir' denmediği için LLM her sorguya duygu uyduruyordu
    ("Kasvetin içinde kaybolmuş hissettiğin..."). Artık yalnızca açıkça söylenmişse doldurulur.
  * Metin ve liste aynı LLM çağrısından gelir; 'uygun olmayanları eklemedim' deyip 5 film
    döndürme çelişkisi (Q45) ortadan kalkar. Hiç uygun aday yoksa boş liste + dürüst mesaj.
"""
import asyncio
import json
import os
import random
import re
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError
from logger import logger
from shared import GraphState, chat_statuses, ctx   # TEK GraphState tanımı (intent_data dahil)

try:
    from reranker import rerank as _rerank
except ImportError:
    _rerank = None

load_dotenv(override=True)   # .env, sistem ortam değişkenini ezsin

# --- Ollama bağlantısı -------------------------------------------------------
# gemma4:31b-cloud, yerel Ollama üzerinden (ollama signin yapılmış) bulutta çalışır.


MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
OLLAMA_HOST = (os.getenv("OLLAMA_HOST") or "https://ollama.com").rstrip("/")
_ollama_key = os.getenv("OLLAMA_API_KEY")
# trust_env=False: sistemdeki HTTP(S)_PROXY / ALL_PROXY değişkenleri localhost isteğini
# bozmasın ("All connection attempts failed" hatasının sık nedeni).
_client_kwargs = {"trust_env": False}
if _ollama_key:
    _client_kwargs["headers"] = {"Authorization": f"Bearer {_ollama_key}"}
aclient = ChatOllama(model=MODEL_NAME, temperature=0.7, base_url=OLLAMA_HOST, client_kwargs=_client_kwargs)
logger.info(f"Ollama -> {OLLAMA_HOST} | model={MODEL_NAME}")

# ---- ayarlar --------------------------------------------------------------
MAX_MOVIES = 5              # sonuç sayısı
MAX_MOVIES_PEOPLE = 8       # yönetmen/oyuncu filmografisi sorgularında
CANDIDATES_TO_LLM = 15      # seçim aşamasına giden aday sayısı
MIN_VOTES = 100             # normal sorgularda oy eşiği
MIN_VOTES_OBSCURE = 20      # "kült / az bilinen / bağımsız" sorgularında
TOOL_TIMEOUT = 25           # sn
RERANKER_BOOST = float(os.getenv("RERANKER_BOOST", "0.10"))   # reranker skor ağırlığı
RERANKER_TOP_N = int(os.getenv("RERANKER_TOP_N", "50"))        # reranker'a giden max aday
MMR_ENABLED = os.getenv("MMR_ENABLED", "true").lower() in ("true", "1", "yes")
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.7"))             # alaka (1.0) vs çeşitlilik (0.0) dengesi

TMDB_GENRES = ["Action", "Adventure", "Animation", "Comedy", "Crime", "Documentary", "Drama", "Family",
               "Fantasy", "History", "Horror", "Music", "Mystery", "Romance", "Science Fiction",
               "Thriller", "War", "Western"]
_GENRE_CANON = {g.lower(): g for g in TMDB_GENRES}
_GENRE_CANON.update({"sci-fi": "Science Fiction", "scifi": "Science Fiction", "sci fi": "Science Fiction",
                     "bilim kurgu": "Science Fiction", "aksiyon": "Action", "macera": "Adventure",
                     "animasyon": "Animation", "komedi": "Comedy", "suç": "Crime", "belgesel": "Documentary",
                     "dram": "Drama", "aile": "Family", "fantastik": "Fantasy", "tarih": "History",
                     "korku": "Horror", "müzik": "Music", "gizem": "Mystery", "romantik": "Romance",
                     "gerilim": "Thriller", "savaş": "War"})


def _to_lc_message(msg):
    if isinstance(msg, BaseMessage):
        return msg
    if isinstance(msg, dict):
        role, content = msg.get("role", "user"), msg.get("content", "")
        if role == "system":
            return SystemMessage(content=content)
        if role == "user":
            return HumanMessage(content=content)
        if role == "assistant":
            return AIMessage(content=content)
        if role == "tool":
            return ToolMessage(content=content,
                               tool_call_id=msg.get("tool_call_id", msg.get("name", "call_default")),
                               name=msg.get("name", "tool"))
    return HumanMessage(content=str(msg))


# ============================================================================
# 1) YAPISAL SORGU ÇIKARIMI (tek LLM çağrısı)
# ============================================================================
class QuerySpec(BaseModel):
    intent: str = "recommendation"            # general | movie_info | recommendation
    mood: Optional[str] = None                # SADECE kullanıcı duygusunu açıkça söylediyse
    include_genres: List[str] = Field(default_factory=list)
    exclude_genres: List[str] = Field(default_factory=list)
    reference_movie: Optional[str] = None     # "X'e benzer" denen film (orijinal adı)
    movie_title: Optional[str] = None         # belirli bir film hakkında soru (oyuncular, yönetmen...)
    director: Optional[str] = None
    actor: Optional[str] = None
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    semantic_query_en: Optional[str] = None   # tema/atmosfer, İngilizce, sayısal kısıtsız
    must_have: List[str] = Field(default_factory=list)   # kullanıcının sert kriterleri (seçimde kontrol edilir)
    obscure: bool = False                     # kült / az bilinen / düşük bütçe / düşük puanlı istiyor


EXTRACT_SYSTEM = """Sen bir film botunun sorgu analiz bileşenisin. Kullanıcı mesajından SADECE aşağıdaki JSON'u üret.

intent:
- "general": selam, sohbet, filmle ilgisiz.
- "movie_info": belirli BİR filmin detayı soruluyor (oyuncular, yönetmen, puan...). movie_title'ı doldur.
- "recommendation": film önerisi / arama / filmografi.

Alanlar (yoksa null ya da []):
- mood: SADECE kullanıcı duygusunu AÇIKÇA söylediyse ("canım sıkkın", "mutluyum", "yorgunum"). Merak, keşfetme isteği, tür ya da atmosfer tercihi ruh hali DEĞİLDİR -> null. Emin değilsen null.
- include_genres: şu listeden: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, War, Western. Yalnızca kullanıcı istediyse / çok belirginse.
- exclude_genres: kullanıcının AÇIKÇA istemediği türler ("korku olmasın", "aksiyon barındırmayan", "süper kahraman olmasın" -> Action). Aynı listeden.
- reference_movie: "X'e benzer" denen filmin ORİJİNAL (İngilizce) adı. Yalnızca kullanıcı bir film verip benzerini istiyorsa.
- movie_title: intent=movie_info ise filmin orijinal adı.
- director / actor: adı geçiyorsa.
- min_rating / max_rating: YALNIZCA kullanıcı sayısal puan sınırı verdiyse (0-10). "5'in altında" -> max_rating=5. "8 üzeri" -> min_rating=8. Varsayılan değer UYDURMA.
- year_from / year_to: dönem belirtildiyse ("1920'ler" -> 1920 ve 1929, "2000'ler" -> 2000 ve 2009).
- semantic_query_en: filmin konusunu/atmosferini tarif eden 1-2 cümlelik İNGİLİZCE açıklama. Sayısal kısıtları ve oyuncu/yönetmen adlarını KOYMA.
- must_have: kullanıcının sert kriterlerinin kısa listesi, Türkçe ("uzaylılar ekranda görünmez", "tek mekanda geçer", "siyah beyaz"). Tür/puan/yıl sınırlarını buraya da yaz.
- obscure: kullanıcı kült, az bilinen, bağımsız, düşük bütçeli ya da düşük puanlı film istiyorsa true.
- Önceki konuşma varsa ("daha komik olsun" gibi devam mesajları) onu dikkate al.

Çıktı: yalnızca JSON."""

_NULLISH = {"", "null", "none", "yok", "-", "n/a", "belirtilmedi"}


def _parse_json(raw: str) -> dict:
    clean = re.sub(r"```json\s?|```", "", raw or "").strip()
    m = re.search(r"(\{.*\})", clean, re.DOTALL)
    return json.loads(m.group(1) if m else clean)


def _canon_genres(values) -> List[str]:
    out = []
    for v in values or []:
        g = _GENRE_CANON.get(str(v).strip().lower())
        if g and g not in out:
            out.append(g)
    return out


def _clean_spec(d: dict) -> dict:
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, str) and v.strip().lower() in _NULLISH:
            v = None
        out[k] = v
    for k in ("include_genres", "exclude_genres", "must_have"):
        v = out.get(k)
        if v is None:
            out[k] = []
        elif isinstance(v, str):
            out[k] = [x.strip() for x in v.split(",") if x.strip()]
    out["include_genres"] = _canon_genres(out["include_genres"])
    out["exclude_genres"] = _canon_genres(out["exclude_genres"])
    if str(out.get("intent", "")).lower() not in ("general", "movie_info", "recommendation"):
        out["intent"] = "recommendation"
    out["intent"] = str(out["intent"]).lower()
    out["obscure"] = bool(out.get("obscure")) and str(out.get("obscure")).lower() != "false"
    return out


def parse_spec(raw: str) -> QuerySpec:
    try:
        return QuerySpec.model_validate(_clean_spec(_parse_json(raw)))
    except (ValueError, ValidationError, TypeError):
        logger.warning("QuerySpec ayrıştırılamadı; varsayılan öneri niyetine düşüldü.")
        return QuerySpec()   # güvenlik ağı: düz semantik arama


def _history_text(state: GraphState, n: int = 4) -> str:
    msgs = [_to_lc_message(m) for m in (state.get("messages") or [])][-n:]
    lines = []
    for m in msgs:
        who = "Kullanıcı" if isinstance(m, HumanMessage) else "Asistan"
        lines.append(f"{who}: {str(m.content)[:400]}")
    return "\n".join(lines)


async def analyze_intent_node(state: GraphState):
    sid = state.get("session_id")
    if sid:
        chat_statuses[sid] = "🔍 Sorgu analiz ediliyor..."

    hist = _history_text(state)
    user = (f"Önceki konuşma:\n{hist}\n\n" if hist else "") + f"Mesaj: {state['prompt']}"
    resp = await aclient.ainvoke([SystemMessage(content=EXTRACT_SYSTEM), HumanMessage(content=user)])
    spec = parse_spec(resp.content)
    return {"intent": spec.intent, "intent_data": spec.model_dump()}


# ============================================================================
# 2) ÖNERİ HATTI
# ============================================================================
def _card_genres(c: dict) -> set:
    return {g.strip().lower() for g in (c.get("Türler") or "").split(",") if g.strip()}


def _card_rating(c: dict) -> Optional[float]:
    try:
        return float(c.get("TMDB Puanı"))
    except (TypeError, ValueError):
        return None


def bayes(avg, votes, m=500, c=6.5) -> float:
    try:
        avg, votes = float(avg), float(votes)
    except (TypeError, ValueError):
        return 0.0
    return (votes * avg + m * c) / (votes + m)


def build_plan(spec: QuerySpec, prompt: str, user_id: Optional[int] = None) -> List[tuple]:
    """Hangi araçlar hangi argümanlarla çalışacak? Deterministik; LLM karar vermez."""
    min_votes = MIN_VOTES_OBSCURE if spec.obscure else MIN_VOTES
    excl = ",".join(spec.exclude_genres)
    rng = {"min_rating": spec.min_rating or 0.0, "max_rating": spec.max_rating if spec.max_rating is not None else 10.0,
           "year_from": spec.year_from or 0, "year_to": spec.year_to or 0}
    plan = []

    if spec.reference_movie:
        plan.append(("get_similar_movies", {"movie_title": spec.reference_movie}))
    if spec.director:
        plan.append(("search_movies_in_graph", {"category": "Director", "search_value": spec.director, "min_votes": min_votes}))
    if spec.actor:
        plan.append(("search_movies_in_graph", {"category": "Actor", "search_value": spec.actor, "min_votes": min_votes}))

    # Kullanıcı tanımlıysa ve öneri niyetindeyse GraphRAG Collaborative Filtering ekle
    if user_id and spec.intent == "recommendation":
        cf_args = {"user_id": user_id, "limit": 15, "min_votes": min_votes, **rng}
        if excl:
            cf_args["exclude_genres"] = excl
        plan.append(("get_collaborative_recommendations", {k: v for k, v in cf_args.items() if v not in (None, 0, 0.0) or k == "max_rating"}))

    has_filters = bool(spec.include_genres or spec.exclude_genres or spec.director or spec.actor
                       or spec.min_rating is not None or spec.max_rating is not None
                       or spec.year_from or spec.year_to)
    if has_filters:
        args = {"min_votes": min_votes, **rng}
        if spec.include_genres:
            args["genre_name"] = ",".join(spec.include_genres)
        if excl:
            args["exclude_genres"] = excl
        if spec.director:
            args["director_name"] = spec.director
        if spec.actor:
            args["actor_name"] = spec.actor
        plan.append(("search_movies_by_filters", {k: v for k, v in args.items() if v not in (None, 0, 0.0) or k == "max_rating"}))

    # Tema/atmosfer her zaman anlamsal aramayla da taranır (kişi/tür filtresi tek başına yetmez)
    if not (spec.director or spec.actor) or spec.semantic_query_en:
        plan.append(("search_movies_semantically", {
            "semantic_query": spec.semantic_query_en or prompt,
            "exclude_genres": excl or None, "min_votes": min_votes, **rng, "limit": 25}))
    return [(n, {k: v for k, v in a.items() if v is not None}) for n, a in plan]


def merge_candidates(result_lists: List[List[dict]]) -> List[dict]:
    """Kaynakları movie_id ile birleştir; her kaynaktaki sıradan Reciprocal Rank Fusion skoru üret."""
    by_id = {}
    for lst in result_lists:
        for rank, c in enumerate(lst):
            mid = str(c.get("movie_id") or "")
            if not mid:
                continue
            cur = by_id.setdefault(mid, {**c, "_rrf": 0.0, "_sources": []})
            cur["_rrf"] += 1.0 / (60 + rank)
            cur["_sources"].append(c.get("source", ""))
            for k in ("Özet", "Türler", "Director", "Cast", "Poster", "TMDB Puanı"):
                if not cur.get(k) and c.get(k):
                    cur[k] = c[k]
            cur["vote_count"] = max(cur.get("vote_count", 0), c.get("vote_count", 0))
    return list(by_id.values())


def hard_filter(cands: List[dict], spec: QuerySpec) -> List[dict]:
    """Kullanıcının sert kısıtlarını KODLA uygula. (LLM'in uymasına güvenme.)"""
    excl = {g.lower() for g in spec.exclude_genres}
    min_votes = MIN_VOTES_OBSCURE if spec.obscure else MIN_VOTES
    out = []
    for c in cands:
        if excl & _card_genres(c):
            continue
        r = _card_rating(c)
        if (spec.min_rating is not None or spec.max_rating is not None) and r is None:
            continue
        if spec.min_rating is not None and r < spec.min_rating:
            continue
        if spec.max_rating is not None and r > spec.max_rating:
            continue
        y = c.get("Yıl")
        if y and str(y).isdigit():
            if spec.year_from and int(y) < spec.year_from:
                continue
            if spec.year_to and int(y) > spec.year_to:
                continue
        if c.get("vote_count", 0) < min_votes:
            continue
        if not (c.get("Özet") or "").strip():      # özetsiz kayıt seçim için işe yaramaz
            continue
        out.append(c)
    return out


def score_candidates(cands: List[dict], spec: QuerySpec) -> List[dict]:
    """
    Aday filmleri skorlar: RRF + Bayes + tür uyumu + Reranker (varsa).

    Reranker (bge-reranker-v2-m3) devredeyse ve spec.semantic_query_en doluysa,
    adaylar Cross-Encoder ile yeniden skorlanır ve sonuç, statik formülle
    harmanlanır. Model yüklenemezse veya sorgu yoksa eski formül kullanılır.
    """
    m = 50 if spec.obscure else 500
    inc = {g.lower() for g in spec.include_genres}

    # ── Faz 1: Statik temel skor (her zaman) ──
    for c in cands:
        wr = bayes(c.get("TMDB Puanı"), c.get("vote_count", 0), m=m)
        genre_fit = (len(inc & _card_genres(c)) / len(inc)) if inc else 0.0
        c["_base"] = c["_rrf"] + 0.015 * (wr / 10) + 0.01 * genre_fit
        c["_score"] = c["_base"]

    # ── Faz 2: Cross-Encoder reranker (opsiyonel) ──
    query = spec.semantic_query_en
    if _rerank and query:
        pool = sorted(cands, key=lambda c: c["_base"], reverse=True)[:RERANKER_TOP_N]
        result = _rerank(query, pool)
        if result:
            raw = [s for _, s in result]
            s_min, s_max = min(raw), max(raw)
            s_rng = s_max - s_min if s_max > s_min else 1.0
            for c, s in result:
                norm = (s - s_min) / s_rng
                c["_score"] = c["_base"] + RERANKER_BOOST * norm
                c["_reranker"] = round(norm, 4)
            logger.info("Reranker: %d aday skorlandı (sorgu: %s...)",
                        len(result), query[:60])

    return sorted(cands, key=lambda c: c["_score"], reverse=True)


def candidate_similarity(c1: dict, c2: dict, spec: Optional[QuerySpec] = None) -> float:
    """
    İki aday film arasındaki benzerliği [0, 1] aralığında hesaplar.
    Tür Jaccard'ı, Yönetmen çakışması ve Özet kelime örtüşmesini harmanlar.
    """
    # 1. Tür benzerliği (Jaccard)
    g1 = _card_genres(c1)
    g2 = _card_genres(c2)
    genre_sim = len(g1 & g2) / len(g1 | g2) if (g1 and g2) else 0.0

    # 2. Yönetmen benzerliği (kullanıcı spesifik bir yönetmen aramamışsa)
    d1 = (c1.get("Director") or "").strip().lower()
    d2 = (c2.get("Director") or "").strip().lower()
    dir_sim = 0.0
    if not (spec and spec.director):
        if d1 and d2 and d1 == d2:
            dir_sim = 1.0

    # 3. Özet kelime kümesi benzerliği (Jaccard)
    words1 = set(re.findall(r"\w{4,}", (c1.get("Özet") or "").lower()))
    words2 = set(re.findall(r"\w{4,}", (c2.get("Özet") or "").lower()))
    text_sim = len(words1 & words2) / len(words1 | words2) if (words1 and words2) else 0.0

    return 0.5 * genre_sim + 0.3 * dir_sim + 0.2 * text_sim


def mmr_rerank(
    cands: List[dict],
    limit: int = CANDIDATES_TO_LLM,
    lambda_param: float = MMR_LAMBDA,
    spec: Optional[QuerySpec] = None,
) -> List[dict]:
    """
    Maximal Marginal Relevance (MMR) algoritması.
    Aday havuzundaki filmlerin alaka düzeyini (relevance) korurken,
    seçilen filmler arasındaki benzerliği cezalandırarak çeşitliliği (diversity) artırır.

    MMR = lambda * Rel(d) - (1 - lambda) * max_{s in S} Sim(d, s)
    """
    if not cands or limit <= 0:
        return []
    if len(cands) <= limit or lambda_param >= 1.0:
        return cands[:limit]

    # Alaka puanlarını [0, 1] aralığına normalize et
    raw_scores = [float(c.get("_score", 0.0)) for c in cands]
    s_min, s_max = min(raw_scores), max(raw_scores)
    s_rng = s_max - s_min if s_max > s_min else 1.0
    norm_scores = {id(c): (float(c.get("_score", 0.0)) - s_min) / s_rng for c in cands}

    selected = []
    remaining = list(cands)

    # 1. En yüksek skorlu ilk adayı doğrudan seç
    first = remaining.pop(0)
    first["_mmr"] = 1.0
    selected.append(first)

    # 2. Geriye kalan adaylardan MMR formülüne göre seçim yap
    while len(selected) < limit and remaining:
        best_cand = None
        best_mmr = -float("inf")
        best_idx = -1

        for idx, cand in enumerate(remaining):
            rel = norm_scores[id(cand)]
            max_sim = max(candidate_similarity(cand, s, spec=spec) for s in selected)
            mmr_score = lambda_param * rel - (1.0 - lambda_param) * max_sim

            if mmr_score > best_mmr:
                best_mmr = mmr_score
                best_cand = cand
                best_idx = idx

        if best_cand is not None:
            best_cand["_mmr"] = round(best_mmr, 4)
            selected.append(remaining.pop(best_idx))
        else:
            break

    return selected


def _brief(cands: List[dict]) -> str:
    def line(c):
        score_str = f"Sistem Skoru: {c.get('_score', 0):.2f}"
        return (f"[ID:{c['movie_id']}] {c.get('Film', '')} ({c.get('Yıl', '')}) "
                f"[{score_str}] "
                f"puan {c.get('TMDB Puanı', '?')} ({c.get('vote_count', 0)} oy) | {c.get('Türler', '')} | "
                f"{(c.get('Özet') or '')[:260]}")
    return "\n".join(line(c) for c in cands)


def _select_system(spec: QuerySpec, limit: int) -> str:
    mood_field = ('  "mood_response": "kullanıcının belirttiği ruh haline 1 cümlelik empati notu",\n'
                  if spec.mood else "")
    return f"""Bir film öneri sisteminin SON SEÇİM aşamasısın. Kullanıcıya 'sen' diye hitap et.
Adaylar arasından isteğe en uygun EN FAZLA {limit} filmi, EN UYGUNDAN başlayarak sırala.

Kurallar:
- Adayların yanındaki [Sistem Skoru: X] değeri bizim algoritmik uygunluk puanımızdır. Adayları seçerken ve sıralarken öncelikle Sistem Skoru yüksek olanlara MUTLAKA öncelik ver.
- Yalnızca verilen [ID:...] değerlerini kullan; film uydurma.
- Kullanıcının sert kriterlerinden birini açıkça ihlal eden filmi SEÇME. Emin değilsen seçme.
  (örn. "uzaylılar görünmesin" denmişse uzaylıların göründüğü bir film; "banka soygunu" denmişse soygun olmayan film.)
- Gerekçe (reason) SADECE adayın satırındaki bilgiye (tür, yıl, puan, özet) dayansın. Satırda olmayan
  bilgiyi (besteci, renk paleti, sahne detayı) iddia etme.
- Hiçbir aday kriterleri tam karşılamıyorsa no_exact_match=true yap ve yalnızca gerçekten yakın 1-3 filmi ver.
  Hiçbiri yakın değilse ranked'i boş bırak. Sayıyı doldurmak için alakasız film ekleme.
{"- Kullanıcı ruh halini açıkça söyledi: mood_response yaz." if spec.mood else "- Ruh hali belirtilmedi: mood_response alanını EKLEME, duygu uydurma."}

Yalnızca şu JSON'u döndür:
{{
  "text": "1-2 cümlelik samimi giriş (no_exact_match ise bunu dürüstçe söyle)",
{mood_field}  "no_exact_match": false,
  "ranked": [{{"id": "<ID>", "reason": "1 cümlelik gerekçe"}}]
}}"""


def parse_selection(raw: str, valid_ids: set, limit: int):
    """-> (text, mood_response, no_exact, [(id, reason)]) ; ayrıştırma başarısızsa None."""
    try:
        w = _parse_json(raw)
    except Exception:
        return None
    ranked = w.get("ranked")
    if isinstance(ranked, dict):                       # {id: reason} biçimine de tolerans
        ranked = [{"id": k, "reason": v} for k, v in ranked.items()]
    if not isinstance(ranked, list):
        return None
    picks, seen = [], set()
    for r in ranked:
        rid = str(r.get("id", "")).strip().replace("ID:", "") if isinstance(r, dict) else ""
        if rid in valid_ids and rid not in seen:       # uydurma / tekrar eden id'leri at
            seen.add(rid)
            picks.append((rid, str(r.get("reason", "")).strip()))
        if len(picks) == limit:
            break
    return w.get("text") or "", w.get("mood_response") or "", bool(w.get("no_exact_match")), picks


def _strip_internal(c: dict) -> dict:
    return {k: v for k, v in c.items() if not k.startswith("_") and k not in ("source", "score", "relaxed")}


async def recommendation_node(state: GraphState):
    sid = state.get("session_id")
    if sid:
        chat_statuses[sid] = "🧠 Öneri motoru hazırlanıyor..."

    spec = QuerySpec.model_validate(_clean_spec(state.get("intent_data") or {}))   # tekrar LLM çağrısı YOK
    tool_calls_log, tool_results_log = [], []

    async def run_tool(name, args, tag=None) -> List[dict]:
        try:
            result = await asyncio.wait_for(ctx.session.call_tool(name, args), timeout=TOOL_TIMEOUT)
            raw = result.content[0].text
        except Exception as e:
            raw = f"Araç hatası: {e}"
        tool_calls_log.append({"tool_name": tag or name, "arguments": args})
        tool_results_log.append({"tool_name": tag or name, "raw_result": raw})
        try:
            data = json.loads(raw)
            return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
        except Exception:
            return []

    # -- 1. paralel retrieval ---------------------------------------------------
    plan = build_plan(spec, state["prompt"], user_id=state.get("user_id"))
    if sid:
        chat_statuses[sid] = "🎬 Hibrit motor verileri topluyor..."
    results = await asyncio.gather(*[run_tool(n, a) for n, a in plan])

    # -- 2. birleştir, sert filtre, skorla --------------------------------------
    merged = merge_candidates(results)
    filtered = hard_filter(merged, spec)
    ranked_c = score_candidates(filtered, spec)
    if MMR_ENABLED:
        top = mmr_rerank(ranked_c, limit=CANDIDATES_TO_LLM, lambda_param=MMR_LAMBDA, spec=spec)
    else:
        top = ranked_c[:CANDIDATES_TO_LLM]
    reranker_active = any(c.get("_reranker") is not None for c in top)
    logger.info("pipeline: plan=%d araç, birleşik=%d, sert filtre sonrası=%d, LLM'e=%d, reranker=%s, mmr=%s",
                len(plan), len(merged), len(filtered), len(top),
                "aktif" if reranker_active else "pasif",
                "aktif" if MMR_ENABLED else "pasif")

    limit = MAX_MOVIES_PEOPLE if (spec.director or spec.actor) else MAX_MOVIES
    text = "Aradığın kriterlere uygun film bulunamadı. Farklı bir tür veya oyuncu denemek ister misin?"
    mood_response, final_ids, reasons = "", [], {}
    lite_by_id = {c["movie_id"]: c for c in top}

    # -- 3. sıralı seçim (tek LLM çağrısı) --------------------------------------
    if top:
        if sid:
            chat_statuses[sid] = "✍️ En iyi sonuçlar seçiliyor..."
        
        # JSON yapısının korunabilmesi için sınır kaldırıldı
        persona = state.get("persona") or "{}"
        
        must = "; ".join(spec.must_have) or "-"
        human = (f"İstek: {state['prompt']}\nSert kriterler: {must}\n"
                 f"Ruh hali: {spec.mood or 'belirtilmedi'}\n"
                 f"Kullanıcı profili (JSON): {persona}\n\n"
                 f"Adaylar:\n{_brief(top)}")
        try:
            resp = await aclient.ainvoke([SystemMessage(content=_select_system(spec, limit)),
                                          HumanMessage(content=human)])
            parsed = parse_selection(resp.content, set(lite_by_id), limit)
        except Exception:
            logger.error("Seçim aşaması hatası", exc_info=True)
            parsed = None

        if parsed is None:          # LLM çıktısı bozuk -> skor sırasıyla ilk 3 (dürüst etiketle)
            final_ids = [c["movie_id"] for c in top[:3]]
            text = "Kriterlerine en yakın bulabildiklerim bunlar:"
            reasons = {i: "Arama skoruna göre en yakın eşleşme." for i in final_ids}
        else:
            t, mood, _no_exact, picks = parsed
            final_ids = [i for i, _ in picks]                 # LLM'in SIRASI korunur
            reasons = {i: r for i, r in picks}
            mood_response = mood if spec.mood else ""         # ruh hali söylenmediyse asla gösterme
            if final_ids:
                text = t or "İşte senin için seçtiklerim:"
                if any(lite_by_id.get(i, {}).get("relaxed") for i in final_ids):
                    text += " (Tüm kriterleri birebir karşılayan film bulunamadı; en yakınları gösteriyorum.)"
            else:
                text = t or text

    # -- 4. yalnızca seçilenleri zenginleştir (fragman, platform, kadro) --------
    movies = []
    if final_ids:
        full = await run_tool("enrich_movies", {"movie_ids": [int(i) for i in final_ids]}, tag="enrich_movies")
        full_by_id = {c["movie_id"]: c for c in full}
        for i in final_ids:                                   # sırayı aday listesinden değil final_ids'ten al
            card = full_by_id.get(i) or lite_by_id.get(i)
            if card:
                card = _strip_internal(card)
                card["Neden Önerildi"] = reasons.get(i, "")
                movies.append(card)

    return {
        "final_output": json.dumps({"type": "movie_list", "text": text,
                                    "mood_response": mood_response, "movies": movies}, ensure_ascii=False),
        "tool_calls": tool_calls_log,
        "tool_results": tool_results_log,
    }


# ============================================================================
# 3) FİLM BİLGİSİ (tek film sorusu: oyuncular, yönetmen, puan...)
# ============================================================================
async def movie_info_node(state: GraphState):
    sid = state.get("session_id")
    if sid:
        chat_statuses[sid] = "🎞️ Film bilgisi getiriliyor..."
    spec = QuerySpec.model_validate(_clean_spec(state.get("intent_data") or {}))
    title = spec.movie_title or spec.reference_movie or state["prompt"]
    args = {"title": title, "year": spec.year_from or 0}
    try:
        result = await asyncio.wait_for(ctx.session.call_tool("get_movie_card", args), timeout=TOOL_TIMEOUT)
        raw = result.content[0].text
        cards = [c for c in json.loads(raw) if isinstance(c, dict)]
    except Exception:
        raw, cards = "[]", []
    text = f"{cards[0]['Film']} hakkında bilgiler:" if cards else f"'{title}' adında bir film bulamadım."
    return {
        "final_output": json.dumps({"type": "movie_list", "text": text, "mood_response": "",
                                    "movies": [_strip_internal(c) for c in cards]}, ensure_ascii=False),
        "tool_calls": [{"tool_name": "get_movie_card", "arguments": args}],
        "tool_results": [{"tool_name": "get_movie_card", "raw_result": raw}],
    }


async def general_chat_node(state: GraphState):
    sid = state.get("session_id")
    if sid:
        chat_statuses[sid] = "✍️ Yanıt oluşturuluyor..."
    lc_messages = [_to_lc_message(m) for m in state.get("messages", [])]
    if not lc_messages:
        lc_messages = [HumanMessage(content=state.get("prompt", ""))]
    response = await aclient.ainvoke(lc_messages)
    return {"final_output": response.content}


# --- Grafik kurulumu ---------------------------------------------------------
def route_by_intent(state: GraphState):
    intent = state.get("intent")
    if intent == "general":
        return "general_chatter"
    if intent == "movie_info" and (state.get("intent_data") or {}).get("movie_title"):
        return "movie_info"
    return "recommendation_engine"


workflow = StateGraph(GraphState)
workflow.add_node("intent_analyzer", analyze_intent_node)
workflow.add_node("recommendation_engine", recommendation_node)
workflow.add_node("movie_info", movie_info_node)
workflow.add_node("general_chatter", general_chat_node)
workflow.set_entry_point("intent_analyzer")
workflow.add_conditional_edges("intent_analyzer", route_by_intent)
workflow.add_edge("recommendation_engine", END)
workflow.add_edge("movie_info", END)
workflow.add_edge("general_chatter", END)
app = workflow.compile()


async def get_tools(session):
    mcp_tools = await session.list_tools()
    return [
        {
            'type': 'function',
            'function': {
                'name': tool.name,
                'description': tool.description,
                'parameters': tool.inputSchema,
            },
        }
        for tool in mcp_tools.tools
    ]

# Geriye dönük uyumluluk için alias
get_ollama_tools = get_tools


async def generate_user_profile(chat_history_text, favorites_text, current_profile_json="{}"):
    profiler_instructions = """
    Sen bir JSON profil güncelleme motorusun. Görevin, kullanıcının mevcut profilini (eski ağırlıklar) alıp, yeni etkileşimler (sohbet geçmişi, like/dislike geri bildirimleri ve favoriler) ışığında güncellemektir.
    
    KURALLAR:
    1. Decay (Zamanla Unutma): Mevcut profildeki ilgi alanlarının puanlarını (0.0 - 1.0 arası) yaklaşık %15-20 oranında düşür. (Örn: eskiden 0.8 olan ama yeni sohbette geçmeyen bir türü 0.65'e indir). Puanı 0.2'nin altına düşenleri listeden çıkar.
    2. Yeni İlgi Alanları: Yeni sohbetlerde, favorilerde veya 'Beğendi' (Like) bildirimlerinde geçen tür, yönetmen, oyuncu ve temaların puanlarını artır veya listeye yeni ekle (başlangıç puanı 0.8 - 1.0 arası).
    3. Negatif Sinyaller (Dislike): 'Beğenmedi' olarak işaretlenen veya sohbette istemediğini belirttiği ögeleri "avoid" (kaçınılanlar) listesine ekle veya mevcut puanını sıfırla.
    4. SADECE geçerli bir JSON döndür. Asla markdown (```json) veya ekstra açıklama ekleme.

    BEKLENEN JSON FORMATI:
    {
        "genres": {"Sci-Fi": 0.85, "Comedy": 0.4},
        "directors": {"Christopher Nolan": 0.9},
        "actors": {"Cillian Murphy": 0.7},
        "avoid": ["Horror", "Romance"],
        "vibe_summary": "1-2 cümlelik genel zevk özeti"
    }
    """

    user_input = f"""
    MEVCUT PROFİL JSON:
    {current_profile_json}

    YENİ FAVORİLER:
    {favorites_text}

    YENİ SOHBET GEÇMİŞİ VE GERİ BİLDİRİMLER:
    {chat_history_text}
    """

    try:
        response = await aclient.ainvoke([
            SystemMessage(content=profiler_instructions),
            HumanMessage(content=user_input)
        ])
        content = response.content.strip()
        # Markdown kod bloklarını temizle
        clean_content = re.sub(r'```json\s?|```', '', content).strip()
        
        # Geçerli bir JSON olup olmadığını test et
        json.loads(clean_content)
        return clean_content
    except Exception as e:
        logger.error("JSON Profil oluşturma hatası", exc_info=True)
        # LLM hata yaparsa profili sıfırlamak yerine eski profili koru
        return current_profile_json if current_profile_json.strip() else "{}"


async def generate_user_suggestions(chat_history_text, favorites_text):
    prompt_instructions = """
    Sen bir film öneri botunun asistanısın. 
    Kullanıcının film tercihleri, geçmiş sohbetleri ve favori filmlerine dayanarak, kullanıcının tıklayıp hızlıca sohbet başlatabileceği 3 adet yaratıcı ve kişiselleştirilmiş film öneri sorgusu (butonu) üret.
    
    Örnekler:
    - Bilim kurgu filmlerine ilgiliyse: "Bilim kurgu filmi öner" veya "Yapay zeka temalı film öner"
    - Christopher Nolan'ı seviyorsa: "Christopher Nolan filmi öner"
    - Klasik korku/gizem seviyorsa: "Tüyler ürpertici bir gizem filmi öner"
    - Canı sıkkınsa veya eğlenceli bir şeyler aradıysa: "Modumu yükseltecek bir komedi öner"
    
    Kurallar:
    1. Sorgular kısa, net, merak uyandırıcı ve doğrudan olsun (en fazla 4-6 kelime).
    2. Liste olarak SADECE geçerli bir JSON dizisi döndür. Başka hiçbir açıklama, giriş veya markdown olmasın.
    3. JSON Formatı: ["Sorgu 1", "Sorgu 2", "Sorgu 3"]
    """

    user_input = f"""
    KULLANICI FAVORİLERİ:
    {favorites_text}

    SOHBET GEÇMİŞİ:
    {chat_history_text}
    """

    try:
        response = await aclient.ainvoke([
            SystemMessage(content=prompt_instructions),
            HumanMessage(content=user_input)
        ])
        content = response.content.strip()
        clean_content = re.sub(r'```json\s?|```', '', content).strip()
        match = re.search(r'(\[.*\])', clean_content, re.DOTALL)
        if match:
            suggestions = json.loads(match.group(1))
            if isinstance(suggestions, list) and len(suggestions) >= 3:
                return suggestions[:3]
    except Exception as e:
        logger.error("Öneri oluşturma hatası", exc_info=True)
    
    return ["Bilim kurgu filmi öner", "Nolan filmi öner", "Tim Burton filmi öner"]

# ============================================================================
# 3) ARKA PLAN ÖNERİSİ: adaylar araçlardan gelir, LLM sadece seçer + yazar
# ============================================================================
async def _call_tool_cards(name: str, args: dict) -> List[dict]:
    try:
        result = await asyncio.wait_for(ctx.session.call_tool(name, args), timeout=TOOL_TIMEOUT)
        data = json.loads(result.content[0].text)
        return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
    except Exception:
        logger.warning(f"Aday aracı başarısız: {name} {args}", exc_info=True)
        return []


async def gather_push_candidates(user_id: int, favs: List[dict], exclude_ids: set,
                                 exclude_titles: set, n_seeds: int = 3, top_n: int = 12) -> List[dict]:
    # Favorilerden rastgele tohumlar: her seferinde farklı bir "benzer filmler" havuzu
    seeds = random.sample(favs, min(n_seeds, len(favs))) if favs else []
    jobs = [_call_tool_cards("get_similar_movies", {"movie_title": s["Film"]}) for s in seeds]
    jobs.append(_call_tool_cards("get_collaborative_recommendations",
                                 {"user_id": user_id, "limit": 15, "min_votes": MIN_VOTES, "max_rating": 10.0}))
    results = await asyncio.gather(*jobs)

    lists = []
    for seed, cards in zip(seeds, results[:len(seeds)]):
        for c in cards:
            c["_because"] = seed["Film"]          # gerekçe için hangi favoriden geldi
        lists.append(cards)
    lists.append(results[-1])                      # collaborative: _because yok

    candidates = []
    for c in merge_candidates(lists):
        mid = str(c.get("movie_id") or "")
        title = (c.get("Film") or "").lower().strip()
        if not mid or mid in exclude_ids or title in exclude_titles:   # favori + önceki öneriler
            continue
        if c.get("vote_count", 0) < MIN_VOTES:
            continue
        if not (c.get("Özet") or "").strip():
            continue
        c["_pscore"] = c["_rrf"] + 0.015 * (bayes(c.get("TMDB Puanı"), c.get("vote_count", 0)) / 10)
        candidates.append(c)

    candidates.sort(key=lambda c: c["_pscore"], reverse=True)
    return candidates[:top_n]


PICK_SYSTEM = """Sen samimi bir film danışmanısın. ADAY LİSTESİNDEN kullanıcının profiline en uygun TEK filmi seç ve kısa bir bildirim yaz.
Kurallar:
- SADECE listedeki bir filmi seç; id'yi listeden AYNEN kopyala.
- Mesaj Türkçe, en fazla 150 karakter, samimi ve merak uyandırıcı olsun.
- Mesajda yalnızca listede yazan bilgileri kullan. Yönetmen, oyuncu, yıl, ödül gibi listede olmayan hiçbir şeyi söyleme.
- "Neden aday" alanında bir film adı varsa mesajda ona atıf yapabilirsin.
- Profilde kaçındığı belirtilen tür veya temalardan seçme.
Çıktı SADECE JSON: {"id": "...", "message": "..."}"""


def _cand_line(c: dict) -> str:
    why = f"{c['_because']} filmine benzer" if c.get("_because") else "zevkine yakın kullanıcılar sevdi"
    return (f"[ID:{c['movie_id']}] {c.get('Film', '')} ({c.get('Yıl', '')}) | puan {c.get('TMDB Puanı', '?')} | "
            f"{c.get('Türler', '')} | {(c.get('Özet') or '')[:200]} | Neden aday: {why}")


def _fallback_message(c: dict) -> str:
    if c.get("_because"):
        return f"{c['_because']} filmini sevdiysen {c['Film']} de hoşuna gidebilir! 🍿"[:150]
    return f"Senin için bir film buldum: {c['Film']}! 🍿"[:150]


async def build_push_recommendation(user_id: int, persona: str, favs: List[dict],
                                    exclude_ids: set, exclude_titles: set) -> Optional[dict]:
    """-> {"movie_id", "title", "message", "reason"} veya None (uygun aday yoksa)."""
    cands = await gather_push_candidates(user_id, favs, exclude_ids, exclude_titles)
    if not cands:
        logger.info(f"Push adayı yok (user={user_id})")
        return None
    by_id = {str(c["movie_id"]): c for c in cands}

    pick_id, message = None, None
    try:
        user_msg = (f"PROFİL: {persona}\nFAVORİLER: {', '.join(f['Film'] for f in favs[:10]) or 'yok'}\n\n"
                    "ADAY LİSTESİ:\n" + "\n".join(_cand_line(c) for c in cands))
        resp = await aclient.ainvoke([SystemMessage(content=PICK_SYSTEM), HumanMessage(content=user_msg)])
        raw = resp.content
        text = " ".join(c.get("text", "") if isinstance(c, dict) else str(c) for c in raw) if isinstance(raw, list) else str(raw)
        clean = re.sub(r'```json\s?|```', '', text).strip()
        m = re.search(r'(\{.*\})', clean, re.DOTALL)
        data = json.loads(m.group(1) if m else clean)
        pid = str(data.get("id", "")).replace("ID:", "").strip()
        if pid in by_id:
            pick_id, message = pid, (data.get("message") or "").strip()
        else:
            logger.warning(f"LLM listede olmayan id döndürdü: {pid!r}")       # uydurma id'yi at
    except Exception:
        logger.error("Öneri seçimi (LLM) başarısız, ilk aday kullanılacak", exc_info=True)

    chosen = by_id.get(pick_id) or cands[0]
    if not message:
        message = _fallback_message(chosen)
    if len(message) > 180:
        message = message[:177] + "…"

    reason = (f"{chosen['_because']} filmini sevdiğin için" if chosen.get("_because")
              else "Zevkine yakın kullanıcılar bu filmi sevdi")
    return {"movie_id": str(chosen["movie_id"]), "title": chosen.get("Film"),
            "message": message, "reason": reason}
