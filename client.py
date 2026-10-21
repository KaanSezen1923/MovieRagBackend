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
  * Konu/tema sorgularında (örn. "vampir dizileri öner") QuerySpec.keyword çıkarımı ve otomatik
    keyword algılama eklendi; filtre ve graf araçlarına keyword aktarılır, keyword gevşetilmiş
    (relaxed=["keyword"]) genel tür dolgularının RRF skoru düşürülür.
"""
import asyncio
import json
import os
import random
import re
from itertools import zip_longest
from typing import List, Optional
from urllib.parse import urlparse

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, ValidationError
from uvicorn import logging
from logger import logger
from shared import GraphState, chat_statuses, ctx   # TEK GraphState tanımı (intent_data dahil)
from movie_common import norm

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
SERIES_VOTE_FACTOR = 0.3    # dizilerin oy sayıları düşüktür: eşik = MIN_VOTES * bu çarpan
SERIES_TV_SHARE = 0.4       # media_type='both' iken aday havuzunda dizi payı
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

_PROMPT_KEYWORD_MAP = {
    "vampir": "vampire",
    "vampire": "vampire",
    "zombi": "zombie",
    "zombie": "zombie",
    "kurt adam": "werewolf",
    "kurtadam": "werewolf",
    "werewolf": "werewolf",
    "cadi": "witch",
    "witch": "witch",
    "hayalet": "ghost",
    "seri katil": "serial killer",
    "serial killer": "serial killer",
    "zamanda yolculuk": "time travel",
    "zaman yolculugu": "time travel",
    "time travel": "time travel",
    "siberpunk": "cyberpunk",
    "cyberpunk": "cyberpunk",
    "banka soygunu": "bank robbery",
    "soygun": "heist",
    "yapay zeka": "artificial intelligence",
    "artificial intelligence": "artificial intelligence",
    "uzayli": "alien",
    "astronot": "astronaut",
    "hapishane": "prison",
    "distopya": "dystopia",
    "distopik": "dystopia",
    "kiyamet sonrasi": "post-apocalyptic",
    "hafiza kaybi": "amnesia",
    "deniz feneri": "lighthouse",
    "ejderha": "dragon",
    "suikastci": "assassin",
    "dedektif": "detective",
    "mahkeme": "courtroom",
    "hayatta kalma": "survival",
    "simulasyon": "simulation",
}


def _infer_keyword_from_text(*texts: Optional[str]) -> Optional[str]:
    joined = norm(" ".join(t for t in texts if t))
    if not joined:
        return None
    for pat, kw in sorted(_PROMPT_KEYWORD_MAP.items(), key=lambda x: -len(x[0])):
        if re.search(rf"\b{re.escape(pat)}", joined):
            return kw
    return None


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
    keyword: Optional[str] = None             # spesifik konu/tema/varlık varsa İngilizce TMDB keyword ("vampire" vb.)
    min_rating: Optional[float] = None
    max_rating: Optional[float] = None
    year_from: Optional[int] = None
    year_to: Optional[int] = None
    semantic_query_en: Optional[str] = None   # tema/atmosfer, İngilizce, sayısal kısıtsız
    must_have: List[str] = Field(default_factory=list)   # kullanıcının sert kriterleri (seçimde kontrol edilir)
    obscure: bool = False                     # kült / az bilinen / düşük bütçe / düşük puanlı istiyor
    media_type: str = "movie"                 # movie | tv | both

    if not hasattr(BaseModel, "model_validate"):
        @classmethod
        def model_validate(cls, obj):
            return cls.parse_obj(obj)

        def model_dump(self):
            return self.dict()


EXTRACT_SYSTEM = """Sen bir film botunun sorgu analiz bileşenisin. Kullanıcı mesajından SADECE aşağıdaki JSON'u üret.

intent:
- "general": selam, sohbet, filmle ilgisiz.
- "movie_info": belirli BİR filmin detayı soruluyor (oyuncular, yönetmen, puan...). movie_title'ı doldur.
- "recommendation": film önerisi / arama / filmografi.

Alanlar (yoksa null ya da []):
- media_type: "movie" (kullanıcı film/sinema dedi), "tv" (dizi/sezon/bölüm dedi), "both" (ikisini de istedi YA DA
  hiçbirini belirtmedi: "bir şey öner", "bu akşam ne izlesem", "korku izlemek istiyorum"). Önceki konuşmadaki tercihi koru.
- mood: SADECE kullanıcı duygusunu AÇIKÇA söylediyse ("canım sıkkın", "mutluyum", "yorgunum"). Merak, keşfetme isteği, tür ya da atmosfer tercihi ruh hali DEĞİLDİR -> null. Emin değilsen null.
- include_genres: şu listeden: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, War, Western. Yalnızca kullanıcı istediyse / çok belirginse.
- exclude_genres: kullanıcının AÇIKÇA istemediği türler ("korku olmasın", "aksiyon barındırmayan", "süper kahraman olmasın" -> Action). Aynı listeden.
- reference_movie: "X'e benzer" denen film VEYA dizinin ORİJİNAL (İngilizce) adı. Yalnızca kullanıcı bir film verip benzerini istiyorsa.
- movie_title: intent=movie_info ise filmin/dizinin orijinal adı.
- director / actor: adı geçiyorsa.
- keyword: kullanıcının istediği spesifik bir konu, yaratık, alt-tema veya mekan varsa tekil İNGİLİZCE TMDB anahtar kelimesi (örn. "vampire", "zombie", "werewolf", "witch", "ghost", "time travel", "serial killer", "cyberpunk", "heist", "alien", "space", "artificial intelligence", "survival", "post-apocalyptic", "amnesia", "lighthouse", "prison", "detective"). Sadece genel tür ("komedi öner", "aksiyon filmi") istendiyse null.
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
    if isinstance(out.get("keyword"), list):
        out["keyword"] = ", ".join(str(x).strip() for x in out["keyword"] if str(x).strip()) or None
    elif isinstance(out.get("keyword"), str):
        out["keyword"] = out["keyword"].strip() or None
    if str(out.get("intent", "")).lower() not in ("general", "movie_info", "recommendation"):
        out["intent"] = "recommendation"
    out["intent"] = str(out["intent"]).lower()
    mt = str(out.get("media_type") or "movie").strip().lower()
    out["media_type"] = mt if mt in ("movie", "tv", "both") else "movie"
    out["obscure"] = bool(out.get("obscure")) and str(out.get("obscure")).lower() != "false"
    return out


def parse_spec(raw: str) -> QuerySpec:
    try:
        return QuerySpec.model_validate(_clean_spec(_parse_json(raw)))
    except (ValueError, ValidationError, TypeError):
        logger.warning("QuerySpec ayrıştırılamadı; varsayılan öneri niyetine düşüldü.")
        return QuerySpec()   # güvenlik ağı: düz semantik arama


def _fallback_spec_from_prompt(prompt: str) -> QuerySpec:
    """LLM çağrısı zaman aşımına uğrarsa veya hata verirse prompt'tan deterministik QuerySpec üretir."""
    p_norm = norm(prompt or "")
    mt = "tv" if any(w in p_norm for w in ("dizi", "dizisi", "dizileri", "sezon", "bolum")) else "movie"
    kw = _infer_keyword_from_text(prompt)
    inc = []
    for k, g in _GENRE_CANON.items():
        if re.search(rf"\b{re.escape(norm(k))}\b", p_norm) and g not in inc:
            inc.append(g)
    return QuerySpec(
        intent="recommendation",
        media_type=mt,
        include_genres=inc,
        keyword=kw,
        semantic_query_en=kw or prompt,
    )


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
    try:
        resp = await aclient.ainvoke([SystemMessage(content=EXTRACT_SYSTEM), HumanMessage(content=user)])
        spec = parse_spec(resp.content)
        # Canlı LLM çalışırken spesifik konu kelimesi ("vampir" -> "vampire" vb.) boş bırakıldıysa tamamla
        if isinstance(aclient, ChatOllama) and spec.intent == "recommendation" and not spec.keyword:
            inferred_kw = _infer_keyword_from_text(state.get("prompt"), spec.semantic_query_en)
            if inferred_kw:
                spec.keyword = inferred_kw
    except Exception:
        logger.error("Niyet analizi LLM hatası; deterministik yedek spec kullanılıyor", exc_info=True)
        spec = _fallback_spec_from_prompt(state.get("prompt", ""))
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


# ---- kişiselleştirme yardımcıları -------------------------------------------
PERSONA_WEIGHT = float(os.getenv("PERSONA_WEIGHT", "0.02"))   # skora persona katkısı (RRF/Bayes ölçeğinde)
SEEN_PENALTY = float(os.getenv("SEEN_PENALTY", "0.02"))       # daha önce gösterilene hafif ceza
PROFILE_HALF_LIFE_DAYS = float(os.getenv("PROFILE_HALF_LIFE_DAYS", "60"))


def _persona_obj(raw) -> dict:
    """Persona JSON metni (ya da dict) -> dict. Bozuksa boş dict."""
    if isinstance(raw, dict):
        return raw
    try:
        p = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return p if isinstance(p, dict) else {}


def _weights(d, canon_genres: bool = False) -> dict:
    """{ad: ağırlık} -> küçük harfli anahtarlı, yalnızca sayısal değerli sözlük."""
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                key = str(k).strip().lower()
                if canon_genres:
                    key = _GENRE_CANON.get(key, key).lower()
                out[key] = float(v)
    return out


def _avoid_genres(p: dict) -> set:
    """Persona 'avoid' listesindeki TMDB türleri (küçük harf). Tür olmayan temalar LLM'e kalır."""
    av = p.get("avoid")
    out = set()
    for a in (av if isinstance(av, list) else []):
        g = _GENRE_CANON.get(str(a).strip().lower())
        if g:
            out.add(g.lower())
    return out


def _top_genres(p: dict, n: int = 2) -> List[str]:
    avoid = _avoid_genres(p)
    items = []
    for k, w in _weights(p.get("genres"), canon_genres=True).items():
        g = _GENRE_CANON.get(k)
        if g and g.lower() not in avoid:
            items.append((w, g))
    return [g for _, g in sorted(items, reverse=True)[:n]]


def _persona_fit(c: dict, p: dict) -> float:
    """Adayın profile uyumu [0,1]: tür ağırlığı (%60) + yönetmen/oyuncu eşleşmesi (%40)."""
    if not isinstance(p, dict) or not p:
        return 0.0
    gw = _weights(p.get("genres"), canon_genres=True)
    genre = max((gw.get(g, 0.0) for g in _card_genres(c)), default=0.0)
    people = {**_weights(p.get("actors")), **_weights(p.get("directors"))}
    crew = {x.strip().lower() for x in f"{c.get('Director') or ''},{c.get('Cast') or ''}".split(",") if x.strip()}
    person = max((people.get(n, 0.0) for n in crew), default=0.0)
    return min(1.0, 0.6 * genre + 0.4 * person)


def decay_profile(profile, days: float, half_life: float = PROFILE_HALF_LIFE_DAYS, floor: float = 0.15) -> dict:
    """Zamana bağlı yaşlandırma (KODDA). Yarı ömür: half_life gün; floor altı girdiler silinir."""
    p = _persona_obj(profile)
    if days and days > 0:
        f = 0.5 ** (days / half_life)
        for k in ("genres", "directors", "actors"):
            d = p.get(k) if isinstance(p.get(k), dict) else {}
            p[k] = {n: round(float(w) * f, 3) for n, w in d.items()
                    if isinstance(w, (int, float)) and not isinstance(w, bool) and float(w) * f >= floor}
    return p


def sanitize_profile(profile) -> dict:
    """LLM'in ürettiği profili şemaya zorla: ağırlıklar [0,1], türler TMDB adlarına normalize."""
    p = _persona_obj(profile)

    def clamp(d, genres: bool = False) -> dict:
        res = {}
        if isinstance(d, dict):
            for k, v in d.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    name = str(k).strip()
                    if genres:
                        name = _GENRE_CANON.get(name.lower())
                    if not name:
                        continue
                    res[name] = max(res.get(name, 0.0), round(min(1.0, max(0.0, float(v))), 3))
        return res

    av = p.get("avoid")
    vs = p.get("vibe_summary")
    return {
        "genres": clamp(p.get("genres"), genres=True),
        "directors": clamp(p.get("directors")),
        "actors": clamp(p.get("actors")),
        "avoid": [str(a).strip() for a in av if str(a).strip()][:20] if isinstance(av, list) else [],
        "vibe_summary": vs.strip()[:400] if isinstance(vs, str) else "",
    }


def _build_movie_plan(spec: QuerySpec, prompt: str, user_id: Optional[int] = None) -> List[tuple]:
    """FİLM araçları. Hangi araçlar hangi argümanlarla çalışacak? Deterministik; LLM karar vermez."""
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
    if spec.keyword:
        plan.append(("search_movies_in_graph", {"category": "Keyword", "search_value": spec.keyword, "min_votes": min_votes}))

    # Kullanıcı tanımlıysa ve öneri niyetindeyse GraphRAG Collaborative Filtering ekle
    if user_id and spec.intent == "recommendation":
        cf_args = {"user_id": user_id, "limit": 15, "min_votes": min_votes, **rng}
        if excl:
            cf_args["exclude_genres"] = excl
        plan.append(("get_collaborative_recommendations", {k: v for k, v in cf_args.items() if v not in (None, 0, 0.0) or k == "max_rating"}))

    has_filters = bool(spec.include_genres or spec.exclude_genres or spec.director or spec.actor
                       or spec.keyword or spec.min_rating is not None or spec.max_rating is not None
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
        if spec.keyword:
            args["keyword"] = spec.keyword
        plan.append(("search_movies_by_filters", {k: v for k, v in args.items() if v not in (None, 0, 0.0) or k == "max_rating"}))

    # Tema/atmosfer her zaman anlamsal aramayla da taranır (kişi/tür filtresi tek başına yetmez)
    if not (spec.director or spec.actor) or spec.semantic_query_en:
        plan.append(("search_movies_semantically", {
            "semantic_query": spec.semantic_query_en or prompt,
            "exclude_genres": excl or None, "min_votes": min_votes, **rng, "limit": 25}))
    return [(n, {k: v for k, v in a.items() if v is not None}) for n, a in plan]


def _build_series_plan(spec: QuerySpec, prompt: str) -> List[tuple]:
    """DİZİ araçları (film planının karşılığı). TMDB dizilerde oyuncu/yaratıcı filtresi olmadığı için
    kişi sorguları Neo4j grafına gider."""
    base = MIN_VOTES_OBSCURE if spec.obscure else MIN_VOTES
    min_votes = max(5, int(base * SERIES_VOTE_FACTOR))
    excl = ",".join(spec.exclude_genres)
    rng = {"min_rating": spec.min_rating or 0.0, "max_rating": spec.max_rating if spec.max_rating is not None else 10.0,
           "year_from": spec.year_from or 0, "year_to": spec.year_to or 0}
    plan = []
    if spec.reference_movie:
        plan.append(("get_similar_series", {"series_title": spec.reference_movie}))
    if spec.director:      # dizide 'yönetmen' yerine yaratıcı
        plan.append(("search_series_in_graph", {"category": "Director", "search_value": spec.director, "min_votes": min_votes}))
    if spec.actor:
        plan.append(("search_series_in_graph", {"category": "Actor", "search_value": spec.actor, "min_votes": min_votes}))
    if spec.keyword:
        plan.append(("search_series_in_graph", {"category": "Keyword", "search_value": spec.keyword, "min_votes": min_votes}))

    if spec.include_genres or spec.exclude_genres or spec.keyword or spec.min_rating is not None \
            or spec.max_rating is not None or spec.year_from or spec.year_to:
        args = {"min_votes": min_votes, **rng}
        if spec.include_genres:
            args["genre_name"] = ",".join(spec.include_genres)
        if excl:
            args["exclude_genres"] = excl
        if spec.keyword:
            args["keyword"] = spec.keyword
        plan.append(("search_series_by_filters", {k: v for k, v in args.items() if v not in (None, 0, 0.0) or k == "max_rating"}))

    if not (spec.director or spec.actor) or spec.semantic_query_en:
        plan.append(("search_series_semantically", {
            "semantic_query": spec.semantic_query_en or prompt,
            "exclude_genres": excl or None, "min_votes": min_votes, **rng, "limit": 25}))
    return [(n, {k: v for k, v in a.items() if v is not None}) for n, a in plan]


def build_plan(spec: QuerySpec, prompt: str, user_id: Optional[int] = None) -> List[tuple]:
    """media_type'a göre film ve/veya dizi araçlarını birleştirir."""
    plan = []
    if spec.media_type in ("movie", "both"):
        plan += _build_movie_plan(spec, prompt, user_id)
    if spec.media_type in ("tv", "both"):
        plan += _build_series_plan(spec, prompt)
    return plan


def balance_media(ranked: List[dict], k: int, tv_share: float = SERIES_TV_SHARE) -> List[dict]:
    """'both' sorgularında film/dizi dengesi: aday havuzunun en az ~%tv_share'i dizi olsun (varsa)."""
    tv = [c for c in ranked if c.get("media_type") == "tv"]
    mv = [c for c in ranked if c.get("media_type") != "tv"]
    n_tv = min(len(tv), round(k * tv_share))
    n_mv = min(len(mv), k - n_tv)
    n_tv = min(len(tv), k - n_mv)
    return sorted(tv[:n_tv] + mv[:n_mv], key=lambda c: c.get("_score", 0.0), reverse=True)


def merge_candidates(result_lists: List[List[dict]]) -> List[dict]:
    """Kaynakları movie_id ile birleştir; her kaynaktaki sıradan Reciprocal Rank Fusion skoru üret."""
    by_id = {}
    for lst in result_lists:
        for rank, c in enumerate(lst):
            mid = str(c.get("movie_id") or "")
            if not mid:
                continue
            key = f"{c.get('media_type') or 'movie'}:{mid}"      # film 1399 ile dizi 1399 aynı şey DEĞİL
            cur = by_id.setdefault(key, {**c, "_rrf": 0.0, "_sources": [], "_key": key})
            # Keyword gevşetilerek (relaxed=['keyword']) dönen genel tür dolgularının RRF ağırlığını düşür
            relaxed_list = c.get("relaxed") or []
            w_rrf = 0.35 if "keyword" in relaxed_list else 1.0
            cur["_rrf"] += w_rrf * (1.0 / (60 + rank))
            cur["_sources"].append(c.get("source", ""))
            if not relaxed_list and cur.get("relaxed"):
                cur.pop("relaxed", None)
            for k in ("Özet", "Türler", "Director", "Cast", "Poster", "TMDB Puanı"):
                if not cur.get(k) and c.get(k):
                    cur[k] = c[k]
            cur["vote_count"] = max(cur.get("vote_count", 0), c.get("vote_count", 0))
    return list(by_id.values())


def hard_filter(cands: List[dict], spec: QuerySpec, blocked=frozenset(), avoid=frozenset()) -> List[dict]:
    """Kullanıcının sert kısıtlarını KODLA uygula. (LLM'in uymasına güvenme.)
    blocked: dislike edilen film id'leri. avoid: profildeki kaçınılan türler (küçük harf);
    kullanıcı bu sorguda o türü AÇIKÇA istediyse (include_genres) engellenmez."""
    avoid = set(avoid) - {g.lower() for g in spec.include_genres}
    excl = {g.lower() for g in spec.exclude_genres} | avoid
    min_votes = MIN_VOTES_OBSCURE if spec.obscure else MIN_VOTES
    out = []
    for c in cands:
        if str(c.get("movie_id") or "") in blocked:
            continue
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
        mv = max(5, int(min_votes * SERIES_VOTE_FACTOR)) if c.get("media_type") == "tv" else min_votes
        if c.get("vote_count", 0) < mv:
            continue
        if not (c.get("Özet") or "").strip():      # özetsiz kayıt seçim için işe yaramaz
            continue
        out.append(c)
    return out


def score_candidates(cands: List[dict], spec: QuerySpec, persona: Optional[dict] = None,
                     seen=frozenset()) -> List[dict]:
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
        wr = bayes(c.get("TMDB Puanı"), c.get("vote_count", 0),
                   m=(max(10, m // 5) if c.get("media_type") == "tv" else m))
        genre_fit = (len(inc & _card_genres(c)) / len(inc)) if inc else 0.0
        pfit = _persona_fit(c, persona or {})
        c["_persona_fit"] = round(pfit, 3)
        c["_base"] = (c["_rrf"] + 0.015 * (wr / 10) + 0.01 * genre_fit
                      + PERSONA_WEIGHT * pfit
                      - (SEEN_PENALTY if str(c.get("movie_id") or "") in seen else 0.0))
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
                norm_s = (s - s_min) / s_rng
                c["_score"] = c["_base"] + RERANKER_BOOST * norm_s
                c["_reranker"] = round(norm_s, 4)
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


def _brief(cands: List[dict], keyword: Optional[str] = None) -> str:
    def line(c):
        score_str = f"Sistem Skoru: {c.get('_score', 0):.2f}"
        pfit = c.get("_persona_fit", 0.0)
        pfit_str = f" | Profil Uyumu: %{int(pfit*100)}" if pfit > 0 else ""
        kind = "DİZİ" if c.get("media_type") == "tv" else "FİLM"
        srcs = c.get("_sources") or []
        kw_tag = ""
        if keyword and not c.get("relaxed") and any(
            s in srcs for s in ("graph_keyword", "graph_series_keyword", "semantic", "semantic_series", "tmdb_filter")
        ):
            kw_tag = f" [Etiket: {keyword}]"
        return (f"[ID:{c['_key']}] [{kind}]{kw_tag} {c.get('Film', '')} ({c.get('Yıl', '')}) "
                f"[{score_str}{pfit_str}] "
                f"puan {c.get('TMDB Puanı', '?')} ({c.get('vote_count', 0)} oy) | {c.get('Türler', '')} | "
                f"{(c.get('Özet') or '')[:260]}")
    return "\n".join(line(c) for c in cands)


def _select_system(spec: QuerySpec, limit: int) -> str:
    mood_field = ('  "mood_response": "kullanıcının belirttiği ruh haline 1 cümlelik empati notu",\n'
                  if spec.mood else "")
    mix_rule = ("- Kullanıcı film ya da dizi ayrımı yapmadı: uygunsa hem [FİLM] hem [DİZİ] içeren karışık bir liste ver.\n"
                if spec.media_type == "both" else
                "- Yalnızca [DİZİ] adaylarını seç.\n" if spec.media_type == "tv" else "")
    return f"""Bir film ve dizi öneri sisteminin SON SEÇİM aşamasısın. Kullanıcıya 'sen' diye hitap et.
Adaylar arasından isteğe en uygun EN FAZLA {limit} içeriği (film veya dizi), EN UYGUNDAN başlayarak sırala.

Kurallar:
- Adayların yanındaki [Sistem Skoru: X] değeri algoritmik uygunluk puanımızdır. Sıralarken Sistem Skoru ve Profil Uyumu yüksek olanlara MUTLAKA öncelik ver.
- Kullanıcı Profili (JSON): Kullanıcının zevklerine (sevdiği türler, yönetmenler, oyuncular) uygun içeriklere öncelik ver. Gerekçede (reason), adayın kullanıcının profiline/zevkine nasıl hitap ettiğini de samimi şekilde belirt. Profildeki 'avoid' (kaçınılan) ögeleri içeren adayları seçme.
- Yalnızca verilen [ID:...] değerlerini (örn. movie:27205 ya da tv:1399) AYNEN kullan; film/dizi uydurma.
{mix_rule}
- Kullanıcının sert kriterlerinden birini açıkça ihlal eden filmi SEÇME. Emin değilsen seçme.
  (örn. "uzaylılar görünmesin" denmişse uzaylıların göründüğü bir film; "banka soygunu" denmişse soygun olmayan film.)
- Gerekçe (reason) SADECE adayın satırındaki bilgiye (tür, yıl, puan, etiket, özet) dayansın. Satırda olmayan
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
        if rid not in valid_ids and rid.isdigit():          # LLM "27205" yazdıysa: tek eşleşme varsa tamamla
            hit = [k for k in (f"movie:{rid}", f"tv:{rid}") if k in valid_ids]
            if len(hit) == 1:
                rid = hit[0]
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
    tool_errors = []
    _ERR_PREFIXES = ("Araç hatası", "Semantik arama hatası", "Graf arama hatası",
                     "Dizi semantik arama hatası", "Dizi graf arama hatası")

    async def run_tool(name, args, tag=None) -> List[dict]:
        try:
            result = await asyncio.wait_for(ctx.session.call_tool(name, args), timeout=TOOL_TIMEOUT)
            raw = result.content[0].text
        except Exception as e:
            raw = f"Araç hatası: {e}"
        if raw.startswith(_ERR_PREFIXES):
            tool_errors.append(raw)
            logger.error("Araç hatası (%s): %s", name, raw)
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
    persona_p = _persona_obj(state.get("persona"))
    filtered = hard_filter(merged, spec, set(state.get("blocked_ids") or []), _avoid_genres(persona_p))
    ranked_c = score_candidates(filtered, spec, persona_p, set(state.get("seen_ids") or []))
    pool_n = CANDIDATES_TO_LLM * 2 if spec.media_type == "both" else CANDIDATES_TO_LLM
    if MMR_ENABLED:
        top = mmr_rerank(ranked_c, limit=pool_n, lambda_param=MMR_LAMBDA, spec=spec)
    else:
        top = ranked_c[:pool_n]
    if spec.media_type == "both":
        top = balance_media(top, CANDIDATES_TO_LLM)
    reranker_active = any(c.get("_reranker") is not None for c in top)
    logger.info("pipeline: plan=%d araç, birleşik=%d, sert filtre sonrası=%d, LLM'e=%d, reranker=%s, mmr=%s",
                len(plan), len(merged), len(filtered), len(top),
                "aktif" if reranker_active else "pasif",
                "aktif" if MMR_ENABLED else "pasif")

    limit = MAX_MOVIES_PEOPLE if (spec.director or spec.actor) else MAX_MOVIES
    noun = {"movie": "film", "tv": "dizi"}.get(spec.media_type, "film ya da dizi")
    text = f"Aradığın kriterlere uygun {noun} bulunamadı. Farklı bir tür veya oyuncu denemek ister misin?"
    if not top and tool_errors:       # "bulunamadı" değil, gerçek neden: servis hatası
        text = "Şu anda arama servisine tam olarak ulaşamıyorum. Lütfen biraz sonra tekrar dene."
    mood_response, final_ids, reasons = "", [], {}
    lite_by_id = {c["_key"]: c for c in top}

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
                 f"Adaylar:\n{_brief(top, keyword=spec.keyword)}")
        try:
            resp = await aclient.ainvoke([SystemMessage(content=_select_system(spec, limit)),
                                          HumanMessage(content=human)])
            parsed = parse_selection(resp.content, set(lite_by_id), limit)
        except Exception:
            logger.error("Seçim aşaması hatası", exc_info=True)
            parsed = None

        if parsed is None:          # LLM çıktısı bozuk -> skor sırasıyla ilk 3 (dürüst etiketle)
            final_ids = [c["_key"] for c in top[:3]]
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
        mv_ids = [int(k.split(":", 1)[1]) for k in final_ids if k.startswith("movie:")]
        tv_ids = [int(k.split(":", 1)[1]) for k in final_ids if k.startswith("tv:")]
        jobs = []
        if mv_ids:
            jobs.append(run_tool("enrich_movies", {"movie_ids": mv_ids}, tag="enrich_movies"))
        if tv_ids:
            jobs.append(run_tool("enrich_series", {"series_ids": tv_ids}, tag="enrich_series"))
        full_by_key = {}
        for lst in await asyncio.gather(*jobs):
            for c in lst:
                full_by_key[f"{c.get('media_type') or 'movie'}:{c['movie_id']}"] = c
        for k in final_ids:                                   # sırayı aday listesinden değil final_ids'ten al
            card = full_by_key.get(k) or lite_by_id.get(k)
            if card:
                pfit = lite_by_id.get(k, {}).get("_persona_fit")
                card = _strip_internal(card)
                card.setdefault("media_type", "tv" if k.startswith("tv:") else "movie")
                card["Neden Önerildi"] = reasons.get(k, "")
                if pfit is not None:
                    card["profil_uyumu"] = pfit
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
    tools = {"movie": ["get_movie_card"], "tv": ["get_series_card"],
             "both": ["get_movie_card", "get_series_card"]}[spec.media_type]
    cards, calls, results = [], [], []
    for tool in tools:                      # 'both': önce film, bulunamazsa dizi
        try:
            result = await asyncio.wait_for(ctx.session.call_tool(tool, args), timeout=TOOL_TIMEOUT)
            raw = result.content[0].text
            cards = [c for c in json.loads(raw) if isinstance(c, dict)]
        except Exception:
            raw, cards = "[]", []
        calls.append({"tool_name": tool, "arguments": args})
        results.append({"tool_name": tool, "raw_result": raw})
        if cards:
            break
    noun = {"movie": "film", "tv": "dizi"}.get(spec.media_type, "film ya da dizi")
    text = f"{cards[0]['Film']} hakkında bilgiler:" if cards else f"'{title}' adında bir {noun} bulamadım."
    return {
        "final_output": json.dumps({"type": "movie_list", "text": text, "mood_response": "",
                                    "movies": [_strip_internal(c) for c in cards]}, ensure_ascii=False),
        "tool_calls": calls,
        "tool_results": results,
    }


async def general_chat_node(state: GraphState):
    sid = state.get("session_id")
    if sid:
        chat_statuses[sid] = "✍️ Yanıt oluşturuluyor..."
    lc_messages = [_to_lc_message(m) for m in state.get("messages", [])]
    if not lc_messages:
        lc_messages = [HumanMessage(content=state.get("prompt", ""))]
    try:
        response = await aclient.ainvoke(lc_messages)
        return {"final_output": response.content}
    except Exception:
        logger.error("Genel sohbet LLM hatası", exc_info=True)
        return {"final_output": "Merhaba! Sana nasıl bir film veya dizi önermemi istersin?"}


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
app = workflow.compile(debug=os.getenv("GRAPH_DEBUG", "false").lower() == "true")


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
    1. Zamana bağlı yaşlandırma (decay) KOD tarafında zaten uygulandı: mevcut profildeki ağırlıkları DÜŞÜRME,
       yeni etkileşim yoksa olduğu gibi koru.
    2. Yeni sinyal: YALNIZCA "YENİ FAVORİLER", kullanıcının kendi (user) mesajları ve "Beğendi" geri bildirimleri
       ilgi sinyalidir. Listede olan tür/yönetmen/oyuncu için puanı +0.1 artır (en çok 1.0); listede yoksa
       0.6-0.8 arası bir puanla ekle.
    3. Asistan (assistant) satırlarındaki "Önerilen filmler" kullanıcının zevki DEĞİLDİR; yalnızca kullanıcının
       onlara verdiği tepki (istek, beğeni, itiraz) sinyaldir.
    4. Negatif sinyal: 'Beğenmedi' olan veya sohbette istemediğini belirttiği ögeleri "avoid" listesine ekle
       (tür ise TMDB tür adıyla: Action, Horror, Science Fiction...) ve varsa puanını sıfırla.
    5. SADECE geçerli bir JSON döndür. Asla markdown (```json) veya ekstra açıklama ekleme.

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
        clean_content = re.sub(r'```json\s?|```', '', content).strip()

        profile = sanitize_profile(json.loads(clean_content))
        return json.dumps(profile, ensure_ascii=False)
    except Exception as e:
        logger.error("JSON Profil oluşturma hatası", exc_info=True)
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


def _pick_seeds(pool: List[dict], n: int) -> List[dict]:
    """Yeni -> eski sıralı havuzdan, yeniye daha yüksek ağırlıkla n tohum seç (tekrarsız)."""
    pool = list(pool)
    seeds = []
    while pool and len(seeds) < n:
        weights = [1.0 / (1 + i) for i in range(len(pool))]
        seeds.append(pool.pop(random.choices(range(len(pool)), weights=weights)[0]))
    return seeds


async def gather_push_candidates(user_id: int, favs: List[dict], exclude_ids: set, exclude_titles: set,
                                 persona: Optional[dict] = None, liked: Optional[List[dict]] = None,
                                 n_seeds: int = 3, top_n: int = 12) -> List[dict]:
    persona = persona or {}
    avoid = _avoid_genres(persona)

    # Tohum havuzu: beğenilenler + favoriler (her ikisi de yeni -> eski), dönüşümlü ve tekrarsız.
    pool, seen_seed = [], set()
    for s in (x for pair in zip_longest(liked or [], favs or []) for x in pair if x):
        sid = str(s.get("movie_id") or "")
        if s.get("Film") and sid not in seen_seed:
            seen_seed.add(sid)
            pool.append(s)
    seeds = _pick_seeds(pool[:12], n_seeds)

    seed_jobs = []
    for s in seeds:
        args = {"movie_title": s["Film"]}
        if str(s.get("movie_id") or "").isdigit():
            args["movie_id"] = int(s["movie_id"])      # başlıkla arama aynı adlı filmleri karıştırır
        seed_jobs.append(_call_tool_cards("get_similar_movies", args))

    cf_args = {"user_id": user_id, "limit": 15, "min_votes": MIN_VOTES, "max_rating": 10.0}
    if avoid:
        cf_args["exclude_genres"] = ", ".join(sorted(avoid))
    extra_jobs = [_call_tool_cards("get_collaborative_recommendations", cf_args)]

    # Soğuk başlangıç: tohum yoksa profilin en güçlü türlerinden aday çek.
    top_genres = _top_genres(persona, 2)
    if not seeds and top_genres:
        f_args = {"genre_name": ",".join(top_genres), "min_votes": MIN_VOTES, "limit": 20}
        if avoid:
            f_args["exclude_genres"] = ", ".join(sorted(avoid))
        extra_jobs.append(_call_tool_cards("search_movies_by_filters", f_args))

    results = await asyncio.gather(*seed_jobs, *extra_jobs)
    seed_results, extra_results = results[:len(seeds)], results[len(seeds):]

    lists = []
    for seed, cards in zip(seeds, seed_results):
        for c in cards:
            c["_because"] = seed["Film"]          # gerekçe için hangi favori/beğeniden geldi
        lists.append(cards)
    lists.extend(extra_results)                    # collaborative / persona türü: _because yok

    candidates = []
    for c in merge_candidates(lists):
        mid = str(c.get("movie_id") or "")
        title = (c.get("Film") or "").lower().strip()
        if not mid or mid in exclude_ids or title in exclude_titles:   # favori + gösterilenler + dislike
            continue
        if avoid & _card_genres(c):                                      # profilde kaçınılan türler
            continue
        if c.get("vote_count", 0) < MIN_VOTES:
            continue
        if not (c.get("Özet") or "").strip():
            continue
        c["_pscore"] = (c["_rrf"] + 0.015 * (bayes(c.get("TMDB Puanı"), c.get("vote_count", 0)) / 10)
                        + PERSONA_WEIGHT * _persona_fit(c, persona))
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
                                    exclude_ids: set, exclude_titles: set,
                                    liked: Optional[List[dict]] = None) -> Optional[dict]:
    """-> {"movie_id", "title", "message", "reason", "source"} veya None (uygun aday yoksa)."""
    cands = await gather_push_candidates(user_id, favs, exclude_ids, exclude_titles,
                                         persona=_persona_obj(persona), liked=liked)
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
    srcs = " ".join(chosen.get("_sources") or [])
    source = ("similar" if chosen.get("_because")
              else "persona_genre" if "tmdb_filter" in srcs else "collaborative")
    return {"movie_id": str(chosen["movie_id"]), "title": chosen.get("Film"),
            "message": message, "reason": reason, "source": source}


async def _cli_main():
    """Terminal testi: MCP havuzunu kurar, kullanıcı profili destekli ve şeffaf sohbet döngüsünü çalıştırır."""
    import sys
    import time
    from mcp import StdioServerParameters
    from shared import MCPSessionPool

    sample_profiles = {
        "1": {
            "name": "Bilim Kurgu & Nolan Tutkunu (Korku Karşıtı)",
            "data": {
                "genres": {"Science Fiction": 0.95, "Drama": 0.8, "Mystery": 0.75},
                "directors": {"Christopher Nolan": 0.95, "Denis Villeneuve": 0.85},
                "actors": {"Cillian Murphy": 0.85, "Leonardo DiCaprio": 0.8},
                "avoid": ["Horror"],
                "vibe_summary": "Zihin büken felsefi bilim kurgu, derin dram ve gizem seviyor. Korku ögelerinden nefret ediyor."
            }
        },
        "2": {
            "name": "Romantik Komedi & Animasyon Sever (Aksiyon/Savaş Karşıtı)",
            "data": {
                "genres": {"Romance": 0.95, "Comedy": 0.85, "Animation": 0.8},
                "directors": {"Richard Linklater": 0.85, "Woody Allen": 0.75},
                "actors": {"Ryan Gosling": 0.85, "Emma Stone": 0.9},
                "avoid": ["Action", "War"],
                "vibe_summary": "İç ısıtan duygusal hikayeleri, zeki komedileri seviyor. Şiddet ve savaştan uzak duruyor."
            }
        },
        "3": {
            "name": "Karanlık Suç, Neo-Noir & David Fincher Hayranı (Komediden Kaçınır)",
            "data": {
                "genres": {"Crime": 0.95, "Thriller": 0.9, "Mystery": 0.85},
                "directors": {"David Fincher": 0.95, "Martin Scorsese": 0.85},
                "actors": {"Brad Pitt": 0.85, "Jake Gyllenhaal": 0.8},
                "avoid": ["Comedy"],
                "vibe_summary": "Karanlık psikolojik gerilim, neo-noir ve karmaşık suç hikayelerini seviyor."
            }
        }
    }

    current_profile = sample_profiles["1"]["data"]
    current_profile_name = sample_profiles["1"]["name"]
    blocked_ids = set()
    seen_ids = set()

    def show_profile():
        print("\n" + "=" * 70)
        print("👤 AKTİF KULLANICI PROFİLİ:")
        if not current_profile or not any(current_profile.values()):
            print("   (Boş Profil - Profilsiz Genel Öneri Modu)")
        else:
            print(f"   Adı: {current_profile_name}")
            print(f"   Vibe: {current_profile.get('vibe_summary', '-')}")
            print(f"   Sevdiği Türler: {current_profile.get('genres', {})}")
            print(f"   Sevdiği Yönetmenler: {current_profile.get('directors', {})}")
            print(f"   Sevdiği Oyuncular: {current_profile.get('actors', {})}")
            print(f"   Kaçınılanlar (Avoid): {current_profile.get('avoid', [])}")
        if blocked_ids:
            print(f"   Engellenen Film ID'leri: {list(blocked_ids)}")
        if seen_ids:
            print(f"   İzlenmiş / Görülen ID'ler: {list(seen_ids)}")
        print("=" * 70)

    print("\n" + "=" * 70)
    print("🎬 MOVIE & TV RAG SİSTEMİ - PROFİL ETKİSİ KONSOL TESTİ")
    print("=" * 70)
    print("Kullanılabilir Komutlar:")
    print("  /profile        -> Mevcut profili ve kısıtlamaları gösterir")
    print("  /1, /2, /3      -> Hazır test profillerine geçiş yapar")
    print("  /clear          -> Profili temizler (profilsiz modu karşılaştırmak için)")
    print("  /avoid <tür>    -> Kaçınılan tür ekler/çıkarır (örn: /avoid Horror)")
    print("  /block <id>     -> Dislike edilen ID ekler (örn: /block 155)")
    print("  /seen <id>      -> Daha önce görülmüş ID ekler")
    print("  exit / quit     -> Testten çıkar")
    print("=" * 70)

    show_profile()

    pool = MCPSessionPool(pool_size=1)
    print("\n⏳ MCP sunucusu başlatılıyor...")
    await pool.start(StdioServerParameters(command=sys.executable, args=["server.py"]), get_tools)
    ctx.session = pool
    ctx.tools = pool.tools
    print("✅ MCP sunucusu hazır! Sorgularınızı yazabilirsiniz.\n")

    try:
        while True:
            raw_input = await asyncio.to_thread(input, "\n💬 Kullanıcı İsteği: ")
            text = raw_input.strip()
            if not text:
                continue
            if text.lower() in ("exit", "quit"):
                break

            # Komut kontrolleri
            if text == "/profile":
                show_profile()
                continue
            if text in ("/1", "/2", "/3"):
                num = text[1]
                current_profile = sample_profiles[num]["data"]
                current_profile_name = sample_profiles[num]["name"]
                print(f"✅ Profil değiştirildi: {current_profile_name}")
                show_profile()
                continue
            if text in ("/clear", "/empty", "/0"):
                current_profile = {}
                current_profile_name = "Boş Profil"
                print("✅ Profil temizlendi! Şu an profilsiz genel mod aktif.")
                continue
            if text.startswith("/avoid "):
                genre = text.split(" ", 1)[1].strip()
                av = current_profile.setdefault("avoid", [])
                if genre in av:
                    av.remove(genre)
                    print(f"✅ '{genre}' avoid listesinden çıkarıldı.")
                else:
                    av.append(genre)
                    print(f"🚫 '{genre}' avoid listesine eklendi.")
                continue
            if text.startswith("/block "):
                bid = text.split(" ", 1)[1].strip()
                blocked_ids.add(bid)
                print(f"🚫 ID={bid} engellenenler listesine eklendi.")
                continue
            if text.startswith("/seen "):
                sid = text.split(" ", 1)[1].strip()
                seen_ids.add(sid)
                print(f"👁️ ID={sid} görülenler listesine eklendi.")
                continue

            # State hazırla
            persona_json = json.dumps(current_profile, ensure_ascii=False) if current_profile else "{}"
            state: GraphState = {
                "prompt": text,
                "session_id": "console_session",
                "messages": [],
                "persona": persona_json,
                "intent": "",
                "intent_data": {},
                "blocked_ids": list(blocked_ids),
                "seen_ids": list(seen_ids),
            }

            start_t = time.time()
            print(f"\n🚀 SORGULANIYOR: '{text}'")
            if current_profile:
                print(f"👤 Aktif Profil: {current_profile_name}")
                if current_profile.get("avoid"):
                    print(f"🚫 Avoid Filtresi: {current_profile.get('avoid')}")

            try:
                # Düğümleri astream ile adım adım izle
                async for event in app.astream(state):
                    for node_name, state_update in event.items():
                        if node_name == "intent_analyzer":
                            intent = state_update.get("intent")
                            data = state_update.get("intent_data", {})
                            print(f"\n🧠 [NİYET ANALİZİ]: {intent}")
                            if data:
                                print(f"   Parametreler: {json.dumps(data, ensure_ascii=False)}")
                        elif node_name == "recommendation_engine":
                            calls = state_update.get("tool_calls", [])
                            print(f"\n🎬 [ÖNERİ MOTORU]: {len(calls)} araç sorgulandı")
                            for c in calls:
                                print(f"   ➔ {c.get('tool_name')}: {json.dumps(c.get('arguments'), ensure_ascii=False)}")

                        if "final_output" in state_update:
                            raw_out = state_update["final_output"]
                            try:
                                parsed = json.loads(raw_out)
                                print("\n" + "-" * 70)
                                print(f"🤖 ASİSTAN CEVABI:\n{parsed.get('text', '')}")
                                if parsed.get("mood_response"):
                                    print(f"\n🎭 Ruh Hali Tepkisi: {parsed.get('mood_response')}")

                                movies = parsed.get("movies", [])
                                if movies:
                                    print(f"\n🍿 ÖNERİLEN İÇERİKLER ({len(movies)} adet):")
                                    for idx, m in enumerate(movies, 1):
                                        kind = "DİZİ" if m.get("media_type") == "tv" else "FİLM"
                                        pfit = m.get("profil_uyumu")
                                        pfit_str = f" | ⭐ Profil Uyumu: %{int(pfit * 100)}" if pfit is not None and pfit > 0 else ""
                                        print(f"\n   {idx}. [{kind}] {m.get('Film')} ({m.get('Yıl', '?')}) - TMDB: {m.get('TMDB Puanı')}{pfit_str}")
                                        print(f"      Türler: {m.get('Türler', '-')}")
                                        if m.get("Director"):
                                            print(f"      Yönetmen/Yaratıcı: {m.get('Director')}")
                                        if m.get("Cast"):
                                            print(f"      Kadro: {m.get('Cast')}")
                                        print(f"      💡 Neden Önerildi: {m.get('Neden Önerildi', '-')}")
                            except Exception:
                                print(f"\nÇıktı: {raw_out}")

                elapsed = time.time() - start_t
                print(f"\n⏱️ Toplam Süre: {elapsed:.2f} saniye")
                print("=" * 70)
            except Exception as e:
                print(f"\n❌ Çalıştırma hatası: {e}")
                logger.error("Konsol çalıştırma hatası", exc_info=True)
    finally:
        await pool.aclose()


if __name__ == "__main__":
    asyncio.run(_cli_main())