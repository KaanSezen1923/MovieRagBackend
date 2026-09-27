import fetch_movies
import json
import re
import os
import operator
from typing import TypedDict, Annotated, List

import ollama
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from logger import logger
from shared import ctx, chat_statuses

load_dotenv()

MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma4:31b-cloud")
host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
aclient = ollama.AsyncClient(host=host)


class GraphState(TypedDict):
    prompt: str
    persona: str
    intent: str
    messages: Annotated[List[dict], operator.add]
    final_output: str
    tools: List[dict]
    tool_calls: list
    tool_results: list
    session_id: str


async def analyze_intent_node(state: GraphState):
    session_id = state.get("session_id")
    if session_id:
        chat_statuses[session_id] = "🔍 Sorgu analiz ediliyor..."

    # 1. Intent ve Reasoning BİRLEŞİK Prompt
    system_msg = f"""
Sen bir film botu analiz bileşenisin.
Kullanıcı mesajını inceleyerek SADECE aşağıdaki formatta bir JSON döndür. 
Eğer kullanıcı sadece selam veriyorsa veya film önerisiyle ilgisizse "intent": "general" yap, diğer alanları boş bırak.
Eğer film/dizi önerisi veya detayı istiyorsa "intent": "recommendation" yap ve diğer alanları doldur.

Kullanıcı Personası: {state.get('persona', '')}

ÖNEMLİ KURAL: Duygu kelimelerini 'keyword' olarak değil, 'genre' (örn: karanlık için Thriller/Horror) olarak değerlendir.

Şema:
{{
  "intent": "general VEYA recommendation",
  "mood": "kullanıcının ruh hali (örn: üzgün, heyecanlı, nostaljik, keyifli), yoksa null",
  "genre": "en uygun TMDB türü/türleri (Action, Comedy, Drama vb.), yoksa null",
  "keyword": "net tematik nesneler (ai, space, alien vb.), yoksa null",
  "reference_movie": "beğenilen/sorulan spesifik film adı, yoksa null",
  "avoid": "istenmeyen tür/konu, yoksa null",
  "reasoning": "kararlarının kısa gerekçesi"
}}
"""
    response = await aclient.chat(
        model=MODEL_NAME,
        messages=[
            {'role': 'system', 'content': system_msg},
            {'role': 'user', 'content': state['prompt']},
        ]
    )
    raw_content = response['message']['content']
    try:
        clean = re.sub(r'```json\s?|```', '', raw_content).strip()
        m = re.search(r'(\{.*\})', clean, re.DOTALL)
        intent_data = json.loads(m.group(1)) if m else json.loads(clean)
    except Exception:
        intent_data = {"intent": "recommendation"} # Güvenlik ağı

    intent = intent_data.get("intent", "recommendation").lower()
    return {
        "intent": "general" if "general" in intent else "recommendation", 
        "intent_data": intent_data
    }


async def recommendation_node(state: GraphState):
    session_id = state.get("session_id")
    if session_id:
        chat_statuses[session_id] = "🧠 Öneri motoru hazırlanıyor..."

    model_name = MODEL_NAME
    MAX_TURNS = 3
    MAX_MOVIES = 5

    # Intent node'dan gelen veriyi doğrudan kullan (Ekstra LLM çağrısı KALDIRILDI)
    intent_data = state.get('intent_data', {})
    
    mood_hint     = intent_data.get('mood') or ''
    avoid_hint    = intent_data.get('avoid') or ''
    reasoning_txt = intent_data.get('reasoning') or ''
    reference_movie = intent_data.get('reference_movie') or ''

    avoid_clause     = f"\n- Kaçın: {avoid_hint}" if avoid_hint else ""
    mood_clause      = f"\n- Kullanıcı şu an '{mood_hint}' hissediyor." if mood_hint else ""
    reference_clause = f"\n- Kullanıcı '{reference_movie}' filmini beğendi; benzerlerini ara." if reference_movie else ""

    # ── 1. REASONING KATMANI ──────────────────────────────────────────────────
    reasoning_system = f"""
Sen bir film öneri motorunun analiz bileşenisin.
Kullanıcı mesajını ve sohbet geçmişini inceleyerek aşağıdaki JSON şemasını doldur.
Çıktı SADECE JSON olacak, başka açıklama olmayacak.

Kullanıcı Personası: {state['persona']}

ÖNEMLİ KURAL: Kullanıcı "karanlık, kasvetli, ağır, psikolojik, heyecanlı" gibi duygu/atmosfer kelimeleri kullanırsa, bunları ASLA 'keyword' olarak yazma! Bunun yerine bunları ilgili türe dönüştür (Örn: karanlık/kasvetli için 'genre' alanına "Thriller", "Horror" veya "Mystery" ekle). Keyword alanı sadece çok net tematik nesneler/kavramlar içindir (örn: time travel, ai, zombies, aliens, space).

Şema:
{{
  "mood": "kullanıcının ruh hali (örn: üzgün, heyecanlı, nostaljik, keyifli, ...)",
  "genre": "en uygun TMDB türü/türleri — şu listeden seç: Action, Adventure, Animation, Comedy, Crime, Documentary, Drama, Family, Fantasy, History, Horror, Music, Mystery, Romance, Science Fiction, Thriller, War, Western. Birden fazla tür varsa virgülle ayır: 'Science Fiction,Action'. Tür yoksa null.",
  "director_name": "açıkça belirtilmişse yönetmen adı, yoksa null",
  "actor_name": "açıkça belirtilmişse oyuncu adı, yoksa null",
  "keyword": "sadece net tematik nesne/kavramlar (örn: time travel, ai, alien). Duygu veya atmosfer kelimelerini buraya YAZMA, yoksa null.",
  "reference_movie": "kullanıcının açıkça beğendiğini belirttiği spesifik bir film adı varsa buraya İngilizce/orijinal adıyla yaz, yoksa null",
  "min_rating": "minimum IMDb puanı 0.0-10.0 arasında float, belirtilmemişse 6.5",
  "avoid": "kullanıcının istemediği tür veya konu varsa buraya yaz, yoksa null",
  "reasoning": "kararlarının kısa gerekçesi (1 cümle)"
}}
"""
    reasoning_resp = await aclient.chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': reasoning_system},
            {'role': 'user',   'content': state['prompt']},
        ]
    )
    raw_reasoning = reasoning_resp['message']['content']

    try:
        clean_r = re.sub(r'```json\s?|```', '', raw_reasoning).strip()
        m = re.search(r'(\{.*\})', clean_r, re.DOTALL)
        intent_data: dict = json.loads(m.group(1)) if m else {}
    except Exception:
        intent_data = {}

    # ── 2. ANA SYSTEM MESAJI ─────────────────────────────────────────────────
    mood_hint     = intent_data.get('mood') or ''
    avoid_hint    = intent_data.get('avoid') or ''
    reasoning_txt = intent_data.get('reasoning') or ''
    reference_movie = intent_data.get('reference_movie') or ''

    avoid_clause     = f"\n- Kaçın: {avoid_hint}" if avoid_hint else ""
    mood_clause      = f"\n- Kullanıcı şu an '{mood_hint}' hissediyor." if mood_hint else ""
    reference_clause = f"\n- Kullanıcı '{reference_movie}' filmini beğendi; benzerlerini ara." if reference_movie else ""

    system_msg = f"""
Sen deneyimli, empatik bir film danışmanısın. Elinde iki kaynak var:
1. `search_movies_in_graph`: Yerel hızlı graf veritabanı.
2. `search_movies_by_filters` / `get_similar_movies`: TMDB Canlı API.

GÖREV VE KURALLAR:
- ASLA aynı anda hem yerel grafı hem de TMDB filtrelerini çağırma. Önce `search_movies_in_graph` aracını dene.
- Eğer yerel graf aracı boş sonuç (`[]`) döndürürse, o zaman TMDB araçlarına geçiş yapabilirsin.
- Kullanıcı adı belli tek bir filmin detayını soruyorsa `get_movie_card` aracını kullan.
- Araç çağırdıktan sonra sonuçları kendin yazmana gerek yok; film kartları sistem tarafından oluşturulur.{avoid_clause}{mood_clause}{reference_clause}

Kullanıcı Personası  : {state['persona']}
Anlık Ruh Hali Analizi: {mood_hint or 'belirtilmedi'}
Reasoning            : {reasoning_txt or '-'}
"""

    # ── 3. AGENTIC DÖNGÜ ─────────────────────────────────────────────────────
    current_messages = [{'role': 'system', 'content': system_msg}] + state['messages']
    tool_calls_log, tool_results_log, collected = [], [], []

    def _cards(raw: str) -> list:
        try:
            data = json.loads(raw)
            return [x for x in data if isinstance(x, dict)] if isinstance(data, list) else []
        except Exception:
            return []

    def _brief(cards: list, with_id: bool = False) -> str:
        def _line(c):
            prefix = f"[ID:{c.get('movie_id', '')}] " if with_id else ""
            return f"- {prefix}{c.get('Film', '')} ({c.get('Yıl', '')}): {c.get('Türler', '')} — {c.get('Özet', '')[:120]}"
        return "\n".join(_line(c) for c in cards)

    async def run_tool(name, args, tag=None):
        try:
            result = await ctx.session.call_tool(name, args)
            raw = result.content[0].text
        except Exception as e:
            raw = f"Araç hatası: {e}"
        tool_calls_log.append({"tool_name": tag or name, "arguments": args})
        tool_results_log.append({"tool_name": tag or name, "raw_result": raw})
        cards = _cards(raw)
        collected.extend(cards)
        return _brief(cards) if cards else raw  # LLM'e kısa özet ver

    for _ in range(MAX_TURNS):
        response = await aclient.chat(
            model=model_name, messages=current_messages,
            tools=state['tools'], options={"temperature": 0.4},
        )
        message = response['message']
        if not message.get('tool_calls'):
            break
        if session_id:
            chat_statuses[session_id] = "🎬 Hibrit motor verileri topluyor..."
        current_messages.append(message)
        for tc in message['tool_calls']:
            name, args = tc['function']['name'], tc['function']['arguments']
            content = await run_tool(name, args)
            current_messages.append({'role': 'tool', 'content': content, 'name': name})

    # ── 4. FALLBACK: hiç kart toplanamadıysa ────────────────────────────────
    if not collected and state['tools']:
        ref_movie = intent_data.get('reference_movie')
        if ref_movie:
            name, args = 'get_similar_movies', {"movie_title": ref_movie}
        elif intent_data.get('genre'):
            name, args = 'search_movies_by_filters', {"genre_name": intent_data['genre'], "min_rating": 6.0}
        else:
            name, args = 'search_movies_by_filters', {"min_rating": 7.0}
        if session_id:
            chat_statuses[session_id] = "🎬 TMDB Canlı Yedek Arama Devrede..."
        await run_tool(name, args, tag=f"{name}_fallback")

    # ── 5. KARTLAR KODDAN, METİN LLM'DEN ────────────────────────────────────
    # ── 5. KARTLAR KODDAN, METİN LLM'DEN (HAKEMLİ FİLTRELEME) ─────────────────
    seen, all_movies = set(), []
    for c in collected:
        key = c.get("Film", "").lower()
        if key and key not in seen:
            seen.add(key)
            all_movies.append(c)
            
    # LLM'in çok fazla token ile boğulmaması için maksimum 15 film gönderin
    all_movies = all_movies[:15] 

    text = ("Aradığın kriterlere uygun film bulunamadı. "
            "Farklı bir tür veya oyuncu denemek ister misin?")
    mood_response, reasons = "", {}
    final_movies = []

    if all_movies:
        text = "İşte senin için seçtiklerim:"
        if session_id:
            chat_statuses[session_id] = "✍️ En iyi sonuçlar seçiliyor ve yanıt oluşturuluyor..."

        if mood_hint:
            mood_field_schema = '  "mood_response": "kullanıcının ruh haline özel 1 cümlelik empati notu",\n'
            mood_instruction = f"Ruh hali: {mood_hint}"
        else:
            mood_field_schema = ''
            mood_instruction = "Ruh hali: belirtilmedi (mood_response alanını ekleme)"

        try:
            writer = await aclient.chat(
                model=model_name,
                messages=[
                    {'role': 'system', 'content':
                        'Empatik bir film danışmanısın. Sana sunulan filmler arasından kullanıcının isteğine EN ÇOK UYAN maksimum 5 tanesini seç.\n'
                        'SADECE şu JSON formatını doldur ve başka hiçbir açıklama, giriş metni veya markdown (```) ekleme:\n'
                        '{\n'
                        '  "text": "samimi 1-2 cümlelik giriş",\n'
                        f'{mood_field_schema}'
                        '  "reasons": {"<seçtiğin filmin köşeli parantez içindeki ID\'si, aynen>": "1 cümlelik öneri gerekçesi"}\n'
                        '}\n'
                        'ÖNEMLİ KURAL: reasons sözlüğüne SADECE seçtiğin (en fazla 5) filmi ekle. İlgisiz olanları yoksay. Yeni film uydurma.'},
                    {'role': 'user', 'content':
                        f"İstek: {state['prompt']}\n{mood_instruction}\nFilmler:\n{_brief(all_movies, with_id=True)}"},
                ],
            )
            
            raw_w_content = writer['message']['content'].strip()
            clean_w = re.sub(r'```json\s?|```', '', raw_w_content).strip()
            match_w = re.search(r'(\{.*\})', clean_w, re.DOTALL)
            w = json.loads(match_w.group(1)) if match_w else json.loads(clean_w)
            
            text = w.get("text") or text
            mood_response = w.get("mood_response", "") if mood_hint else ""
            reasons = w.get("reasons", {}) or {}
            
        except Exception as e:
            logger.error("Metin üretimi hatası - varsayılan metne geçildi", exc_info=True)
            mood_response = ""
            reasons = {}

        # LLM'in seçtiği (reasons sözlüğünde anahtarı bulunan) filmleri filtrele
        for m in all_movies:
            m_id = m.get("movie_id", "")
            m_title = m.get("Film", "")
            
            reason = reasons.get(m_id) or reasons.get(m_title)
            if reason:
                m["Neden Önerildi"] = reason
                final_movies.append(m)
                if len(final_movies) == MAX_MOVIES: # Maksimum 5 filme ulaşıldığında dur
                    break
        
        # Fallback: Eğer LLM JSON formatını bozup hiçbir film seçemediyse mecburen ilk 5'i al
        if not final_movies:
            final_movies = all_movies[:MAX_MOVIES]

    return {
        "final_output": json.dumps(
            {"type": "movie_list", "text": text, "mood_response": mood_response, "movies": final_movies},
            ensure_ascii=False),
        "tool_calls": tool_calls_log,
        "tool_results": tool_results_log,
    }


async def general_chat_node(state: GraphState):
    session_id = state.get("session_id")
    if session_id:
        chat_statuses[session_id] = "✍️ Yanıt oluşturuluyor..."
    response = await aclient.chat(model=MODEL_NAME, messages=state['messages'])
    return {"final_output": response['message']['content']}


# --- Grafik Kurulumu ---
def route_by_intent(state: GraphState):
    return "recommendation_engine" if state["intent"] == "recommendation" else "general_chatter"


workflow = StateGraph(GraphState)
workflow.add_node("intent_analyzer", analyze_intent_node)
workflow.add_node("recommendation_engine", recommendation_node)
workflow.add_node("general_chatter", general_chat_node)

workflow.set_entry_point("intent_analyzer")
workflow.add_conditional_edges("intent_analyzer", route_by_intent)
workflow.add_edge("recommendation_engine", END)
workflow.add_edge("general_chatter", END)

app = workflow.compile()


async def get_ollama_tools(session):
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


async def generate_user_profile(chat_history_text, favorites_text):
    model_name = MODEL_NAME
    
    profiler_instructions = """
    Sen uzman bir kullanıcı deneyimi analistisin. 
    Sana verilen "Favori Listesi" (kullanıcının açıkça beğendiğini belirttikleri) ve 
    "Sohbet Geçmişi" (kullanıcının doğal etkileşimleri) verilerini birleştirerek derinlikli bir Persona Özeti oluştur.

    Analizinde şu hiyerarşiyi izle:
    1. Temel İlgi Alanları: Favori listesindeki film, yönetmen ve türler.
    2. Davranışsal Analiz: Sohbet geçmişinden anlaşılan güncel ruh hali ve tercih değişimleri.
    3. Kaçınılanlar: Sevmediği veya ilgilenmediği belirtilen içerikler.
    4. İletişim Tonu: Kullanıcının dil kullanımı (resmi, samimi, kısa, detaycı).

    Çıktı Kuralları:
    - Üçüncü şahıs ağzından yaz.
    - Maksimum 3-4 cümle ile net bir profil çiz.
    - "Kullanıcı..." diye başla.
    """

    user_input = f"""
    KULLANICI FAVORİLERİ:
    {favorites_text}

    SOHBET GEÇMİŞİ:
    {chat_history_text}
    """

    response = await aclient.chat(
        model=model_name,
        messages=[
            {'role': 'system', 'content': profiler_instructions},
            {'role': 'user', 'content': user_input}
        ]
    )
    return response['message']['content']

async def generate_user_suggestions(chat_history_text, favorites_text):
    model_name = MODEL_NAME
    
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
        response = await aclient.chat(
            model=model_name,
            messages=[
                {'role': 'system', 'content': prompt_instructions},
                {'role': 'user', 'content': user_input}
            ]
        )
        content = response['message']['content'].strip()
        clean_content = re.sub(r'```json\s?|```', '', content).strip()
        match = re.search(r'(\[.*\])', clean_content, re.DOTALL)
        if match:
            suggestions = json.loads(match.group(1))
            if isinstance(suggestions, list) and len(suggestions) >= 3:
                return suggestions[:3]
    except Exception as e:
        logger.error("Öneri oluşturma hatası", exc_info=True)
    
    return ["Bilim kurgu filmi öner", "Nolan filmi öner", "Tim Burton filmi öner"]


async def generate_push_message(persona: str, fav_titles: list):
    """Push bildirimi metni ve önerilen film adını üretir -> (message, movie_title)."""
    system_prompt = """Sen heyecanlı ve samimi bir film danışmanısın.
    Kullanıcının personasına ve favori filmlerine bakarak, ona izlemesi için *rastgele ve ilgi çekici* kısa bir bildirim mesajı (maksimum 150 karakter) yaz ve önerdiğin spesifik filmin tam adını belirt.

    Çıktıyı SADECE aşağıdaki JSON formatında ver, başka hiçbir metin veya açıklama ekleme:
    {
      "message": "En son Inception'ı sevmiştin, tam senin tarzına göre akıl bükücü bir film buldum: Shutter Island! Bakmak ister miydin?",
      "movie_title": "Shutter Island"
    }
    """
    user_msg = f"Persona: {persona}\nFavoriler: {', '.join(fav_titles)}"
    try:
        response = await aclient.chat(
            model=MODEL_NAME,
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_msg},
            ],
        )
        content = response['message']['content'].strip()
        clean = re.sub(r'```json\s?|```', '', content).strip()
        m = re.search(r'(\{.*\})', clean, re.DOTALL)
        data = json.loads(m.group(1)) if m else json.loads(clean)
        return data.get("message"), data.get("movie_title")
    except Exception as e:
        logger.error("Push bildirimi üretme hatası", exc_info=True)
        return "Senin için yepyeni film önerilerim var, keşfetmek için dokun! 🍿", None