"""
Raporda gördüğümüz hataları (Q12, Q34, Q36, Q43, Q45, mood uydurma) sahte LLM + sahte MCP ile
yeniden üretip yeni hattın düzelttiğini doğrular. Gerçek LangGraph kullanılır; ağ gerekmez.
Çalıştır:  python tests/test_pipeline.py   (proje kökünden)
"""
import asyncio, json, os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- dış bağımlılık taklitleri (logger.py sizde var; NVIDIA istemcisi ağ ister) ---
import logging
sys.modules["logger"] = types.SimpleNamespace(logger=logging.getLogger("test"))
class _Dummy:
    def __init__(self, *a, **k): pass
nv = types.ModuleType("langchain_nvidia_ai_endpoints"); nv.ChatNVIDIA = _Dummy
sys.modules["langchain_nvidia_ai_endpoints"] = nv

import client
from shared import ctx
from langchain_core.messages import AIMessage

# --- sahte LLM ---------------------------------------------------------------
class FakeLLM:
    def __init__(self): self.extract_calls = 0; self.select_calls = 0; self.spec = {}; self.select = None
    async def ainvoke(self, messages):
        system = messages[0].content
        if "sorgu analiz bileşenisin" in system:
            self.extract_calls += 1
            return AIMessage(content=json.dumps(self.spec))
        if "SON SEÇİM" in system:
            self.select_calls += 1
            human = messages[1].content
            ids = [l.split("]")[0].replace("[ID:", "") for l in human.splitlines() if l.startswith("[ID:")]
            out = self.select(ids, human) if callable(self.select) else self.select
            return AIMessage(content=out if isinstance(out, str) else json.dumps(out))
        return AIMessage(content="sohbet")
llm = FakeLLM(); client.aclient = llm

# --- sahte MCP oturumu --------------------------------------------------------
def card(i, title, year, rating, votes, genres, overview="Bir film özeti.", source="t"):
    return {"movie_id": str(i), "Film": title, "Yıl": str(year), "TMDB Puanı": f"{rating:.1f}",
            "vote_count": votes, "Türler": genres, "Özet": overview, "Poster": "", "Fragman": "",
            "Şu Anki Platform(lar)": "", "Neden Önerildi": "", "Director": "", "Cast": "", "source": source}

class FakeSession:
    def __init__(self): self.pool = []; self.calls = []
    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if name == "enrich_movies":
            body = [dict(c, Fragman="https://yt/x", **{"Şu Anki Platform(lar)": "Netflix"})
                    for i in args["movie_ids"] for c in self.pool if c["movie_id"] == str(i)]
        elif name == "get_movie_card":
            body = [dict(self.pool[0], Director="X")] if self.pool else []
        else:
            body = self.pool
        return types.SimpleNamespace(content=[types.SimpleNamespace(text=json.dumps(body))])
sess = FakeSession(); ctx.session = sess

async def ask(prompt, spec, pool, select):
    llm.extract_calls = llm.select_calls = 0
    llm.spec, llm.select, sess.pool, sess.calls = spec, select, pool, []
    out = await client.app.ainvoke({"prompt": prompt, "persona": "", "messages": [], "session_id": None})
    return json.loads(out["final_output"]) if out["final_output"].startswith("{") else out["final_output"]

def titles(r): return [m["Film"] for m in r["movies"]]

async def main():
    # (1) Q36: "TMDB < 5" -> Ed Wood (7.5) kodla elenir; LLM elenmiş id'yi seçmeye çalışsa bile giremez
    pool = [card(522,"Ed Wood",1994,7.5,3000,"Comedy, Drama"), card(10,"Plan 9",1959,4.0,900,"Horror, Science Fiction"),
            card(11,"The Room",2003,3.7,2000,"Drama, Romance")]
    r = await ask("kötü ama kült film", {"intent":"recommendation","max_rating":5,"obscure":True},
                  pool, lambda ids, h: {"text":"t","ranked":[{"id":"522","reason":"x"},{"id":"10","reason":"a"},{"id":"11","reason":"b"}]})
    assert titles(r) == ["Plan 9", "The Room"], titles(r)
    print("OK Q36  max_rating kodda uygulandı ->", titles(r))

    # (2) Q12: "korku olmasın" -> Horror etiketli film LLM'e hiç gitmez
    pool = [card(747,"Shaun of the Dead",2004,7.5,5000,"Horror, Comedy"), card(1,"Zombieland",2009,7.3,9000,"Comedy, Horror"),
            card(2,"Warm Bodies",2013,6.8,4000,"Comedy, Romance, Fantasy")]
    seen = {}
    def sel(ids, h): seen["ids"] = ids; return {"text":"t","ranked":[{"id":i,"reason":"r"} for i in ids]}
    r = await ask("zombili ama korku değil", {"intent":"recommendation","exclude_genres":["Horror"]}, pool, sel)
    assert seen["ids"] == ["2"] and titles(r) == ["Warm Bodies"], (seen, titles(r))
    print("OK Q12  exclude_genres kodda uygulandı ->", titles(r))

    # (3) Q43/Q34: sıra LLM'in sırası olmalı, araç dönüş sırası değil
    pool = [card(1,"Inferno",2016,6.1,3000,"Mystery, Thriller"), card(2,"Chuck",2025,7.3,900,"Drama"),
            card(77,"Memento",2000,8.2,14000,"Mystery, Thriller")]
    r = await ask("tersten anlatılan hafıza kaybı filmi", {"intent":"recommendation"}, pool,
                  lambda ids, h: {"text":"t","ranked":[{"id":"77","reason":"a"},{"id":"1","reason":"b"}]})
    assert titles(r) == ["Memento", "Inferno"], titles(r)
    print("OK Q43  LLM sırası korundu ->", titles(r))

    # (4) mood söylenmediyse LLM uydursa bile gösterilmez; söylendiyse gösterilir
    sel_mood = lambda ids, h: {"text":"t","mood_response":"Kasvetin içinde kaybolmuş hissettiğin...","ranked":[{"id":ids[0],"reason":"r"}]}
    r = await ask("deniz fenerinde korku", {"intent":"recommendation","mood":None}, pool, sel_mood)
    assert r["mood_response"] == "", r["mood_response"]
    r = await ask("canım çok sıkkın komedi", {"intent":"recommendation","mood":"üzgün"}, pool, sel_mood)
    assert r["mood_response"] != ""
    print("OK mood  yalnızca açıkça söylenmişse dolu")

    # (5) Q45: uygun aday yok -> boş liste + dürüst mesaj (dolgu yok)
    r = await ask("60'larda geçen politik müzikal", {"intent":"recommendation"}, pool,
                  {"text":"Bu kriterlere uyan film bulamadım.","no_exact_match":True,"ranked":[]})
    assert r["movies"] == [] and "bulamadım" in r["text"], r
    print("OK Q45  uygun aday yoksa dolgu yapılmadı")

    # (6) uydurma / tekrar eden id'ler atılır
    r = await ask("x", {"intent":"recommendation"}, pool,
                  {"text":"t","ranked":[{"id":"999","reason":"uydurma"},{"id":"77","reason":"a"},{"id":"77","reason":"tekrar"}]})
    assert titles(r) == ["Memento"], titles(r)
    print("OK      uydurma ve tekrar eden id'ler elendi")

    # (7) LLM çıktısı bozuksa skor sırasıyla ilk 3 (çökme yok)
    r = await ask("x", {"intent":"recommendation"}, pool, "bu json değil")
    assert len(r["movies"]) == 3 and "yakın" in r["text"], r
    print("OK      bozuk LLM çıktısında güvenli geri dönüş")

    # (8) intent_data state'te taşınıyor: çıkarım 1 kez, seçim 1 kez (eskiden çıkarım 2 kez yapılıyordu)
    await ask("x", {"intent":"recommendation"}, pool, lambda ids, h: {"text":"t","ranked":[{"id":ids[0],"reason":"r"}]})
    assert (llm.extract_calls, llm.select_calls) == (1, 1), (llm.extract_calls, llm.select_calls)
    print("OK      LLM çağrısı: çıkarım=1, seçim=1 (eski: çıkarım=2 + ajan döngüsü + writer)")

    # (9) araçlar paralel + sadece seçilenler zenginleştirildi
    await ask("Inception gibi", {"intent":"recommendation","reference_movie":"Inception","semantic_query_en":"dream heist"},
              pool, lambda ids, h: {"text":"t","ranked":[{"id":"77","reason":"r"}]})
    names = [n for n, _ in sess.calls]
    assert names.count("enrich_movies") == 1 and sess.calls[-1][1]["movie_ids"] == [77], sess.calls
    assert {"get_similar_movies", "search_movies_semantically"} <= set(names), names
    print("OK      retrieval araçları:", [n for n in names if n != "enrich_movies"], "| enrich:", sess.calls[-1][1])

    # (10) film bilgisi sorusu ajan hattına girmeden get_movie_card'a gider
    r = await ask("Matrix'in oyuncuları kimler?", {"intent":"movie_info","movie_title":"The Matrix"}, pool, None)
    assert sess.calls[0][0] == "get_movie_card" and llm.select_calls == 0 and r["mood_response"] == ""
    print("OK      movie_info rotası: get_movie_card, seçim LLM'i yok, mood yok")

    # (11) çıkarım JSON'u bozuk/hatalı tiplerle gelse de çökmez
    for bad in ('{"intent":"recommendation","min_rating":"null","include_genres":"korku, bilim kurgu","obscure":"false"}',
                "düz metin"):
        spec = client.parse_spec(bad)
        assert spec.intent == "recommendation" and spec.obscure is False
    assert client.parse_spec('{"include_genres":"korku, bilim kurgu"}').include_genres == ["Horror", "Science Fiction"]
    print("OK      bozuk/eksik çıkarım JSON'u güvenle ayrıştırılıyor")

asyncio.run(main())
print("\nTÜM TESTLER GEÇTİ")
