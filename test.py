import asyncio
import json
import os
import sys
import time
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from client import app, get_ollama_tools
from shared import ctx

# 10 Kolay, 10 Orta, 30 Zor olmak üzere toplam 50 test sorusu
sorular = [
    # ---------------- KOLAY SEVİYE (10 Soru) ----------------
    # Sistemin temel graf (Neo4j) ve API (TMDB) araçlarını doğrudan test eder.
    "Bilim kurgu filmi öner.",
    "Christopher Nolan'ın yönettiği filmler neler?",
    "Bana bir komedi filmi bul.",
    "Inception filmine benzer filmler tavsiye et.",
    "Brad Pitt'in oynadığı aksiyon filmleri.",
    "The Matrix'in oyuncu kadrosunda kimler var?",
    "Ailece izlenecek bir animasyon filmi arıyorum.",
    "Korku türünde en popüler filmler hangileri?",
    "Bana romantik bir film tavsiye eder misin?",
    "Tarihi belgesel arıyorum, ne izleyebilirim?",

    # ---------------- ORTA SEVİYE (10 Soru) ----------------
    # Niyet (intent) analizini, ruh hali (mood) eşleştirmesini ve filtre kombinasyonlarını test eder.
    "Bugün canım çok sıkkın, beni hemen neşelendirecek ve çok düşündürmeyecek bir komedi öner.",
    "İçinde zombiler olan ama kesinlikle korku türünde olmayan, daha çok komedi ağırlıklı bir film var mı?",
    "Sadece 8 puanın üzerindeki Leonardo DiCaprio'nun gizem veya gerilim filmlerini listele.",
    "Uzayda geçen ve yalnızlık teması üzerine kurulu derin bir dram arıyorum.",
    "Hem bilim kurgu hem de western türlerini harmanlayan, aksiyon dozu yüksek bir film öner.",
    "Quentin Tarantino'nun tarzına benzeyen ama onun yönetmediği suç filmleri nelerdir?",
    "Dün gece Interstellar izleyip büyülendim, bana tam olarak onun atmosferinde felsefi bir uzay filmi lazım.",
    "İçinde yapay zeka geçen ama aksiyon barındırmayan, tamamen psikolojik odaklı filmler.",
    "İzledikten sonra insanın içini ısıtan (feel-good) romantik komediler öner.",
    "Gerçek olaylara dayanan, 2. Dünya Savaşı'nda geçen ve IMDB puanı 7'den yüksek savaş dramaları.",

    # ---------------- ZOR SEVİYE (30 Soru) ----------------
    # 'search_movies_semantically' vektör aramasını, Neo4j fallback mekanizmasını, soyut konseptleri, 
    # persona çakışmalarını ve 'avoid' (kaçınma) parametresindeki zorlu sınırları test eder.
    "Varoluşsal bir kriz yaşayan ama aynı zamanda seyirciyi kahkahalara boğan absürt bir kara komedi arıyorum.",
    "İzledikten sonra günlerce tavanı izletip hayatı sorgulatacak, İskandinav sinemasına benzeyen soğuk atmosferli felsefi bir film.",
    "İçinde zamanda yolculuk olsun ama kelebek etkisi yüzünden her şeyin mahvolduğu ve mutlu sonla bitmeyen karanlık bir film.",
    "Hiç diyalog olmayan veya çok az konuşulan, derdini tamamen görsel anlatım ve renk paletiyle anlatan bir bilim kurgu.",
    "Başrolünde Christian Bale olsun ama film asla bir aksiyon veya süper kahraman filmi olmasın, sadece ağır bir dram olsun.",
    "Siberpunk bir evrende geçen, yapay zekanın insanlığı yok etmek yerine onlarla din ve tanrı üzerine tartıştığı bir senaryo.",
    "Dışarıda yağmur yağıyor, elimde kahvem var; bana yavaş tempolu, bol diyaloglu, gotik bir kütüphanede geçen gizem filmi öner.",
    "Ana karakterin aslında filmin başından beri ölü veya bir simülasyonda olduğu, izleyiciyi kandıran akıl bükücü (mind-bending) bir kurgu.",
    "Sadece tek bir odanın içinde geçen ve karakterlerin arasındaki diyalogla gerilimi zirveye taşıyan klostrofobik bir film.",
    "Ne çok sanat filmi kadar ağır olsun ne de Hollywood aksiyonları kadar aptalca; tam ortasında, zekice kurgulanmış bir banka soygunu.",
    "Renk teorisinin hikaye anlatımında kullanıldığı, sadece kırmızı ve siyah renklerin ağırlıkta olduğu bir intikam filmi.",
    "Bir aşk hikayesi anlatsın ama romantik filmlerin hiçbir klişesini barındırmasın, filmin sonunda karakterler birbirlerinden nefret etsin.",
    "Müzikleri Hans Zimmer'a ait olmayan ama onun tarzında çok destansı (epik) müziklere sahip olan bir orta çağ filmi arıyorum.",
    "İçinde dev canavarlar (kaiju) olsun ama asıl odak noktası canavarlar değil, hükümetlerin bürokratik beceriksizliği olan satirik bir yapım.",
    "Kurgu mu (mockumentary) yoksa gerçek bir belgesel mi olduğunu filmin en sonuna kadar anlayamayacağımız tuhaf bir yapım.",
    "TMDB puanı 5'in altında olan ama sinema tarihinde gizli bir başyapıt (cult classic) kabul edilen, çöp (trash) sinema örneği bir film.",
    "Hem 1920'lerin caz çağında geçsin hem de içinde Lovecraft tarzı kozmik korku ögeleri barındıran bir dedektiflik hikayesi olsun.",
    "Hiç tanınmayan oyuncuların oynadığı, bütçesi inanılmaz düşük ama senaryosuyla gişe rekorları kıran filmlere taş çıkartan bağımsız bir bilim kurgu.",
    "İnsanlığın sonu gelmiş ama bu durumun son derece umutlu, renkli ve neşeli bir dille anlatıldığı distopik bir film.",
    "İki farklı zaman diliminde paralel ilerleyen, 1950'lerdeki bir olayın günümüzdeki bir cinayeti çözdüğü bulmaca kurgulu bir neo-noir.",
    "Kötü karakterin (antagonist) felsefi olarak tamamen haklı olduğu ve filmin sonunda kahramanın kaybettiği ahlaki açıdan gri bir suç filmi.",
    "Başrol oyuncusunun aynı zamanda filmin yönetmeni olduğu, Avrupa sinemasından bol ödüllü, diyalogsuz bir psikolojik gerilim.",
    "Olayların kronolojik olarak sondan başa doğru anlatıldığı, hafıza kaybı temalı ve izleyiciyi sürekli şüphede bırakan bir yapım.",
    "Uzaylı istilası olsun ama uzaylıları film boyunca ekranda hiç görmediğimiz, sadece insanların paranoyasını izlediğimiz bir psikolojik korku.",
    "Sadece müzik ve dans koreografileriyle ağır bir politik sistem eleştirisi yapan, 1960'larda geçen bir müzikal drama.",
    "Karın altında, dış dünyadan izole bir yerde mahsur kalmış bir grup insanın yavaş yavaş delirdiği hayatta kalma (survival) gerilimi.",
    "İçinde 'Matrix' veya 'Simülasyon' kelimesi geçmeyen ama gerçekliğin sahte olduğu hissini Matrix'ten çok daha bilimsel bir dille işleyen yapım.",
    "Romantik komedi gibi başlayıp ilk yarının sonunda aniden inanılmaz vahşi ve kanlı bir uzaylı istilasına dönüşen dengesiz bir film.",
    "İzleyicinin dördüncü duvarı yıkarak sürekli karakterle konuştuğu, meta-anlatı (metafiction) özelliklerine sahip absürt macera.",
    "Sadece 1800'lü yıllarda, denizin ortasındaki bir fenerde geçen, denizci mitolojisiyle insanın aklını yitirmesini anlatan siyah beyaz bir korku."
]

async def run_tests():
    test_sonuclari = []
    output_file = "test_raporu_client_nano3.json"
    
    server_params = StdioServerParameters(
        command=sys.executable,
        args=["server.py"],
        env=os.environ.copy()
    )

    print(f"🎬 MCP Sunucusu başlatılıyor ve {len(sorular)} test sorusu client mimarisi ile koşturuluyor...\n")

    async with AsyncExitStack() as stack:
        stdio_transport = await stack.enter_async_context(stdio_client(server_params))
        read, write = stdio_transport
        session = await stack.enter_async_context(ClientSession(read, write))
        await session.initialize()

        # client.py içerisindeki ctx oturumunu bağla ve tool listesini çek
        ctx.session = session
        tools = await get_ollama_tools(session)

        for index, soru in enumerate(sorular, 1):
            print(f"[{index}/{len(sorular)}] Test ediliyor: {soru}")

            initial_state = {
                "prompt": soru,
                "persona": "Sinema tutkunu, kaliteli film önerileri seven meraklı bir kullanıcı.",
                "intent": "",
                "messages": [{"role": "user", "content": soru}],
                "final_output": "",
                "tools": tools,
                "tool_calls": [],
                "tool_results": [],
                "session_id": f"test_session_{index}"
            }

            start_time = time.time()
            kategori = None
            arama_degeri = soru
            calisan_dugumler = []
            db_cevabi = []
            final_answer = None

            try:
                # LangGraph asenkron akışı üzerinden düğüm durumlarını topla
                async for event in app.astream(initial_state):
                    for node_name, node_state in event.items():
                        calisan_dugumler.append(node_name)

                        if node_name == "intent_analyzer":
                            kategori = node_state.get("intent")
                        elif node_name == "recommendation_engine":
                            final_answer = node_state.get("final_output")
                            db_cevabi = node_state.get("tool_results", [])
                        elif node_name == "general_chatter":
                            final_answer = node_state.get("final_output")

                end_time = time.time()
                gecen_sure = round(end_time - start_time, 2)

                test_sonuclari.append({
                    "soru": soru,
                    "tespit_edilen_kategori": kategori,
                    "arama_degeri": arama_degeri,
                    "calisan_dugumler": calisan_dugumler,
                    "veritabani_cevabi": db_cevabi,
                    "sistem_yaniti": final_answer,
                    "yanit_suresi_saniye": gecen_sure
                })
                print(f"  -> Başarılı ({gecen_sure}s)")

            except Exception as e:
                print(f"  ❌ Hata oluştu: {str(e)}")
                test_sonuclari.append({
                    "soru": soru,
                    "hata": str(e)
                })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_sonuclari, f, ensure_ascii=False, indent=4)

    print(f"\n✅ Tüm testler tamamlandı! Sonuçlar '{output_file}' dosyasına kaydedildi.")

if __name__ == "__main__":
    asyncio.run(run_tests())