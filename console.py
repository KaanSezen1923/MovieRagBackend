import asyncio
import time
import json
from client import app, GraphState

async def run_console_test(prompt: str):
    print(f"\n{'='*50}\nTEST BAŞLIYOR: '{prompt}'\n{'='*50}")
    
    initial_state: GraphState = {
        "prompt": prompt,
        "session_id": "console_test_session",
        "messages": [],
        "persona": "Konsol test kullanıcısı",
        "intent": "",
        "intent_data": {}
    }

    start_time = time.time()
    step_start_time = start_time

    try:
        # astream() her düğüm tamamlandığında bir yield döner
        async for event in app.astream(initial_state):
            # event dict yapısındadır: {"node_name": {guncellenen_state_verileri}}
            for node_name, state_update in event.items():
                step_end_time = time.time()
                elapsed = step_end_time - step_start_time
                step_start_time = step_end_time  # Bir sonraki ölçüm için zamanı sıfırla

                print(f"\n🟢 [DÜĞÜM: {node_name.upper()}] - Çalışma Süresi: {elapsed:.2f} sn")
                
                # Niyet analizinden dönen sonuçları göster
                if "intent" in state_update:
                    print(f"   ➔ Niyet (Intent): {state_update.get('intent')}")
                    print(f"   ➔ Çıkarılan Veri: {json.dumps(state_update.get('intent_data'), ensure_ascii=False)}")
                
                # Öneri veya bilgi hattından dönen araç kullanım istatistiklerini göster
                if "tool_calls" in state_update:
                    t_calls = state_update.get("tool_calls", [])
                    print(f"   ➔ Çalıştırılan Araç Sayısı: {len(t_calls)}")
                    for t in t_calls:
                        print(f"      - {t.get('tool_name')}: {t.get('arguments')}")
                
                # Final çıktısının özetini göster
                if "final_output" in state_update:
                    out = state_update["final_output"]
                    try:
                        parsed = json.loads(out)
                        print(f"   ➔ Ajanın Mesajı: {parsed.get('text', '')}")
                        movies = parsed.get("movies", [])
                        if movies:
                            print(f"   ➔ Bulunan Filmler ({len(movies)} adet):")
                            for m in movies:
                                print(f"      - {m.get('Film')} ({m.get('TMDB Puanı')})")
                    except json.JSONDecodeError:
                        print(f"   ➔ Çıktı (Düz Metin): {str(out)[:150]}...")

        total_time = time.time() - start_time
        print(f"\n{'='*50}\n🏁 TEST BİTTİ - Toplam Süre: {total_time:.2f} sn\n{'='*50}\n")

    except Exception as e:
        print(f"\n❌ Hata oluştu: {str(e)}")

if __name__ == "__main__":
    # Test edilecek örnek sorgular
    test_prompts = [
        "Uzayda geçen, yapay zeka temalı yalnızlık hissi veren bir bilim kurgu öner",
        "Inception filminin yönetmeni kim?",
        "Nasılsın, bugün sana ne sorabilirim?"
    ]
    
    for prompt in test_prompts:
        asyncio.run(run_console_test(prompt))
        time.sleep(1) # Testler arası ufak bir bekleme