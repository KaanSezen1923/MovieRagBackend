# test_chat_accuracy.py
import pytest
import json
from client import app as graph_app

# 50 soruluk setin temsili bir kısmı
TEST_QUERIES = [
    ("Bana uzayda geçen bilim kurgu filmleri öner", "Science Fiction"),
    ("Çok üzgünüm, modumu yükseltecek bir komedi öner", "Comedy"),
]

@pytest.mark.asyncio
@pytest.mark.parametrize("prompt, expected_genre", TEST_QUERIES)
async def test_movie_recommendations(prompt, expected_genre):
    initial_state = {
        "prompt": prompt,
        "persona": "Standart kullanıcı",
        "messages": [{"role": "user", "content": prompt}],
        "tools": [], # Mock araçlar veya gerçek araç setini geçirebilirsiniz
        "intent": "",
        "session_id": "test_session_123"
    }

    result = await graph_app.ainvoke(initial_state)
    output_json = result.get("final_output")
    
    assert output_json is not None, "Çıktı boş dönmemeli"
    
    data = json.loads(output_json)
    
    if data.get("type") == "movie_list":
        movies = data.get("movies", [])
        
        # 1. Beklenen film sayısı kontrolü
        assert 1 <= len(movies) <= 5, "Film sayısı 1 ile 5 arasında olmalı"
        
        for movie in movies:
            # 2. "Neden Önerildi" alanı doluluk kontrolü
            assert "Neden Önerildi" in movie, "Neden Önerildi alanı eksik"
            assert len(movie["Neden Önerildi"]) > 10, "Neden Önerildi açıklaması çok kısa veya boş"
            
            # 3. Tür eşleşme kontrolü (Gevşek kontrol, LLM farklı tür de önerebilir ama ana tür olmalı)
            genres = movie.get("Türler", "")
            assert expected_genre in genres, f"Beklenen tür {expected_genre}, {genres} içinde bulunamadı"