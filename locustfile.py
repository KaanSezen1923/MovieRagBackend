# locustfile.py
from locust import HttpUser, task, between
import random
import uuid

class MovieAIUser(HttpUser):
    wait_time = between(1, 3) # Her istek arası 1-3 saniye bekle
    
    def on_start(self):
        # Yük testi için her sanal kullanıcıya benzersiz bir oturum ve token (mock veya gerçek) atanır
        self.session_id = str(uuid.uuid4())
        # API'niz token gerektiriyorsa buraya geçerli bir token ekleyin
        self.headers = {"Authorization": ""} 
        
    @task(3)
    def ask_recommendation(self):
        prompts = [
            "Bana Nolan filmi öner.",
            "Canım sıkkın, komedi izlemek istiyorum.",
            "Zombilerle ilgili bilim kurgu arıyorum.",
            "90'ların klasik aksiyon filmleri."
        ]
        payload = {
            "prompt": random.choice(prompts),
            "session_id": self.session_id
        }
        # Chat endpoint'ini zorla
        with self.client.post("/chat", json=payload, headers=self.headers, catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if "answer" not in data:
                    response.failure("Yanıt 'answer' anahtarı içermiyor.")
            else:
                response.failure(f"Hata kodu döndü: {response.status_code}")

    @task(1)
    def fetch_favorites(self):
        # Veritabanı havuzunu (asyncpg) eşzamanlı okumalarla test et
        self.client.get("/favorites", headers=self.headers)