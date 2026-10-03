import streamlit as st
import requests
import json
import uuid

# Backend API Adresi
API_URL = "http://localhost:3000"

st.set_page_config(page_title="Movie Explorer AI", page_icon="🍿", layout="wide")

# --- Session State Başlatma ---
if "token" not in st.session_state:
    st.session_state.token = None
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "username" not in st.session_state:
    st.session_state.username = ""
if "menu_selection" not in st.session_state:
    st.session_state.menu_selection = "Sohbet"

# --- Yardımcı Fonksiyonlar ---
def get_headers():
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

def login(username, password):
    res = requests.post(f"{API_URL}/login", json={"username": username, "password": password})
    if res.status_code == 200:
        data = res.json()
        st.session_state.token = data["access_token"]
        st.session_state.username = data["username"]
        st.success(data["message"])
        st.rerun()
    else:
        st.error("Giriş başarısız. Lütfen bilgilerinizi kontrol edin.")

def signup(username, email, password):
    res = requests.post(f"{API_URL}/signup", json={"username": username, "email": email, "password": password})
    if res.status_code == 200:
        st.success("Kayıt başarılı! Şimdi giriş yapabilirsiniz.")
    else:
        st.error(res.json().get("detail", "Kayıt sırasında bir hata oluştu."))

def load_chat_history(session_id):
    st.session_state.session_id = session_id
    res = requests.get(f"{API_URL}/chat/{session_id}", headers=get_headers())
    if res.status_code == 200:
        st.session_state.messages = res.json().get("history", [])
    st.session_state.menu_selection = "Sohbet"
    st.rerun()

def start_new_chat():
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.menu_selection = "Sohbet"
    st.rerun()

def get_favorite_ids():
    if not st.session_state.token: return []
    res = requests.get(f"{API_URL}/favorites/ids", headers=get_headers())
    if res.status_code == 200:
        return res.json().get("favorite_ids", [])
    return []

def toggle_favorite(movie, is_saved):
    movie_id = movie.get("movie_id") or movie.get("id") or movie.get("Film")
    if is_saved:
        requests.delete(f"{API_URL}/favorites/{movie_id}", headers=get_headers())
    else:
        payload = {
            "movie_id": str(movie_id),
            "title": movie.get("Film") or movie.get("title", ""),
            "genres": movie.get("Türler") or movie.get("genres", ""),
            "director": movie.get("Director") or movie.get("director", ""),
            "cast_members": movie.get("Cast") or movie.get("cast", ""),
            "poster_url": movie.get("Poster") or movie.get("posterUrl", ""),
            "imdb_rating": str(movie.get("TMDB Puanı") or movie.get("rating", "")),
            "trailer_url": movie.get("Fragman") or movie.get("trailerUrl", "")
        }
        requests.post(f"{API_URL}/favorites", headers=get_headers(), json=payload)

def render_movie_card(movie, fav_ids):
    """Film JSON objesini görselleştirir ve Favori Ekle/Çıkar butonu koyar."""
    movie_id = str(movie.get("movie_id") or movie.get("id") or movie.get("Film"))
    is_saved = movie_id in fav_ids

    with st.container(border=True):
        col1, col2 = st.columns([1, 4])
        with col1:
            if movie.get("Poster") or movie.get("poster_url"):
                st.image(movie.get("Poster") or movie.get("poster_url"), use_container_width=True)
            else:
                st.write("🎬 Görsel Yok")
            
            # Favori Butonu
            btn_label = "❌ Listeden Çıkar" if is_saved else "🔖 İzleme Listesine Ekle"
            btn_type = "secondary" if is_saved else "primary"
            if st.button(btn_label, key=f"fav_{movie_id}_{uuid.uuid4()}", type=btn_type):
                toggle_favorite(movie, is_saved)
                st.rerun()

        with col2:
            st.subheader(f"{movie.get('Film') or movie.get('title', 'Bilinmeyen Film')}")
            st.caption(f"⭐ TMDB Puanı: {movie.get('TMDB Puanı') or movie.get('imdb_rating', '-')} | 🎭 Türler: {movie.get('Türler') or movie.get('genres', '-')}")
            st.write(f"**Yönetmen:** {movie.get('Director') or movie.get('director', '-')} | **Oyuncular:** {movie.get('Cast') or movie.get('cast_members', '-')}")
            st.write(movie.get("Özet") or movie.get("summary", "Özet bulunamadı."))
            
            if movie.get("Neden Önerildi"):
                st.info(f"💡 Neden Önerildi: {movie['Neden Önerildi']}")
                
            cols = st.columns(2)
            if movie.get("Şu Anki Platform(lar)"):
                cols[0].write(f"📺 **İzleme Seçenekleri:** {movie['Şu Anki Platform(lar)']}")
            if movie.get("Fragman") or movie.get("trailer_url"):
                trailer = movie.get("Fragman") or movie.get("trailer_url")
                cols[1].markdown(f"[🎥 Fragmanı İzle]({trailer})")

def process_assistant_message(content, fav_ids):
    """Mesajın düz metin mi yoksa film kartı JSON'u mu olduğunu ayırır."""
    try:
        data = json.loads(content)
        if isinstance(data, dict) and data.get("type") == "movie_list":
            if data.get("text"):
                st.markdown(data["text"])
            if data.get("mood_response"):
                st.caption(f"*{data['mood_response']}*")
            for movie in data.get("movies", []):
                render_movie_card(movie, fav_ids)
            return
    except json.JSONDecodeError:
        pass
    
    st.markdown(content)

# --- Yan Menü (Sidebar) ---
with st.sidebar:
    if not st.session_state.token:
        st.title("🔐 Oturum Aç")
        tab1, tab2 = st.tabs(["Giriş Yap", "Kayıt Ol"])
        
        with tab1:
            with st.form("login_form"):
                l_user = st.text_input("Kullanıcı Adı")
                l_pass = st.text_input("Şifre", type="password")
                if st.form_submit_button("Giriş"):
                    login(l_user, l_pass)
                    
        with tab2:
            with st.form("signup_form"):
                s_user = st.text_input("Kullanıcı Adı")
                s_email = st.text_input("E-posta")
                s_pass = st.text_input("Şifre", type="password")
                if st.form_submit_button("Kayıt Ol"):
                    signup(s_user, s_email, s_pass)
    else:
        st.title(f"👋 Hoş geldin, {st.session_state.username}")
        
        # Mobil menü sekmeleri
        menu = ["Sohbet", "Keşfet", "İzleme Listem", "Profil"]
        st.session_state.menu_selection = st.radio("Menü", menu, index=menu.index(st.session_state.menu_selection))
        
        st.divider()
        if st.session_state.menu_selection == "Sohbet":
            if st.button("➕ Yeni Sohbet", use_container_width=True):
                start_new_chat()
                
            st.subheader("Geçmiş Sohbetler")
            res_sessions = requests.get(f"{API_URL}/sessions", headers=get_headers())
            if res_sessions.status_code == 200:
                sessions = res_sessions.json().get("sessions", [])
                for s in sessions:
                    title = s.get("title", "Yeni Sohbet")
                    if st.button(title[:25] + "...", key=s["session_id"], use_container_width=True):
                        load_chat_history(s["session_id"])
        
        st.divider()
        if st.button("🚪 Çıkış Yap", type="secondary", use_container_width=True):
            st.session_state.token = None
            st.session_state.messages = []
            st.rerun()

# --- Ana Ekran (Sekmelere Göre Yönlendirme) ---
if not st.session_state.token:
    st.info("👈 Uygulamayı kullanmaya başlamak için lütfen sol menüden giriş yapın veya kayıt olun.")
else:
    fav_ids = get_favorite_ids()

    # 1. SOHBET SEKMESİ
    if st.session_state.menu_selection == "Sohbet":
        st.title("🍿 Movie Explorer AI")
        st.caption("Ne izlemek istersin? Kriterlerini yaz veya mikrofona konuş.")

        # Öneriler (Mobil Suggestions)
        suggestions = ['Bilim kurgu filmi öner', 'Nolan filmi öner', 'Ters köşe yapan filmler']
        cols = st.columns(len(suggestions))
        selected_suggestion = None
        for i, suggestion in enumerate(suggestions):
            if cols[i].button(suggestion, key=f"sug_{i}", use_container_width=True):
                selected_suggestion = suggestion

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                process_assistant_message(msg["content"], fav_ids)

        # Sesli Arama
        audio_file = st.file_uploader("🎙️ Sesli arama (.wav, .mp3)", type=["wav", "mp3", "m4a", "ogg"])
        transcribed_text = ""
        if audio_file is not None:
            if st.button("Sesi Çevir"):
                with st.spinner("Ses işleniyor..."):
                    files = {"audio": (audio_file.name, audio_file, "audio/mpeg")}
                    audio_res = requests.post(f"{API_URL}/transcribe", headers=get_headers(), files=files)
                    if audio_res.status_code == 200:
                        transcribed_text = audio_res.json().get("text", "")
                        st.success(f"Çevrilen Metin: {transcribed_text}")
                    else:
                        st.error("Ses çevirisi başarısız oldu.")

        prompt = st.chat_input("Bir film ara, yönetmen sor veya modunu söyle...")
        final_prompt = selected_suggestion or prompt or transcribed_text

        if final_prompt:
            with st.chat_message("user"):
                st.markdown(final_prompt)
            st.session_state.messages.append({"role": "user", "content": final_prompt})

            with st.chat_message("assistant"):
                with st.spinner("AI Düşünüyor..."):
                    payload = {"prompt": final_prompt, "session_id": st.session_state.session_id}
                    chat_res = requests.post(f"{API_URL}/chat", headers=get_headers(), json=payload)
                    
                    if chat_res.status_code == 200:
                        answer_content = chat_res.json().get("answer", "")
                        process_assistant_message(answer_content, fav_ids)
                        st.session_state.messages.append({"role": "assistant", "content": answer_content})
                    else:
                        st.error("Asistana ulaşılamadı.")

    # 2. KEŞFET SEKMESİ
    elif st.session_state.menu_selection == "Keşfet":
        st.title("🧭 Keşfet")
        categories = ['Popüler', 'Vizyondakiler', 'En Çok Oy Alanlar', 'Yakında']
        selected_cat = st.radio("Kategori Seç", categories, horizontal=True)
        
        with st.spinner("Filmler yükleniyor..."):
            res = requests.get(f"{API_URL}/discover?category={selected_cat}", headers=get_headers())
            if res.status_code == 200:
                movies = res.json().get("movies", [])
                if movies:
                    for m in movies:
                        render_movie_card(m, fav_ids)
                else:
                    st.info("Bu kategoride film bulunamadı.")
            else:
                st.error("Keşfet verileri yüklenirken hata oluştu. Lütfen backend'i kontrol edin.")

    # 3. İZLEME LİSTEM SEKMESİ
    elif st.session_state.menu_selection == "İzleme Listem":
        st.title("🔖 İzleme Listem")
        
        with st.spinner("Favorileriniz yükleniyor..."):
            res = requests.get(f"{API_URL}/favorites", headers=get_headers())
            if res.status_code == 200:
                favorites = res.json().get("favorites", [])
                if favorites:
                    for f in favorites:
                        render_movie_card(f, fav_ids)
                else:
                    st.info("Henüz film kaydetmediniz.")
            else:
                st.error("Liste alınırken hata oluştu.")

    # 4. PROFİL SEKMESİ
    elif st.session_state.menu_selection == "Profil":
        st.title("👤 Profil ve İstatistikler")
        
        # İstatistikleri hesaplama
        favs_res = requests.get(f"{API_URL}/favorites", headers=get_headers())
        sess_res = requests.get(f"{API_URL}/sessions", headers=get_headers())
        
        favorites = favs_res.json().get("favorites", []) if favs_res.status_code == 200 else []
        sessions = sess_res.json().get("sessions", []) if sess_res.status_code == 200 else []
        
        genre_counts = {}
        director_counts = {}
        
        for fav in favorites:
            genres_str = fav.get("Türler") or fav.get("genres", "")
            if genres_str:
                for g in genres_str.split(","):
                    clean_g = g.strip()
                    if clean_g:
                        genre_counts[clean_g] = genre_counts.get(clean_g, 0) + 1
            
            director_str = fav.get("Director") or fav.get("director", "")
            if director_str and director_str != "Bilinmiyor":
                clean_d = director_str.strip()
                director_counts[clean_d] = director_counts.get(clean_d, 0) + 1
                
        top_genres = sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        top_directors = sorted(director_counts.items(), key=lambda x: x[1], reverse=True)[:5]

        # İstatistik Gösterimi
        col1, col2 = st.columns(2)
        col1.metric("Kaydedilen Film (Favoriler)", len(favorites))
        col2.metric("Toplam Sohbet", len(sessions))
        
        st.divider()
        
        col3, col4 = st.columns(2)
        with col3:
            st.subheader("En Çok Eklenen Türler")
            if top_genres:
                for name, count in top_genres:
                    st.write(f"🎭 **{name}**: {count} Film")
            else:
                st.write("Veri yok.")
                
        with col4:
            st.subheader("Favori Yönetmenler")
            if top_directors:
                for name, count in top_directors:
                    st.write(f"🎬 **{name}**: {count} Film")
            else:
                st.write("Veri yok.")
        
        st.divider()
        
        # Ayarlar
        st.subheader("Hesap Ayarları")
        with st.expander("Profil Bilgilerini Düzenle"):
            new_user = st.text_input("Yeni Kullanıcı Adı", value=st.session_state.username)
            if st.button("Kullanıcı Adını Güncelle"):
                res = requests.post(f"{API_URL}/update-profile", headers=get_headers(), json={"new_username": new_user})
                if res.status_code == 200:
                    st.session_state.username = new_user
                    st.success("Kullanıcı adı güncellendi!")
                    st.rerun()
                else:
                    st.error("Güncelleme başarısız.")
                    
        with st.expander("Gizlilik ve Güvenlik (Şifre Değiştir)"):
            old_pass = st.text_input("Mevcut Şifre", type="password")
            new_pass = st.text_input("Yeni Şifre", type="password")
            confirm_pass = st.text_input("Yeni Şifre Tekrar", type="password")
            if st.button("Şifreyi Değiştir"):
                if new_pass == confirm_pass:
                    res = requests.post(f"{API_URL}/change-password", headers=get_headers(), json={"old_password": old_pass, "new_password": new_pass})
                    if res.status_code == 200:
                        st.success("Şifreniz başarıyla değiştirildi!")
                    else:
                        st.error("Şifre değiştirilemedi. Mevcut şifrenizi kontrol edin.")
                else:
                    st.error("Yeni şifreler eşleşmiyor.")