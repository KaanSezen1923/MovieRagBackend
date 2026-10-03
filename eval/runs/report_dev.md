# MovieMCP eval raporu

Koşu sayısı: 93; sorgu sayısı: 31. Hücre biçimi: ortalama [%95 bootstrap GA] (n=sorgu).

Hata oranı (timeout/exception): 0.000 [0.00, 0.00] (n=31)

## 1. Sorgu çıkarımı (QuerySpec)

| alan | skor |
|---|---|
| intent | 1.000 [1.00, 1.00] (n=31) |
| include_genres | 1.000 [1.00, 1.00] (n=14) |
| exclude_genres | 1.000 [1.00, 1.00] (n=4) |
| reference_movie | 1.000 [1.00, 1.00] (n=2) |
| movie_title | 1.000 [1.00, 1.00] (n=3) |
| director | 1.000 [1.00, 1.00] (n=2) |
| actor | 1.000 [1.00, 1.00] (n=1) |
| min_rating | 1.000 [1.00, 1.00] (n=30) |
| max_rating | 1.000 [1.00, 1.00] (n=30) |
| year_from | 0.968 [0.90, 1.00] (n=31) |
| year_to | 1.000 [1.00, 1.00] (n=30) |
| mood_null | 1.000 [1.00, 1.00] (n=3) |
| obscure | 1.000 [1.00, 1.00] (n=2) |
| exact | 0.968 [0.90, 1.00] (n=31) |
| **uydurulan sayısal kısıt oranı** (düşük iyi) | 0.000 [0.00, 0.00] (n=30) |
| **uydurulan mood oranı** (düşük iyi) | 0.000 [0.00, 0.00] (n=2) |

## 2. Ablation: hangi bileşen ne katıyor?

`final` = tam sistem. `score_only` = LLM seçimi yok. `no_filter` = sert filtre + LLM yok. `semantic_only` = yalnızca vektör arama. nDCG/P için etiket (judge/insan) gerekir.

### Kalite (sıralama)

| | nDCG@5 | P@5 (≥1) | P@5 (=2) | etiket kapsaması |
|---|---|---|---|---|
| final | 0.845 [0.77, 0.91] (n=24) | 0.953 [0.91, 0.99] (n=24) | 0.760 [0.65, 0.86] (n=24) | 1.000 [1.00, 1.00] (n=24) |
| score_only | 0.719 [0.58, 0.84] (n=24) | 0.817 [0.68, 0.92] (n=24) | 0.617 [0.47, 0.76] (n=24) | 1.000 [1.00, 1.00] (n=24) |
| no_filter | 0.719 [0.58, 0.84] (n=24) | 0.817 [0.68, 0.92] (n=24) | 0.617 [0.47, 0.76] (n=24) | 1.000 [1.00, 1.00] (n=24) |
| semantic_only | 0.746 [0.63, 0.84] (n=24) | 0.783 [0.66, 0.89] (n=24) | 0.608 [0.46, 0.75] (n=24) | 1.000 [1.00, 1.00] (n=24) |

### Kısıt uyumu (sert kısıtlı sorgularda)

| | tüm filmler kısıta uyuyor | ihlalli film oranı (düşük iyi) | istenen türlerin tamamı |
|---|---|---|---|
| final | 1.000 [1.00, 1.00] (n=10) | 0.000 [0.00, 0.00] (n=10) | 1.000 [1.00, 1.00] (n=14) |
| score_only | 1.000 [1.00, 1.00] (n=10) | 0.000 [0.00, 0.00] (n=10) | 1.000 [1.00, 1.00] (n=14) |
| no_filter | 1.000 [1.00, 1.00] (n=10) | 0.000 [0.00, 0.00] (n=10) | 1.000 [1.00, 1.00] (n=14) |
| semantic_only | 1.000 [1.00, 1.00] (n=10) | 0.000 [0.00, 0.00] (n=10) | 1.000 [1.00, 1.00] (n=14) |

## 3. Retrieval tavanı vs seçim

- Aday havuzunun (ilk 15) en iyi 5'iyle ulaşılabilecek nDCG@5 (tavan): 0.979 [0.95, 1.00] (n=24)
- Gerçek `final` nDCG@5: 0.845 [0.77, 0.91] (n=24)
- Havuzda en az 1 ilgili film olan sorgular: 1.000 [1.00, 1.00] (n=24); en az 1 'çok uygun' (=2): 0.958 [0.88, 1.00] (n=24)
- Yorum: tavan düşükse sorun retrieval/filtrede (QuerySpec, plan, semantik arama); tavan yüksek ama final düşükse sorun LLM seçim aşamasında.
- Sert filtre ortalama aday eleme oranı: 0.29%

## 4. Dürüstlük, sadakat, dayanıklılık

| metrik | skor |
|---|---|
| Cevabı olmayan sorularda doğru şekilde boş dönme (yüksek iyi) | 0.500 [0.00, 1.00] (n=2) |
| Cevabı olan sorularda yanlışlıkla boş dönme (düşük iyi) | 0.000 [0.00, 0.00] (n=24) |
| Gerekçelerin karttan desteklenme oranı (judge) | — |
| Persona enjeksiyonuna kanma (düşük iyi) | 0.000 [0.00, 0.00] (n=1) |
| Film bilgisi: bulundu | 1.000 [1.00, 1.00] (n=3) |
| Film bilgisi: doğru film | 0.000 [0.00, 0.00] (n=3) |
| Film bilgisi: doğru yıl (aynı adlı film ayrımı) | 1.000 [1.00, 1.00] (n=3) |
| Film bilgisi: yönetmen+oyuncu dolu | 1.000 [1.00, 1.00] (n=3) |
| Sohbette film listesi dönmedi | 1.000 [1.00, 1.00] (n=2) |

## 5. Tutarlılık (aynı soru, birden fazla koşu)

- Sonuç kümesi Jaccard: 0.957 [0.92, 0.99] (n=31)
- Aynı QuerySpec üretme oranı: 1.000 [1.00, 1.00] (n=31)

## 6. Gecikme / maliyet (yalnızca ilk tekrar; cache ısındıktan sonrakiler hariç)

- p50: 3.2 sn, p95: 6.0 sn
- Ortalama araç çağrısı/sorgu: 2.7

## 7. Sorgu tipine göre kırılım

| tip | n | spec exact | final nDCG@5 | final hard_ok |
|---|---|---|---|---|
| constraint_must | 1 | 1.00 (n=1) | 0.79 (n=1) | — |
| english | 1 | 1.00 (n=1) | 0.87 (n=1) | — |
| general | 2 | 1.00 (n=2) | — | — |
| genre | 1 | 1.00 (n=1) | 1.00 (n=1) | — |
| genre_exclude | 3 | 1.00 (n=3) | 0.99 (n=3) | 1.00 (n=3) |
| info | 2 | 1.00 (n=2) | — | — |
| info_ambiguous | 1 | 1.00 (n=1) | — | — |
| injection | 1 | 1.00 (n=1) | 0.96 (n=1) | — |
| mood | 1 | 1.00 (n=1) | 0.91 (n=1) | — |
| multi_turn | 2 | 1.00 (n=2) | 1.00 (n=2) | 1.00 (n=2) |
| no_answer | 2 | 1.00 (n=2) | — | — |
| obscure | 1 | 1.00 (n=1) | 0.42 (n=1) | — |
| person | 2 | 1.00 (n=2) | 0.78 (n=2) | — |
| person_genre | 1 | 1.00 (n=1) | 0.83 (n=1) | — |
| rating | 2 | 1.00 (n=2) | 0.85 (n=2) | 1.00 (n=2) |
| reference | 2 | 1.00 (n=2) | 0.93 (n=2) | — |
| semantic | 2 | 1.00 (n=2) | 0.66 (n=2) | — |
| typo_informal | 1 | 1.00 (n=1) | 0.57 (n=1) | — |
| year | 3 | 0.67 (n=3) | 0.84 (n=3) | 1.00 (n=3) |
