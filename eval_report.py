"""
Test raporunu (test_raporu_client.json) makine-kontrollü kriterlerle puanlar.
Kullanım:  python eval_report.py test_raporu_client.json
İyileştirme öncesi/sonrası raporu karşılaştırmak için iki kez çalıştır.
(Raporun 'soru' sırası aynı kaldığı sürece numaralar geçerlidir.)
"""
import json, statistics, sys

def movies_of(item):
    s = item["sistem_yaniti"]
    s = json.loads(s) if isinstance(s, str) else s
    return s.get("movies", []) if isinstance(s, dict) else []

def genres(m): return {g.strip().lower() for g in (m.get("Türler") or "").split(",") if g.strip()}
def rating(m):
    try: return float(m.get("TMDB Puanı"))
    except (TypeError, ValueError): return None

def all_have(*gs):      return lambda ms: bool(ms) and all(set(g.lower() for g in gs) <= genres(m) for m in ms)
def none_have(g):       return lambda ms: bool(ms) and all(g.lower() not in genres(m) for m in ms)
def all_rating(lo=0, hi=10):
    return lambda ms: bool(ms) and all(rating(m) is not None and lo <= rating(m) <= hi for m in ms)
def contains(sub, top=None):
    return lambda ms: any(sub.lower() in (m.get("Film") or "").lower() for m in (ms[:top] if top else ms))
def lacks(sub):         return lambda ms: not any(sub.lower() in (m.get("Film") or "").lower() for m in ms)
def no_rating_noise():  return lambda ms: bool(ms) and all(rating(m) not in (0.0, 10.0) for m in ms)
def both(*fs):          return lambda ms: all(f(ms) for f in fs)

# soru no -> [(açıklama, kontrol)]
CHECKS = {
    3:  [("hepsi Comedy", all_have("Comedy"))],
    5:  [("hepsi Action", all_have("Action"))],
    7:  [("hepsi Animation", all_have("Animation"))],
    8:  [("hepsi Horror", all_have("Horror"))],
    9:  [("hepsi Romance", all_have("Romance"))],
    10: [("hepsi Documentary", all_have("Documentary"))],
    12: [("Horror yok", none_have("Horror"))],
    13: [("hepsi >= 8.0", all_rating(lo=8.0))],
    15: [("hepsi SciFi+Western", all_have("Science Fiction", "Western"))],
    18: [("Action yok", none_have("Action")), ("Ex Machina var", contains("Ex Machina"))],
    19: [("hepsi Romance+Comedy", all_have("Romance", "Comedy"))],
    20: [("hepsi >= 7.0 ve War", both(all_rating(lo=7.0), all_have("War")))],
    25: [("Action yok", none_have("Action"))],
    34: [("Godzilla Resurgence ilk 2'de", contains("Resurgence", top=2)), ("Minyonlar yok", lacks("Minyonlar"))],
    36: [("hepsi < 5.0", all_rating(hi=4.99))],
    38: [("puan gürültüsü (0.0/10.0) yok", no_rating_noise())],
    43: [("Akıl Defteri (Memento) 1. sırada", contains("Akıl Defteri", top=1))],
    44: [("'Şey' (uzaylı görünür) yok", lacks("Şey"))],
    49: [("Nirvanna var", contains("Nirvanna"))],
}

def main(path):
    data = json.load(open(path, encoding="utf-8"))
    passed = total = 0
    print(f"{'Q':>3}  {'kontrol':42} sonuç")
    for q in sorted(CHECKS):
        if q > len(data): continue
        ms = movies_of(data[q - 1])
        for desc, fn in CHECKS[q]:
            ok = bool(fn(ms)); total += 1; passed += ok
            print(f"{q:>3}  {desc:42} {'GEÇTİ' if ok else 'KALDI'}")
    print(f"\nKısıt/altın kontrol başarısı: {passed}/{total}  (%{100*passed/total:.0f})")

    allm = [m for d in data for m in movies_of(d)]
    n = len(allm) or 1
    ms_counts = [len(movies_of(d)) for d in data]
    lat = sorted(d["yanit_suresi_saniye"] for d in data)
    print(f"Toplam öneri: {len(allm)}")
    print(f"  izlenebilir ('yok' değil)        : %{100*sum(1 for m in allm if 'yok' not in (m.get('Şu Anki Platform(lar)') or 'yok') and (m.get('Şu Anki Platform(lar)') or '').strip())/n:.0f}")
    print(f"  gerekçesi boş                    : %{100*sum(1 for m in allm if not (m.get('Neden Önerildi') or '').strip())/n:.0f}")
    print(f"  özeti yok                        : %{100*sum(1 for m in allm if 'bulunamadı' in (m.get('Özet') or ''))/n:.0f}")
    print(f"  ≤2 sonuç dönen sorgu             : {sum(1 for c in ms_counts if c <= 2)}/{len(data)}")
    print(f"  gecikme  ort {statistics.mean(lat):.1f}s | medyan {statistics.median(lat):.1f}s | "
          f"p90 {lat[int(len(lat)*.9)]:.1f}s | max {lat[-1]:.1f}s")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "test_raporu_client_nemotron-3.json")
