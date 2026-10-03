# MovieMCP eval paketi

`eval/` klasörünü proje köküne (client.py, server.py, shared.py yanına) koy.

## Kurulum
    pip install pytest        # diğer bağımlılıklar projede zaten var

## 1) Birim testleri (saniyeler, LLM/ağ gerekmez)
    pytest eval -q
- `test_pipeline_units.py`: _clean_spec, parse_spec, hard_filter, merge/RRF, skor, build_plan, parse_selection, yönlendirme
- `test_metrics.py`: eval metriklerinin kendi doğruluğu
- `test_harness_smoke.py`: sahte LLM/MCP ile run_eval -> judge havuzu -> report boru hattı

## 2) Uçtan uca koşu (Ollama + Neo4j + TMDB gerekir)
    python eval/run_eval.py --data eval/data/dev.jsonl --out eval/runs/dev_full.jsonl --repeats 3 --temperature 0
- `--repeats 3`: tutarlılık (Jaccard) ölçümü. Gecikme yalnızca rep0'dan hesaplanır (server cache'i yüzünden).
- Not: `state["messages"]` içine güncel kullanıcı mesajı da eklenir (general_chat_node bunu bekliyor).
  Uygulaman eklemiyorsa `--history-only` ver.

## 3) Etiketleme (LLM-judge, pooling)
    # Judge modeli: sistem modelinden FARKLI, tercihen daha güçlü. Üç yoldan biri:
    #   a) komutta:     --judge-model <ad>
    #   b) .env içinde: JUDGE_MODEL=<ad>
    #   c) PowerShell:  $env:JUDGE_MODEL="<ad>"   |  cmd: set JUDGE_MODEL=<ad>   |  Linux/macOS: export JUDGE_MODEL=<ad>
    python eval/judge.py pool  --runs eval/runs/dev_full.jsonl --judge-model <ad>
    python eval/judge.py faith --runs eval/runs/dev_full.jsonl --judge-model <ad>

## 4) Judge'ı insanla doğrula (atlama!)
    python eval/judge.py sample --runs eval/runs/dev_full.jsonl --n 40
    # data/human_sample_dev.csv içindeki human_score sütununu 0/1/2 ile doldur
    python eval/judge.py agree
İnsan etiketi doldurulan satırlar rapor hesaplanırken judge'ı otomatik ezer.

## 5) Rapor
    python eval/report.py --runs eval/runs/dev_full.jsonl --out eval/runs/report_dev.md

## Veri seti kuralları
- `data/dev.jsonl`: sistemi bunlara göre ayarla. **Nihai sayı için hiç bakmadığın `data/test.jsonl` yaz** ve
  `--data eval/data/test.jsonl` (+ `--labels/--faith/--human` için `_test` dosyaları) ile bir kez çalıştır.
- Gold alanları kısmidir. Sayısal alanlar (min/max_rating, year_from/to) yazılmadıysa `None` beklenir
  (LLM'in sayı uydurmasını yakalar); belirsiz sorguya `"loose": true` ekle.
- Ek alanlar: `expect_empty`, `expect_title`/`expect_year`, `forbid_titles` + `persona` (enjeksiyon testi),
  `must_have` (judge'a gider), `messages` (çok turlu bağlam).
