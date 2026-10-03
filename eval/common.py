"""Eval paketi için ortak yardımcılar."""
import json
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RUNS = ROOT / "runs"


def norm_text(s) -> str:
    """Büyük/küçük harf, aksan ve Türkçe i/ı farkını yok sayan normalizasyon."""
    s = unicodedata.normalize("NFKD", str(s or "")).casefold().replace("ı", "i")
    return "".join(ch for ch in s if not unicodedata.combining(ch)).strip()


def load_jsonl(path) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                out.append(json.loads(line))
    return out


def append_jsonl(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def read_json(path, default=None):
    p = Path(path)
    if not p.exists():
        return default
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
