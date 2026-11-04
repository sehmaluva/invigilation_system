"""Loads config.yaml once and exposes it as CFG (a plain dict)."""
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent           # project folder
CONFIG_PATH = Path(os.environ.get("INVIG_CONFIG", ROOT / "config.yaml"))

with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

DATA_DIR = (ROOT / CFG["server"]["data_dir"]).resolve()
CLIP_DIR = DATA_DIR / "clips"
THUMB_DIR = DATA_DIR / "thumbs"
UPLOAD_DIR = DATA_DIR / "uploads"
SERIES_DIR = DATA_DIR / "series"          # per-job cheating-score histories
for d in (DATA_DIR, CLIP_DIR, THUMB_DIR, UPLOAD_DIR, SERIES_DIR):
    d.mkdir(parents=True, exist_ok=True)


def resolve(path: str) -> Path:
    """Paths in config.yaml are relative to the project folder."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
