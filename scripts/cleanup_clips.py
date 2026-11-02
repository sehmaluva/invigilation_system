"""Delete clips/thumbnails older than N days (default: retention.days in config.yaml).

  python scripts/cleanup_clips.py            # use config value
  python scripts/cleanup_clips.py --days 0   # delete all clips now
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.db import init_db                    # noqa: E402
from app.retention import purge_old_media     # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--days", type=float, default=None)
a = ap.parse_args()
init_db()
print(f"Deleted {len(purge_old_media(a.days))} files")
