"""Delete clips and thumbnails older than retention.days (privacy safeguard)."""
import os
import time

from .config import CFG, CLIP_DIR, THUMB_DIR
from .db import Flag, SessionLocal, audit


def purge_old_media(days=None, user_id=None):
    days = CFG["retention"]["days"] if days is None else days
    cutoff = time.time() - days * 86400
    deleted = []
    for folder in (CLIP_DIR, THUMB_DIR):
        for f in folder.iterdir():
            if f.is_file() and f.stat().st_mtime < cutoff:
                f.unlink()
                deleted.append(str(f))
    with SessionLocal() as db:                       # clear DB paths that now point nowhere
        for flag in db.query(Flag).all():
            if flag.clip_path and not os.path.exists(flag.clip_path):
                flag.clip_path = None
            if flag.thumb_path and not os.path.exists(flag.thumb_path):
                flag.thumb_path = None
        db.commit()
        audit(db, user_id, "retention_purge", f"{days} days", f"deleted {len(deleted)} files")
    return deleted
