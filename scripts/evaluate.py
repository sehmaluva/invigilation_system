"""
Compare a job's flags with a ground-truth CSV (from make_synthetic_video.py or hand-labelled).

A flag is a true positive if its [start_s, end_s] overlaps a cheating interval (± tolerance).
Reports precision/recall at the alert cap (top-K flags by peak score) and for all flags.

Example:
  python scripts/evaluate.py --job-id 1 --gt data/synthetic_gt.csv --cap 10
"""
import argparse
import csv
import json
import sqlite3
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent


def overlaps(a0, a1, b0, b1, tol):
    return a0 <= b1 + tol and b0 - tol <= a1


def score(flags, gt, tol):
    cheat = [g for g in gt if g["is_cheating"]]
    tp = [f for f in flags if any(overlaps(f["start_s"], f["end_s"], g["start_s"], g["end_s"], tol) for g in cheat)]
    hit = [g for g in cheat if any(overlaps(f["start_s"], f["end_s"], g["start_s"], g["end_s"], tol) for f in flags)]
    return {"flags": len(flags), "true_positives": len(tp), "false_positives": len(flags) - len(tp),
            "precision": round(len(tp) / len(flags), 3) if flags else None,
            "recall": round(len(hit) / len(cheat), 3) if cheat else None,
            "cheating_intervals": len(cheat), "intervals_found": len(hit)}


def main():
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--job-id", type=int, required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--db", default=str(ROOT / cfg["server"]["data_dir"] / "invigilation.db"))
    ap.add_argument("--cap", type=int, default=cfg["review"]["alert_cap"])
    ap.add_argument("--tolerance", type=float, default=1.0, help="seconds of slack when matching")
    ap.add_argument("--out", help="optional JSON file for the results")
    a = ap.parse_args()

    gt = [{"start_s": float(r["start_s"]), "end_s": float(r["end_s"]), "is_cheating": int(r["is_cheating"])}
          for r in csv.DictReader(open(a.gt))]
    con = sqlite3.connect(a.db)
    con.row_factory = sqlite3.Row
    flags = [dict(r) for r in con.execute(
        "SELECT id, start_s, end_s, peak_score FROM flags WHERE job_id=? ORDER BY peak_score DESC", (a.job_id,))]
    job = con.execute("SELECT * FROM jobs WHERE id=?", (a.job_id,)).fetchone()
    if job is None:
        raise SystemExit(f"Job {a.job_id} not found in {a.db}")

    hours = (job["video_seconds"] or 0) / 3600
    result = {
        "job_id": a.job_id, "alert_cap": a.cap, "tolerance_s": a.tolerance,
        "at_cap": score(flags[:a.cap], gt, a.tolerance),
        "all_flags": score(flags, gt, a.tolerance),
        "flags_per_hour": round(len(flags) / hours, 2) if hours else None,
        "processing": {"avg_fps": job["avg_fps"], "avg_latency_ms": job["avg_latency_ms"],
                       "video_seconds": job["video_seconds"], "proc_seconds": job["proc_seconds"]},
    }
    print(json.dumps(result, indent=2))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
