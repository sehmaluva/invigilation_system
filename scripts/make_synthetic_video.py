"""
Stitch still images into a test video + a ground-truth CSV of the "cheating" intervals.

Image folder layout (same as examdatasetnew):
    images/
      normal/            *.jpg
      sidewatching/      *.jpg
      ...
Classes listed in --cheat-classes are marked 1. The default is the suspicious
examdatasetnew folders (the binary notebook head is just called "cheating").
A flat folder (no subfolders) also works: every 2nd segment is then marked as cheating
(only useful to test the plumbing, not the model).

Example:
  python scripts/make_synthetic_video.py --images path/to/examdatasetnew/test \
         --out data/synthetic.mp4 --gt data/synthetic_gt.csv --segments 20 --seconds 6
"""
import argparse
import csv
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def letterbox(img, w, h):
    """Fit the image inside w x h on a black background."""
    s = min(w / img.shape[1], h / img.shape[0])
    img = cv2.resize(img, None, fx=s, fy=s)
    canvas = np.zeros((h, w, 3), np.uint8)
    y, x = (h - img.shape[0]) // 2, (w - img.shape[1]) // 2
    canvas[y:y + img.shape[0], x:x + img.shape[1]] = img
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, help="folder of images (class subfolders)")
    ap.add_argument("--out", default="data/synthetic.mp4")
    ap.add_argument("--gt", default="data/synthetic_gt.csv")
    ap.add_argument("--segments", type=int, default=20, help="number of images to show")
    ap.add_argument("--seconds", type=float, default=6, help="seconds each image is shown")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--size", default="1280x720")
    # Ground truth is by folder name. The binary notebook head is just "cheating",
    # which is not a folder in examdatasetnew, so keep the original suspicious folders.
    ap.add_argument("--cheat-classes", nargs="*",
                    default=["backwatching", "showinggestures", "sidewatching", "suspiciouswatching"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    random.seed(a.seed)
    W, H = map(int, a.size.split("x"))

    root = Path(a.images)
    classes = {d.name: [p for p in d.iterdir() if p.suffix.lower() in EXTS]
               for d in sorted(root.iterdir()) if d.is_dir()}
    classes = {k: v for k, v in classes.items() if v}
    flat = not classes
    if flat:
        classes = {"unknown": [p for p in root.iterdir() if p.suffix.lower() in EXTS]}
        print("No class subfolders: every 2nd segment is marked cheating (plumbing test only).")
    if not any(classes.values()):
        sys.exit("No images found.")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (W, H))
    rows, t = [], 0.0
    for i in range(a.segments):
        cls = random.choice(list(classes))
        path = random.choice(classes[cls])
        img = cv2.imread(str(path))
        if img is None:
            continue
        cheat = (i % 2 == 1) if flat else (cls in a.cheat_classes)
        n = int(a.seconds * a.fps)
        base = letterbox(img, W, H)
        for k in range(n):                       # slow zoom so frames are not identical
            z = 1 + 0.03 * k / n
            M = cv2.getRotationMatrix2D((W / 2, H / 2), 0, z)
            vw.write(cv2.warpAffine(base, M, (W, H)))
        rows.append({"start_s": round(t, 2), "end_s": round(t + n / a.fps, 2), "class": cls,
                     "is_cheating": int(cheat), "image": str(path)})
        t += n / a.fps
    vw.release()
    with open(a.gt, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {a.out} ({t:.0f}s) and {a.gt} ({sum(r['is_cheating'] for r in rows)} cheating segments)")


if __name__ == "__main__":
    main()
