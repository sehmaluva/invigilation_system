# AI-Assisted Exam Invigilation Support System (backend + dashboard)

CS/SWE 414 group project. This part covers the flag pipeline, clip export, the review dashboard, and logging of confirm/dismiss decisions.

The system **flags moments for review**. It never accuses anyone; the invigilator watches the clip and decides.

## How it works
```
camera / video -> YOLOv8n person detection + ByteTrack IDs -> crop each student
  -> classifier predict(crops) -> "cheat score" -> rolling mean per student (window_seconds)
  -> flag if it stays above threshold for min_duration_seconds (cooldown per student)
  -> save flag + thumbnail + clip (ring buffer: 8 s before, 3 s after) -> push to dashboard
  -> invigilator confirms or dismisses -> review + audit log -> stats
```
| File | What it does |
|---|---|
| `app/main.py` | FastAPI: API endpoints, WebSocket `/ws`, live preview, dashboard |
| `app/pipeline.py` | One pipeline for files **and** live streams (reconnects if the phone drops) |
| `app/classifier.py` | `predict(crops) -> probs`; loads `models/best.pt` (class list, cheating index, threshold), otherwise a STUB |
| `app/flagging.py` | Temporal smoothing, flag rules, 3x3 hall zones (A1..C3) |
| `app/clips.py` | Ring buffer, clip (H.264 mp4) and thumbnail export |
| `app/db.py` | SQLite tables: users, sources, jobs, flags, reviews, audit_log |
| `app/retention.py` | Deletes clips older than `retention.days` |
| `static/` | Dashboard: Login, Live, Review queue, Stats (plain HTML/JS) |
| `scripts/` | Synthetic test video, evaluation, fake phone stream, cleanup, add user |
| `config.yaml` | All settings (thresholds, fps, classes, model path, retention) |

## 1. Install (once)
Needs Python 3.10–3.12 (3.13 also works).

**Windows (PowerShell)**
```
cd invigilation_system
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
**Linux / macOS**
```
cd invigilation_system
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```
If you have an NVIDIA GPU, install the CUDA build of torch from pytorch.org instead of the `cpu` line.

## 2. Run
```
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Open **http://localhost:8000** and log in with `admin` / `admin123`. Change the password with:
`python scripts/create_user.py admin NEWPASSWORD admin`.
Add invigilators with `python scripts/create_user.py name password`.
On the first session the detector `yolov8n.pt` (6 MB) downloads automatically, so you need internet that first time.

## 3. Use your phone as the camera
1. Install **IP Webcam** (Android, Play Store). On iPhone, or as an alternative, use **DroidCam**.
2. Connect the phone and the laptop to the **same Wi-Fi**. A phone hotspot works too: connect the laptop to it.
3. In IP Webcam, scroll down and tap **Start server**. The screen shows an address like `http://192.168.1.5:8080`.
4. In the dashboard, open **Live** and paste **`http://192.168.1.5:8080/video`** (add `/video`), then click **Start**.
   - DroidCam: `http://PHONE_IP:4747/video`
   - Laptop webcam: type `0`
5. You see the live view with boxes, student IDs, and scores. Flags appear on the right and in **Review queue**.

Tips: put the phone on a stand, landscape, about 2–4 m from 3–6 people. In IP Webcam, set the video resolution to 640x480 or 1280x720. If it won't connect, open the URL in the laptop's browser first. If it doesn't load there, it's a Wi-Fi or firewall problem; university Wi-Fi often blocks device-to-device traffic, so use a hotspot.

**While the STUB model is in use** (no `best.pt`): a student's score rises when their box gets wider, for example when they lean sideways or stretch an arm out. Hold that for about 2 seconds above the threshold to trigger a test flag. Tune `classifier.stub` or `flagging.threshold` in `config.yaml`.

## 4. Load the notebook's best.pt
The Colab notebook `models/Exam_Cheating_TPU_Training.ipynb` trains EfficientNet-B0 with two classes, `["normal", "cheating"]`, and saves `best.pt` plus `metrics.json`. The checkpoint holds the architecture, the class list, and the precision-tuned threshold. The Try-it video section flags with that threshold.

1. Copy `best.pt` to **`models/best.pt`**. If you have `metrics.json`, put it in `models/` as well (used only when the checkpoint itself has no `threshold`).
2. Restart (`docker compose up --build`, or uvicorn).
3. Read the startup log. It prints the model path, arch, class list, **cheating output index**, activation, and the threshold with its source (`checkpoint`, `metrics.json`, or `config`).

`config.yaml` defaults match that Try-it path:

| Setting | Default | Notebook |
|---|---|---|
| `classifier.arch` | `efficientnet_b0` | `ARCH` |
| `classifier.class_names` | `normal`, `cheating` | binary head |
| `classifier.cheating_classes` | `cheating` | softmax column of that name |
| `classifier.crop_pad` | `0` | Try-it uses the tight box (training crops used 0.10) |
| `pipeline.process_fps` | `3` | `TRY_FPS` |
| `pipeline.max_frame_width` | `0` (no downscale) | Try-it keeps the full frame |
| `detector.conf` | `0.30` | `YOLO_CONF` |
| `flagging.window_seconds` | `3` | rolling mean |
| `flagging.min_duration_seconds` | `2` | flag after this long above the line |
| `flagging.cooldown_seconds` | `20` | per student |
| `flagging.threshold` | `0.5` | fallback only |

When `flagging.use_checkpoint_threshold` is true (the default), a threshold saved in `best.pt` replaces the 0.5 fallback. That is the value Try-it uses. Set the flag to false to force the yaml value. Class names and the cheating index also come from the checkpoint when it has them, so an old 6-class list in yaml cannot point at the wrong column. The notebook head is 2-way cross-entropy: the score is a **softmax**, not a sigmoid. A single-logit file still uses a sigmoid.

Preprocessing of each crop is BGR to RGB, divide by 255, resize to 224, ImageNet mean and std — the same as Try-it.

## 5. Score chart
The Live page draws one line per tracked student (smoothed cheating score), the threshold as a dashed line, and shaded bands for flagged periods. Points arrive on the WebSocket. A live session keeps the last `flagging.score_history_seconds` (180) on the server and in the browser. Opening a flag in **Review queue** shows that student's scores around the flag. For a recorded video, the full series is `GET /jobs/{id}/scores`.

## 6. Recorded video and evaluation
- Dashboard: **Live → Upload & analyse**. Or with the API: `POST /videos`.
- Build a test video with known "cheating" times from a dataset folder that has class subfolders:
  ```
  python scripts/make_synthetic_video.py --images PATH/TO/examdatasetnew/test --out data/synthetic.mp4 --gt data/synthetic_gt.csv
  ```
  Upload `data/synthetic.mp4`, note the job number, then run:
  ```
  python scripts/evaluate.py --job-id 1 --gt data/synthetic_gt.csv --cap 10
  ```
  This reports precision and recall at the alert cap, flags per hour, fps, and latency (Venon's metrics).
- Test "live" without a phone: `python scripts/fake_phone_stream.py some_video.mp4`, then start a session with `http://127.0.0.1:8081/video`.

## API (http://localhost:8000/docs for the full list)
`POST /auth/login` · `POST /videos` · `POST /streams` `{"url": "..."}` · `POST /streams/{id}/stop` ·
`GET /jobs/{id}` · `GET /jobs/{id}/preview.mjpg` · `GET /jobs/{id}/scores` · `GET /flags?status=pending&limit=10` ·
`GET /flags/{id}` · `GET /flags/{id}/clip` · `GET /flags/{id}/thumbnail` · `POST /flags/{id}/review` `{"decision": "confirmed|dismissed", "note": ""}` ·
`GET /stats` · `GET /audit` (admin) · `POST /admin/retention/run` (admin) · `WS /ws?token=...`

## Ethics and privacy
- Login is required, and every login, review, clip view, and session start is written to `audit_log`.
- Clips and thumbnails older than `retention.days` (14) are deleted at startup or with `python scripts/cleanup_clips.py`.
- Wording is always "flagged for review". Track IDs are not identities.
- Only film people who have agreed to it, for example a staged mock exam with classmates. A real exam needs permission from the department.

## Docker

From this folder:

```
docker compose up --build
```

Open **http://localhost:8000**. The image installs the CPU build of PyTorch, then `requirements.txt`, and starts the app with `python -m uvicorn` (the same Python that received those packages). Compose mounts only `./data` and `./models`. Those folders do not replace the libraries installed in the image.

## Troubleshooting

**The container exits with `ModuleNotFoundError: No module named 'sqlalchemy'`** (or another library imported by `app/`).

The running image was built without the current dependencies. `docker compose up` on its own reuses that image and does not run `pip install` again. Rebuild:

```
docker compose up --build
```

The Dockerfile runs `python -c "import app.main"` before it finishes the image. If a package is missing, the build stops with that import error. Add the library to `requirements.txt` (every third-party import in `app/` is listed there) and rebuild.

## Known limitations
- Until `best.pt` is added, scores come from a **STUB** and mean nothing.
- Clips are recorded at the processing rate (default 3 fps), so they look choppy.
- If a student leaves the frame and comes back, they may get a new track ID.
- Camera angle and distance matter: the datasets are close-up classroom shots.
- Detection is CPU-friendly (about 30–150 ms per frame on a laptop), but larger halls with many people get slower.

## AI assistance declaration
Parts of this code were generated with an AI assistant. The module's academic integrity policy requires you to **declare AI-assisted work** in your report and submission. Make sure you understand every part before you present it.
