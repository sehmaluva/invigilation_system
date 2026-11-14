# AI-Assisted Exam Invigilation Support System

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/classifier-EfficientNet--B0-EE4C2C.svg)](https://pytorch.org/)
[![YOLOv8](https://img.shields.io/badge/detector-YOLOv8n%20%2B%20ByteTrack-111F68.svg)](https://github.com/ultralytics/ultralytics)
[![Course](https://img.shields.io/badge/BUSE-CS%20%2F%20SWE%20414-0B3D0B.svg)](#course-context)
[![Status](https://img.shields.io/badge/decision--support-flags%20for%20review-important.svg)](#ethics--responsible-use)

An AI decision-support system that watches exam-hall video and **flags possible cheating for invigilator review**. It does not accuse a candidate and it does not decide the outcome. Every alert is a "flagged for review" event: the invigilator watches the clip, confirms or dismisses it, and that decision is written to an audit log.

**Offline evaluation on recorded hall video is the main academic deliverable.** The codebase also includes a live-camera demonstration path (phone or laptop webcam). Formal live deployment in a real exam hall is out of scope.

Repository: [sehmaluva/invigilation_system](https://github.com/sehmaluva/invigilation_system) (private).

---

## Table of contents

- [Course context](#course-context)
- [Team and roles](#team-and-roles)
- [Features](#features)
- [System architecture](#system-architecture)
- [Tech stack](#tech-stack)
- [Prerequisites](#prerequisites)
- [Installation and setup](#installation-and-setup)
  - [Create a virtual environment and install dependencies](#create-a-virtual-environment-and-install-dependencies)
  - [Set up the database and a user](#set-up-the-database-and-a-user)
  - [Run the server and open the dashboard](#run-the-server-and-open-the-dashboard)
- [Configuration](#configuration)
- [How to add the trained model](#how-to-add-the-trained-model)
- [Usage](#usage)
- [API endpoints](#api-endpoints)
- [Evaluation](#evaluation)
  - [Training data and model results](#training-data-and-model-results)
- [Ethics and responsible use](#ethics--responsible-use)
- [Project structure](#project-structure)
- [Contributing](#contributing)
- [Licence / academic use](#licence--academic-use)
- [Known limitations](#known-limitations)

---

## Course context

| | |
|---|---|
| **Module** SWE 414 - *Artificial Intelligence and Intelligent Systems* |
| **Institution** | Bindura University of Science Education |


This repository is the working system for that project: a flag pipeline, clip export, a review dashboard, and logging of confirm/dismiss decisions.

---

## Team and roles

| Name | Role |
|---|---|
| **Nigel Kutadzaushe** ([Nijel20](https://github.com/Nijel20)) | Project lead, ML engineer, and final integration |
| **Malvin T. Machingura** ([sehmaluva](https://github.com/sehmaluva)) | Backend, dashboard, and production-model training |

The GitHub repository is owned by `sehmaluva`. At the time of writing, `Nijel20` has write access. Grad-CAM is part of the evaluation plan (Venon's responsibility) but is **not implemented in this repository yet**.

---

## Features

- **Recorded-video analysis** - upload an exam-hall video; a background worker samples frames and runs the detection/classification pipeline.
- **Person detection and tracking** - YOLOv8n finds people; ByteTrack keeps a stable ID on each person for as long as they stay in view.
- **Behaviour classification** - each person crop (224×224, ImageNet normalisation) is scored by an EfficientNet-B0 binary classifier with classes `normal` and `cheating`.
- **Temporal flagging** - scores are smoothed with a rolling mean. A flag is opened only if the smoothed score stays above the threshold for a minimum duration, then a per-student cooldown applies.
- **Review queue** - pending flags are ranked by peak confidence, capped by `review.alert_cap` (default 10). Each flag has a thumbnail, an H.264 clip (ring buffer: 8 s before, 3 s after), a hall zone on a 3×3 grid (A1–C3), and a score series around the event.
- **Invigilator decisions** - confirm or dismiss a flag, optionally with a note. Decisions are stored and audited.
- **Score chart** - the Live page (Chart.js) draws one line per tracked student (smoothed cheating score), the threshold as a dashed line, and shaded bands for flagged periods. For a recorded job the full series is available from `GET /jobs/{id}/scores`.
- **Real-time preview** - uploaded files play at 1× by default. Display fps, inference fps, and latency are shown under the picture. Boxes are slid forward from the last detection and hidden after about one second.
- **Live-camera demo** - optional path using IP Webcam / DroidCam or the laptop webcam. Useful for a demonstration; not the formal evaluation target.
- **Authentication and audit** - login is required. Logins, reviews, clip views, and session starts are written to `audit_log`.
- **Retention** - clips and thumbnails older than `retention.days` (default 14) are deleted at startup or on demand.
- **Offline evaluation script** - compare a job's flags with a ground-truth CSV and report precision/recall at the alert cap, flags per hour, and processing time.

---

## System architecture

The main deliverable path is offline: a recorded video is uploaded, queued, and processed; flags are stored and reviewed on the dashboard.

```mermaid
flowchart TD
    A[Upload recorded exam video] --> B[Job queue]
    B --> C[Worker samples frames at ~3 fps]
    C --> D[YOLOv8n person detection<br/>+ ByteTrack IDs]
    D --> E[Crop each person<br/>224×224, ImageNet norm]
    E --> F[EfficientNet-B0 classifier<br/>classes: normal, cheating]
    F --> G[Rolling-mean smoothing<br/>~3 s window per track]
    G --> H[Flag if above threshold<br/>for min duration + cooldown]
    H --> I[Save flag, thumbnail,<br/>and clip from ring buffer]
    I --> J[(SQLite: jobs, flags,<br/>reviews, audit_log)]
    J --> K[FastAPI]
    K --> L[Dashboard review queue<br/>confirm or dismiss]
```

A live stream follows the same pipeline after capture. Decoding and the MJPEG preview run on the newest frame so a slow detector does not build up delay. Inference stays at about `pipeline.process_fps` (default 3) and drops stale frames.

---

## Tech stack

| Layer | Technology |
|---|---|
| API and dashboard server | FastAPI, Uvicorn, Pydantic |
| Database | SQLite via SQLAlchemy 2 |
| Auth | Session tokens (`itsdangerous`), hashed passwords |
| Detection / tracking | Ultralytics YOLOv8n, ByteTrack (`bytetrack.yaml`) |
| Classification | PyTorch, torchvision EfficientNet-B0 |
| Video I/O | OpenCV, imageio-ffmpeg (H.264 clips) |
| Frontend | Plain HTML / CSS / JS in `static/`, Chart.js (`static/chart.umd.min.js`) |
| Config | `config.yaml` (PyYAML) |
| Training | Colab notebook `models/Exam_Cheating_TPU_Training.ipynb` (EfficientNet-B0, emits `best.pt` and `metrics.json`) |

There is no `pyproject.toml`; Python dependencies are listed in `requirements.txt`.

---

## Prerequisites

- **Python 3.10–3.12** (3.13 also works).
- **pip**, and a virtual environment (the steps below create one).
- **Internet on first run**, so Ultralytics can download `yolov8n.pt` (~6 MB) if it is not already cached.
- Optional: an **NVIDIA GPU** and the CUDA build of PyTorch, for faster inference.
- A trained checkpoint at `models/best.pt`. Without it the system falls back to a **STUB** whose scores are not meaningful (they rise when a person's box gets wider).
- For the live-camera demo only: the phone and the laptop on the same Wi-Fi (or a phone hotspot), plus **IP Webcam** (Android) or **DroidCam**. University Wi-Fi often blocks device-to-device traffic.

---

## Installation and setup

Clone the private repository (you need access):

```bash
git clone https://github.com/sehmaluva/invigilation_system.git
cd invigilation_system
```

Place the trained model at `models/best.pt` if it is not already there (see [How to add the trained model](#how-to-add-the-trained-model)).

### Create a virtual environment and install dependencies

**Linux / macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

**Windows (PowerShell)**

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA build of PyTorch from [pytorch.org](https://pytorch.org/) instead of the `cpu` index URL, then install `requirements.txt`.

Keep the virtual environment activated for the commands below.

### Set up the database and a user

The SQLite database is created automatically the first time you start the server or run the user script. It lives at `data/invigilation.db` (`server.data_dir` in `config.yaml`). There is no separate migration command: `init_db()` creates the tables (`users`, `sources`, `jobs`, `flags`, `reviews`, `audit_log`).

On first start the app also creates a default admin account from `config.yaml`:

- Username: `admin`
- Password: `admin123` (set under `server.default_admin_password`)

**Change this password immediately** (this also creates the database if it does not exist yet):

```bash
python scripts/create_user.py admin NEWPASSWORD admin
```

Add invigilators with `python scripts/create_user.py NAME PASSWORD` (role defaults to `invigilator`). Pass a third argument of `admin` or `invigilator` to set the role. The same script resets an existing user's password.

### Run the server and open the dashboard

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open [http://localhost:8000](http://localhost:8000) and log in with the admin account you set above. Interactive API docs are at [http://localhost:8000/docs](http://localhost:8000/docs).

---

## Configuration

All runtime settings live in [`config.yaml`](config.yaml). Edit the file and restart the server. Important blocks:

```yaml
classifier:                      # notebook best.pt plugs in here
  model_path: models/best.pt     # if missing, a STUB model is used
  arch: efficientnet_b0          # used only when the checkpoint does not say arch
  input_size: 224                # overridden by the checkpoint when it stores input_size
  crop_pad: 0.0                  # Try-it scores the tight box
  class_names: [normal, cheating]
  cheating_classes: [cheating]
  batch_size: 16

flagging:
  window_seconds: 3              # rolling-mean window
  threshold: 0.5                 # fallback only
  use_checkpoint_threshold: true
  min_duration_seconds: 2        # score must stay above threshold this long
  cooldown_seconds: 20           # no new flag for the same student during this time
```

Related defaults that match the notebook's Try-it path:

| Setting | Default | Meaning |
|---|---|---|
| `pipeline.process_fps` | `3` | Frames analysed per second |
| `pipeline.max_frame_width` | `0` | No downscale on inference (full frame) |
| `pipeline.realtime` | `true` | Uploaded files play at 1× unless overridden |
| `pipeline.overlay_fresh_seconds` | `1.0` | Hide stale live boxes after this many seconds |
| `pipeline.torch_threads` | `2` | Limit Torch threads so the preview is not starved |
| `detector.weights` | `yolov8n.pt` | Downloaded automatically on first run |
| `detector.conf` | `0.30` | YOLO confidence |
| `detector.tracker` | `bytetrack.yaml` | Person tracker |
| `clips.pre_seconds` / `post_seconds` | `8` / `3` | Clip length around a flag |
| `review.alert_cap` | `10` | Max pending flags shown at once (highest score first) |
| `retention.days` | `14` | Age after which clips and thumbnails are deleted |
| `flagging.score_history_seconds` | `180` | Live-chart history kept per track |

**Threshold resolution.** When `flagging.use_checkpoint_threshold` is `true` (the default), the threshold is taken from, in order: the `threshold` field inside `best.pt`, then `models/metrics.json` or `models/threshold.json` beside the checkpoint, then the YAML fallback of `0.5`. Set `use_checkpoint_threshold: false` to force the YAML value. Class names, architecture, input size, and the cheating-class index also come from the checkpoint when it contains them, so a stale class list in YAML cannot point at the wrong column.

The notebook head is 2-way cross-entropy: the cheating score is a **softmax** probability, not a sigmoid. A single-logit file still uses a sigmoid. Crop preprocessing is BGR→RGB, ÷255, resize to 224, ImageNet mean and std - the same as the notebook Try-it path.

---

## How to add the trained model

The Colab notebook [`models/Exam_Cheating_TPU_Training.ipynb`](models/Exam_Cheating_TPU_Training.ipynb) trains EfficientNet-B0 with classes `["normal", "cheating"]` and writes `best.pt` plus `metrics.json`. The checkpoint is a dict that can include `arch`, `model_state_dict`, `class_names`, `cheating_classes`, `threshold`, `input_size`, `mean`, and `std`.

1. Copy the trained file to **`models/best.pt`**. If you have `metrics.json`, put it in `models/` as well (used only when the checkpoint itself has no `threshold`).
2. Stop the running uvicorn process if it is already up, then start it again with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
3. Read the startup log. It prints the model path, architecture, class list, cheating output index, activation, and the threshold with its source (`checkpoint`, `metrics.json`, or `config`).

Until `models/best.pt` exists, the classifier is a **STUB**: a student's score rises when their bounding box gets wider (for example when they lean or stretch an arm). That is only for wiring tests. Do not treat stub scores as evidence of cheating.

A `best.pt` file may already be present in `models/` in some checkouts. `.gitignore` lists `*.pt`, so do not assume a teammate's trained weights were committed. Always confirm the startup log.

The notebook itself (`models/README.txt`) also describes a Try-it section that flags a video with the same 3 fps / 3 s mean / 2 s minimum / 20 s cooldown rules as this backend.

---

## Usage

1. Start the server and open [http://localhost:8000](http://localhost:8000).
2. Log in (see [Installation and setup](#installation-and-setup)).
3. On **Live**, either:
   - **Upload & analyse** a recorded exam video (the main deliverable path), or
   - start a live session by pasting a phone-camera URL (`http://PHONE_IP:8080/video` for IP Webcam, `http://PHONE_IP:4747/video` for DroidCam) or `0` for the laptop webcam.
4. **Play at real time** is checked by default, so a file preview runs at the video's own frame rate while detection stays at `process_fps` (about 3). Uncheck it, or post `realtime=false` to `/videos`, to score the file as fast as the CPU can. Flag times and the chart are always seconds into the video, either way.
5. Watch the annotated preview (boxes, track IDs, smoothed scores). Incoming flags appear on the right. Green = below threshold, orange = above it, red = flagged.
6. Open **Review queue**. Flags are listed highest score first, at most `alert_cap` at a time. Open a flag, watch the clip, and **confirm** or **dismiss** it (optional note).
7. Open **Stats** for flag counts, flags per hour, reviewed precision, average review time, and processing fps / latency.

**Synthetic evaluation video** (from a dataset folder with class subfolders):

```bash
python scripts/make_synthetic_video.py --images PATH/TO/examdatasetnew/test --out data/synthetic.mp4 --gt data/synthetic_gt.csv
```

Upload `data/synthetic.mp4`, note the job number, then run the evaluation script in the next section.

**Live demo without a phone:**

```bash
python scripts/fake_phone_stream.py some_video.mp4
```

Then start a session with `http://127.0.0.1:8081/video`.

Manual clip cleanup: `python scripts/cleanup_clips.py`.

---

## API endpoints

All routes below except `POST /auth/login` and `GET /` require a logged-in user (send the token from login). Admin-only routes are marked. Full interactive list: [http://localhost:8000/docs](http://localhost:8000/docs).

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/auth/login` | Exchange username and password for a token |
| `POST` | `/users` | Create a user (admin) |
| `POST` | `/videos` | Upload a recorded video. Optional form field `realtime=true\|false` (default follows `pipeline.realtime`) |
| `POST` | `/streams` | Start a live session, body `{"url": "...", "name": "..."}` |
| `POST` | `/streams/{job_id}/stop` | Stop a running live job |
| `GET` | `/jobs` | Recent jobs (status, fps, latency, footage seconds) |
| `GET` | `/jobs/{id}` | One job's status and metrics |
| `GET` | `/jobs/{id}/preview.mjpg` | Annotated MJPEG preview |
| `GET` | `/jobs/{id}/scores` | Full per-track score series for a recorded video, or the live history window for a stream |
| `GET` | `/flags?status=pending&limit=10` | Review queue, highest `peak_score` first. `limit` defaults to `review.alert_cap`. Optional `job_id` |
| `GET` | `/flags/{id}` | Flag detail, reviews, threshold, and score series around the event |
| `GET` | `/flags/{id}/clip` | Stream the flag's video clip (audited) |
| `GET` | `/flags/{id}/thumbnail` | Flag thumbnail |
| `POST` | `/flags/{id}/review` | Body `{"decision": "confirmed"\|"dismissed", "note": ""}` |
| `GET` | `/stats` | Flag counts, flags per hour, reviewed precision, review time, processing metrics, model info. Optional `job_id` |
| `GET` | `/config` | Public runtime config (classes, threshold and its source, flagging, alert cap, fps, retention) |
| `GET` | `/audit` | Audit log (admin) |
| `POST` | `/admin/retention/run` | Delete expired clips and thumbnails (admin) |
| `WS` | `/ws?token=...` | Live push of new or updated flags and job status |
| `GET` | `/` | Redirects to the dashboard at `/static/index.html` |

---

## Evaluation

Offline evaluation on recorded hall video is the project's main success criterion. After a job finishes, compare its flags with a ground-truth CSV (from `scripts/make_synthetic_video.py` or a hand-labelled file):

```bash
python scripts/evaluate.py --job-id 1 --gt data/synthetic_gt.csv --cap 10
```

A flag is a true positive if its `[start_s, end_s]` overlaps a cheating interval, with a default tolerance of 1 second (`--tolerance`). The script prints JSON with:

- **precision and recall at the alert cap** (top-K flags by peak score, default `review.alert_cap`) and again for **all flags**
- true-positive and false-positive counts, cheating intervals, and intervals found
- **flags per hour**
- **processing time**: `avg_fps`, `avg_latency_ms`, `video_seconds`, `proc_seconds`

F1 is not printed by the script; it can be computed from the precision and recall it reports. The counts are interval-overlap matches, not a per-frame confusion matrix.

The dashboard **Stats** page adds operational figures: pending / confirmed / dismissed counts, reviewed precision (`confirmed / reviewed`), average review time, and per-job processing fps and latency.

### Training data and model results

The production EfficientNet-B0 was fine-tuned in Google Colab on a TPU (PyTorch, bfloat16, seed 42) from ImageNet weights, with a two-class head and 224 px crops. Training used batch size 64, a frozen-backbone head stage (3 epochs, learning rate 1e-3) then full-network fine-tuning (up to 25 epochs, learning rate 1e-4, weight decay 1e-4, early-stopping patience 5 on validation macro-F1), and class-weighted cross-entropy. Person crops were taken with YOLOv8n (confidence 0.3, 10% pad, crop when the person covers less than 60% of the image, minimum crop 40 px). Files marked as flips or copies were dropped, and near-duplicates were linked by perceptual hash (Hamming distance ≤ 4). Validation was five-fold.

**Datasets.** Training used two public still-image collections: `examdatasetnew` and the Kaggle Exam Cheating Dataset (`ardutraagiginting/exam-cheating-dataset`). After cleaning, the combined set had **3,658 images**: 2,175 cheating and 1,483 normal (2,096 from examdatasetnew, 1,562 from ExamCheatingDataset). The Hashemite student-cheating video dataset (`fairouzfarouq/student-cheating-behavior-dataset`) is listed in the notebook configuration but contributed no samples to this run. The OEP dataset is not used.

The tables below are taken from `SmartInvigilation_runs/tpu_effnet/metrics.json` (out-of-fold predictions pooled across the five folds). Average precision (AP) is the area under the precision–recall curve (PR-AUC). AP does not depend on the operating threshold, so the same AP is shown at both thresholds for a given experiment. ROC-AUC was not computed; it can be obtained from `oof_predictions.csv`.

**Per-fold results at the default threshold 0.5**

| Fold | Best epoch | Precision | Recall | Macro-F1 | AP | n |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6 | 0.866 | 0.826 | 0.819 | 0.920 | 627 |
| 1 | 16 | 0.785 | 0.812 | 0.768 | 0.844 | 631 |
| 2 | 16 | 0.860 | 0.897 | 0.828 | 0.943 | 783 |
| 3 | 3 | 0.843 | 0.858 | 0.811 | 0.885 | 837 |
| 4 | 12 | 0.834 | 0.848 | 0.803 | 0.905 | 780 |

**Pooled cross-validation** (n = 3,658)

| Setting | Precision | Recall | Cheating F1 | Macro-F1 | AP (PR-AUC) | Accuracy | Confusion (TN, FP, FN, TP) |
|---|---:|---:|---:|---:|---:|---:|---|
| Threshold 0.50 | 0.839 | 0.852 | - | 0.807 | 0.902 | 0.815 | 1,128 / 355 / 322 / 1,853 |
| Deployed threshold 0.377 | 0.800 | 0.890 | 0.843 | 0.789 | 0.902 | 0.802 | 1,000 / 483 / 240 / 1,935 |

The deployed threshold is **0.3768**, chosen on the out-of-fold predictions to hit a target precision of 0.80. Relative to the default threshold of 0.5, this raises recall from 0.852 to 0.890 while holding precision at the 0.80 target. The backend reads this threshold from `best.pt` or `metrics.json` when `flagging.use_checkpoint_threshold` is true. Cheating-class F1 at the default threshold 0.5 was not stored in `metrics.json` and is therefore not reported.

**Cross-dataset tests** (train on one collection, test on the other)

| Train → test | n | Threshold | Precision | Recall | Macro-F1 | AP |
|---|---:|---:|---:|---:|---:|---:|
| examdatasetnew → ExamCheatingDataset | 1,562 | 0.50 | 0.630 | 0.633 | 0.549 | 0.663 |
| examdatasetnew → ExamCheatingDataset | 1,562 | 0.377 | 0.607 | 0.749 | 0.517 | 0.663 |
| ExamCheatingDataset → examdatasetnew | 2,096 | 0.50 | 0.671 | 0.774 | 0.605 | 0.703 |
| ExamCheatingDataset → examdatasetnew | 2,096 | 0.377 | 0.657 | 0.844 | 0.588 | 0.703 |

Macro-F1 falls to about 0.55–0.61 when the camera setup changes. That limited transfer is why every flag is reviewed by an invigilator, and why more hall-specific video is the next data priority.

These figures are **crop-level classifier scores**, not the event-level video metrics from `scripts/evaluate.py` (precision and recall at the alert cap, flags per hour, processing time). Those still come from a labelled hall video or the synthetic test video, as described above.

**Explainability.** Grad-CAM heatmaps on flagged clips are planned as part of the evaluation and documentation work. They are **not in this repository yet**.

Unit tests for checkpoint loading and the score-series path live in `tests/test_checkpoint_and_scores.py`. `pytest` is not listed in `requirements.txt`; install it separately if you want to run those tests.

---

## Ethics and responsible use

This system is a **decision-support tool**. It flags moments for review. It does not determine that a candidate cheated. Track IDs are not identities.

| Principle | How the system supports it |
|---|---|
| Human in the loop | An invigilator must watch the clip and confirm or dismiss every flag. Wording in the UI is always "flagged for review". |
| Auditability | Login, failed login, review, clip view, and session start are written to `audit_log`. Admin users can read the log at `GET /audit`. |
| Data retention | Clips and thumbnails older than `retention.days` (14 by default) are deleted at startup, via `POST /admin/retention/run`, or `python scripts/cleanup_clips.py`. |
| Access control | The dashboard and API require a login. Roles are `admin` and `invigilator`. Change the default admin password before any shared use. |
| Candidate privacy | Only film people who have agreed to it - for example a staged mock exam with classmates. A real exam needs permission from the department. Do not publish raw hall footage or clips. |
| Fairness and bias | Camera angle, distance, lighting, clothing, and seating position all affect detection and classification. Results from close-up classroom datasets may not transfer to a large hall. Treat error rates as part of the evaluation write-up, not as proof of guilt. |
| Scope | Formal live deployment in a production exam is out of scope for this course project. |

---

## Project structure

Verified from the repository tree on `main`:

```text
invigilation_system/
├── Dockerfile
├── docker-compose.yml
├── README.md
├── config.yaml
├── requirements.txt
├── .dockerignore
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py          # FastAPI: API, WebSocket /ws, live preview, dashboard
│   ├── pipeline.py      # Shared pipeline for files and live streams
│   ├── classifier.py    # predict(crops) → probs; loads models/best.pt or STUB
│   ├── flagging.py      # Rolling mean, threshold / min duration / cooldown, 3×3 zones
│   ├── clips.py         # Ring buffer, H.264 clip and thumbnail export
│   ├── db.py            # SQLite: users, sources, jobs, flags, reviews, audit_log
│   ├── auth.py          # Login, tokens, password hashing
│   ├── config.py        # Loads config.yaml
│   ├── manager.py       # Background jobs, preview frames, WebSocket fan-out
│   └── retention.py     # Deletes clips older than retention.days
├── models/
│   ├── Exam_Cheating_TPU_Training.ipynb
│   ├── README.txt
│   ├── PUT_best.pt_HERE.txt
│   ├── best.pt          # Trained checkpoint (place or replace here)
│   ├── best_stub.pt     # ImageNet backbone + untrained head (integration only)
│   ├── make_stub.py
│   └── src/build_nb.py  # Regenerates the training notebook
├── scripts/
│   ├── evaluate.py
│   ├── make_synthetic_video.py
│   ├── fake_phone_stream.py
│   ├── create_user.py
│   └── cleanup_clips.py
├── static/
│   ├── index.html       # Login, Live, Review queue, Stats
│   ├── app.js
│   ├── style.css
│   └── chart.umd.min.js
└── tests/
    └── test_checkpoint_and_scores.py
```

`data/` is created at runtime and is gitignored. It holds `invigilation.db`, uploads, clips, and thumbnails.

---

## Contributing

This is a private course-project repository. Work on a feature branch, open a pull request against `main`, and wait for review before merging. Keep commits focused and write messages that say what changed and why.

Suggested workflow:

```bash
git checkout main
git pull
git checkout -b feature/short-description
# ... work, then:
git add -p
git commit -m "Describe the change"
git push -u origin feature/short-description
```

Do not commit secrets, raw exam footage, or large unrelated binaries. `.gitignore` already excludes `.venv/`, `data/`, `__pycache__/`, `*.pt`, `*.mp4`, and `*.zip`. If a teammate needs the trained weights, share `best.pt` out of band or confirm it is already in `models/` before assuming it was pushed.

**Course policy: any AI-assisted work must be declared** in the technical report, the presentation, and in pull requests or commit messages where it is practical to do so. That includes code, documentation, and evaluation write-ups produced with assistance.

---

## Licence / academic use

This repository has no licence file and is a private academic project for CS 414 / SWE 414 at Bindura University of Science Education. It is intended for coursework, demonstration, and examination by the teaching team.

It is not licensed for production invigilation, commercial use, or redistribution of models, videos, or student footage. Third-party components (PyTorch, Ultralytics YOLOv8, FastAPI, Chart.js, and others) remain under their own licences.

---

## Known limitations

- Clips are recorded at the processing rate (default 3 fps), so they look choppy even when the live preview plays at the camera or file rate.
- If a student leaves the frame and comes back, ByteTrack may assign a new ID.
- Boxes older than `overlay_fresh_seconds` (1 s) are hidden rather than left sitting where the person used to be.
- Camera angle and distance matter. The training and Try-it material is based on relatively close classroom shots; a wide hall will be harder.
- Detection is intended to be CPU-friendly, but a larger hall with many people is slower. Torch is limited to two threads by default so the preview stays responsive.
- Grad-CAM explainability is not produced by the current code. Crop-level confusion matrices from training are in `metrics.json` (see [Training data and model results](#training-data-and-model-results)); `scripts/evaluate.py` reports interval-overlap matches, not a per-frame confusion matrix.
- Formal live deployment in a real examination is out of scope.
