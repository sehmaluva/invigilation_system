"""
One pipeline for both recorded files and live streams:

  frame -> YOLOv8 person detection + ByteTrack IDs -> crop each student
        -> classifier.predict(crops) -> cheat score -> rolling mean per track
        -> flag events -> DB + thumbnail + clip + WebSocket push
        -> per-track score series (live chart + GET /jobs/{id}/scores)
"""
import json
import threading
import time

import cv2
import numpy as np

from .classifier import get_classifier, pick_device
from .clips import ClipRecorder, draw
from .config import CFG, SERIES_DIR
from .db import Flag, Job, SessionLocal, now
from .flagging import FlagEngine, TrackHistory, zone_of

P = CFG["pipeline"]
CROP_PAD = float(CFG["classifier"].get("crop_pad", 0) or 0)


# ------------------------------------------------------------------ sources
class FileSource:
    """Reads a video file, skipping frames to reach process_fps. t = seconds into the video."""
    live = False

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {path}")
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.step = max(1, round(self.src_fps / P["process_fps"]))
        self.idx = 0

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        t = self.idx / self.src_fps
        for _ in range(self.step - 1):          # skip frames we won't analyse
            self.cap.grab()
        self.idx += self.step
        return t, frame

    def close(self):
        self.cap.release()


class StreamSource:
    """
    Live camera (phone IP-camera URL or webcam index). A reader thread always keeps
    only the newest frame so we never fall behind; reconnects if the stream drops.
    t = seconds since the session started.
    """
    live = True

    def __init__(self, uri):
        self.uri = int(uri) if str(uri).isdigit() else uri
        self.frame, self.frame_id, self.last_id = None, 0, 0
        self.running, self.failed = True, None
        self.t0 = time.monotonic()
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _open(self):
        cap = cv2.VideoCapture(self.uri)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self):
        S = CFG["stream"]
        tries = 0
        cap = self._open()
        while self.running:
            ok, frame = cap.read() if cap.isOpened() else (False, None)
            if ok:
                tries = 0
                with self.lock:
                    self.frame, self.frame_id = frame, self.frame_id + 1
                continue
            tries += 1                                   # stream dropped -> reconnect
            if self.frame_id == 0 and tries > 2:         # never worked: probably a wrong URL
                self.failed = (f"Cannot open {self.uri}. Check the URL and that the phone "
                               "and laptop are on the same Wi-Fi.")
                break
            if tries > S["max_reconnects"]:
                self.failed = f"Stream lost: could not reconnect to {self.uri}"
                break
            print(f"[stream] no frames from {self.uri}, reconnect {tries}/{S['max_reconnects']}")
            cap.release()
            time.sleep(S["reconnect_seconds"])
            cap = self._open()
        cap.release()

    def read(self):
        """Wait for a frame newer than the last one returned."""
        while self.running and not self.failed:
            with self.lock:
                if self.frame_id != self.last_id:
                    self.last_id = self.frame_id
                    return time.monotonic() - self.t0, self.frame
            time.sleep(0.01)
        return None, None

    def close(self):
        self.running = False


# ------------------------------------------------------------------ pipeline
class Pipeline(threading.Thread):
    def __init__(self, job_id, kind, uri, notify):
        super().__init__(daemon=True)
        self.job_id, self.kind, self.uri = job_id, kind, uri
        self.notify = notify                  # callback(dict) -> pushed to WebSocket clients
        self.stop_event = threading.Event()
        self.preview = None                   # latest annotated JPEG (bytes)
        self.engine = FlagEngine()
        self.history = None                   # TrackHistory, created once the model is loaded
        self.recorder = ClipRecorder(P["process_fps"])
        self._frames_seen = 0

    def stop(self):
        self.stop_event.set()

    # ---- DB helpers (each call uses its own session: we're in a thread)
    def _update_job(self, **fields):
        with SessionLocal() as db:
            job = db.get(Job, self.job_id)
            for k, v in fields.items():
                setattr(job, k, v)
            db.commit()

    def run(self):
        from ultralytics import YOLO          # imported here so the server starts fast
        D = CFG["detector"]
        source = None
        self._update_job(status="running", started_at=now())
        self.notify({"type": "job", "job_id": self.job_id, "status": "running"})
        try:
            yolo = YOLO(D["weights"])
            clf = get_classifier()
            self.engine.threshold = clf.threshold
            self.recorder.threshold = clf.threshold
            # Live view keeps a short window; a recorded file keeps every point.
            self.history = TrackHistory(
                bounded=self.kind == "stream",
                max_seconds=CFG["flagging"].get("score_history_seconds", 180),
                threshold=clf.threshold,
            )
            device = pick_device()
            source = StreamSource(self.uri) if self.kind == "stream" else FileSource(self.uri)
            n, busy, t_last, t_start = 0, 0.0, 0.0, time.monotonic()
            next_tick = time.monotonic()

            while not self.stop_event.is_set():
                if source.live:                       # pace live streams to process_fps
                    time.sleep(max(0.0, next_tick - time.monotonic()))
                    next_tick = time.monotonic() + 1.0 / P["process_fps"]
                t, frame = source.read()
                if frame is None:
                    if getattr(source, "failed", None):
                        raise RuntimeError(source.failed)
                    break
                tic = time.perf_counter()
                self._process(t, frame, yolo, clf, device, D)
                busy += time.perf_counter() - tic
                n, t_last = n + 1, t
                if n % 10 == 0:                        # timing metrics for Venon
                    self._update_job(frames_processed=n, video_seconds=t_last, proc_seconds=busy,
                                     avg_fps=n / max(time.monotonic() - t_start, 1e-6),
                                     avg_latency_ms=1000 * busy / n)

            for ev in self.engine.close_all(t_last):
                self._on_event(ev)
            self.recorder.flush()
            status = "stopped" if self.stop_event.is_set() else "done"
            self._update_job(status=status, finished_at=now(), frames_processed=n,
                             video_seconds=t_last, proc_seconds=busy,
                             avg_fps=n / max(time.monotonic() - t_start, 1e-6),
                             avg_latency_ms=1000 * busy / max(n, 1))
        except Exception as e:
            status = "error"
            print(f"[pipeline] job {self.job_id} error: {e}")
            self._update_job(status="error", error=str(e), finished_at=now())
        finally:
            self._write_series()
            if source:
                source.close()
        self.notify({"type": "job", "job_id": self.job_id, "status": status})

    def _process(self, t, frame, yolo, clf, device, D):
        # 1) optional downscale (0 = off, which matches the notebook Try-it path)
        max_w = int(P.get("max_frame_width") or 0)
        if max_w > 0 and frame.shape[1] > max_w:
            s = max_w / frame.shape[1]
            frame = cv2.resize(frame, None, fx=s, fy=s)
        h, w = frame.shape[:2]

        # 2) detect + track people (class 0 = person)
        res = yolo.track(frame, persist=True, classes=[0], conf=D["conf"], imgsz=D["imgsz"],
                         tracker=D["tracker"], device=device, verbose=False)[0]
        boxes, ids = [], []
        if res.boxes is not None and res.boxes.id is not None:
            for b, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist()):
                if (b[2] - b[0]) >= P["min_crop_px"] and (b[3] - b[1]) >= P["min_crop_px"]:
                    boxes.append(b)
                    ids.append(tid)

        # 3) classify crops in one batch (tight box when crop_pad is 0, like Try-it)
        crops, keep_boxes, keep_ids = [], [], []
        for box, tid in zip(boxes, ids):
            crop = cut_crop(frame, box, CROP_PAD)
            if crop is None:
                continue
            crops.append(crop)
            keep_boxes.append(box)
            keep_ids.append(tid)
        probs = clf.predict(crops)
        scores = clf.cheat_score(probs) if len(crops) else np.zeros(0)

        # 4) smoothing + flag rules, and the per-track series for the chart
        dets, events = [], []
        for tid, box, sc, pr in zip(keep_ids, keep_boxes, scores, probs):
            extra = {"zone": zone_of(box, w, h),
                     "top_class": "stub" if clf.is_stub else clf.top_class(pr)}
            smooth, ev = self.engine.update(t, tid, float(sc), extra)
            if self.history is not None:
                self.history.add(tid, t, float(sc), smooth)
            dets.append((tid, box.tolist(), smooth))
            events += ev
        events += self.engine.expire(t)

        # 5) ring buffer for clips, then handle events
        self.recorder.add(t, frame, dets)
        for ev in events:
            self._on_event(ev)
        self._sync_open_flags()
        self._frames_seen += 1
        if self._frames_seen % 15 == 0:
            self._write_series()
        self._publish_scores(t)

        # 6) annotated live preview
        active = {tid for tid, st in self.engine.tracks.items() if st.active_flag}
        img = draw(frame, dets, threshold=clf.threshold, active=active)
        label = "STUB MODEL - scores not real" if clf.is_stub else "Flags are for review only"
        cv2.putText(img, f"t={t:6.1f}s  people={len(dets)}  {label}", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        self.preview = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 75])[1].tobytes()

    def _on_event(self, event):
        kind, fl = event
        if kind == "open":
            with SessionLocal() as db:
                flag = Flag(job_id=self.job_id, track_id=fl["track_id"], start_s=fl["start_s"],
                            end_s=fl["end_s"], peak_score=fl["peak"], mean_score=fl["peak"],
                            zone=fl["zone"], top_class=fl["top_class"], status="pending",
                            threshold=self.engine.threshold,
                            score_series=json.dumps(self._series_for(fl)))
                db.add(flag)
                db.commit()
                fl["db_id"] = flag.id
                flag.thumb_path = self.recorder.start(flag.id, fl["track_id"], fl["end_s"],
                                                      self._clip_saved)
                db.commit()
                self.notify({"type": "flag_new", "flag": flag_dict(flag)})
            if self.history is not None:
                self.history.upsert_flag(fl["db_id"], fl["track_id"], fl["start_s"], fl["end_s"])
        elif kind == "close" and "db_id" in fl:
            with SessionLocal() as db:
                flag = db.get(Flag, fl["db_id"])
                flag.end_s, flag.peak_score = fl["end_s"], fl["peak"]
                flag.mean_score = fl["sum"] / fl["n"]
                flag.score_series = json.dumps(self._series_for(fl))
                db.commit()
                self.notify({"type": "flag_update", "flag": flag_dict(flag)})
            if self.history is not None:
                self.history.upsert_flag(fl["db_id"], fl["track_id"], fl["start_s"], fl["end_s"])

    def _clip_saved(self, flag_id, path):
        with SessionLocal() as db:
            flag = db.get(Flag, flag_id)
            flag.clip_path = path
            if self.history is not None:
                flag.score_series = json.dumps(
                    self.history.window(flag.track_id, flag.start_s - 10, flag.end_s + 4))
            db.commit()
            self.notify({"type": "flag_update", "flag": flag_dict(flag)})

    def _series_for(self, fl):
        if self.history is None:
            return []
        return self.history.window(fl["track_id"], fl["start_s"] - 10, fl["end_s"] + 4)

    def _sync_open_flags(self):
        if self.history is None:
            return
        for tid, st in self.engine.tracks.items():
            fl = st.active_flag
            if fl and fl.get("db_id"):
                self.history.upsert_flag(fl["db_id"], tid, fl["start_s"], fl["end_s"])

    def _publish_scores(self, t):
        if self.history is None:
            return
        snap = self.history.snapshot()
        t_key = round(float(t), 3)
        points = []
        for tr in snap["tracks"]:
            if not tr["points"]:
                continue
            last = tr["points"][-1]
            if last["t"] == t_key:
                points.append({"track_id": tr["track_id"], "t": last["t"],
                               "raw": last["raw"], "smooth": last["smooth"]})
        self.notify({
            "type": "scores", "job_id": self.job_id, "t": t_key,
            "threshold": snap["threshold"], "bounded": snap["bounded"],
            "history_seconds": snap["history_seconds"],
            "points": points, "flags": snap["flags"],
        })

    def _write_series(self):
        if self.history is None:
            return
        try:
            snap = self.score_snapshot()
            path = SERIES_DIR / f"{self.job_id}.json"
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(snap))
            tmp.replace(path)
        except Exception as e:
            print(f"[pipeline] could not save score series for job {self.job_id}: {e}")

    def score_snapshot(self):
        if self.history is None:
            return None
        snap = self.history.snapshot()
        snap["job_id"] = self.job_id
        snap["kind"] = self.kind
        return snap


def cut_crop(frame, box, pad):
    """Slice a person box the way the notebook Try-it video does (int bounds, optional pad)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    if pad:
        px, py = pad * (x2 - x1), pad * (y2 - y1)
        x1, y1, x2, y2 = x1 - px, y1 - py, x2 + px, y2 + py
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or crop.ndim != 3 or crop.shape[2] < 3:
        return None
    return crop


def flag_dict(f):
    return {"id": f.id, "job_id": f.job_id, "track_id": f.track_id,
            "start_s": round(f.start_s, 1), "end_s": round(f.end_s, 1),
            "peak_score": round(f.peak_score, 3), "mean_score": round(f.mean_score, 3),
            "top_class": f.top_class, "zone": f.zone, "status": f.status,
            "has_clip": bool(f.clip_path), "has_thumb": bool(f.thumb_path),
            "created_at": f.created_at.isoformat() + "Z" if f.created_at else None}
