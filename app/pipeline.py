"""
One pipeline for both recorded files and live streams.

Display and inference do not share a queue:

  decode thread  -> latest frame only (native fps, paced to 1x for files)
  preview thread -> JPEG of that frame with the freshest boxes drawn on it
  infer thread   -> newest frame at process_fps; older frames are dropped

Scores, flags and the chart use the captured frame's video time.
"""
import json
import os
import threading
import time
from collections import deque

import cv2
import numpy as np

from .classifier import configure_compute_threads, get_classifier, pick_device
from .clips import ClipRecorder, draw
from .config import CFG, SERIES_DIR
from .db import Flag, Job, SessionLocal, now
from .flagging import FlagEngine, TrackHistory, zone_of

P = CFG["pipeline"]
CROP_PAD = float(CFG["classifier"].get("crop_pad", 0) or 0)
_STREAM_OPEN = threading.Lock()


def _rate(stamps, now_m):
    """Events per second from monotonic timestamps in the last 1 s."""
    while stamps and now_m - stamps[0] > 1.0:
        stamps.popleft()
    if len(stamps) < 2:
        return float(len(stamps))
    span = stamps[-1] - stamps[0]
    if span < 0.2:
        return float(len(stamps))
    return (len(stamps) - 1) / span


def _wait_until(deadline, stop_event):
    while not stop_event.is_set():
        left = deadline - time.monotonic()
        if left <= 0:
            return True
        time.sleep(min(0.02, left))
    return False


def _sane_fps(value):
    try:
        fps = float(value)
    except (TypeError, ValueError):
        fps = 0.0
    if fps < 1 or fps > 120:
        return 25.0
    return fps


class InferClock:
    """Choose inference frames at process_fps. A late call takes the newest frame and drops the rest."""

    def __init__(self, process_fps):
        self.interval = 1.0 / max(float(process_fps), 0.1)
        self.last_t = None
        self.last_id = None

    def due(self, frame_id, video_t):
        if frame_id is None:
            return False
        if self.last_id == frame_id:
            return False
        if self.last_t is not None and float(video_t) < self.last_t + self.interval - 1e-3:
            return False
        return True

    def mark(self, frame_id, video_t):
        self.last_id = frame_id
        self.last_t = float(video_t)


def _predict_box(st, video_t):
    """Slide the last box by its recent velocity so it stays on the person between samples."""
    box = st["box"]
    prev, prev_t = st["prev_box"], st["prev_t"]
    age = float(video_t) - st["t"]
    if age <= 0 or not prev or prev_t is None:
        return list(box)
    dt = st["t"] - prev_t
    if dt <= 1e-3:
        return list(box)
    age = min(age, 1.0)
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    pcx = (prev[0] + prev[2]) / 2
    pcy = (prev[1] + prev[3]) / 2
    vx, vy = (cx - pcx) / dt, (cy - pcy) / dt
    limit = max(box[2] - box[0], box[3] - box[1], 1.0) * 4.0
    vx = max(-limit, min(limit, vx))
    vy = max(-limit, min(limit, vy))
    dx, dy = vx * age, vy * age
    return [box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy]


class OverlayTracks:
    """Last detections, moved forward in time and hidden once they are stale."""

    def __init__(self, fresh_seconds):
        self.fresh = float(fresh_seconds)
        self.tracks = {}
        self.lock = threading.Lock()

    def update(self, dets, video_t):
        with self.lock:
            for tid, box, score in dets:
                prev = self.tracks.get(tid)
                self.tracks[tid] = {
                    "box": [float(v) for v in box],
                    "t": float(video_t),
                    "score": float(score),
                    "prev_box": prev["box"] if prev else None,
                    "prev_t": prev["t"] if prev else None,
                }

    def visible(self, video_t):
        """(track_id, box, score, fade, det_time) for boxes still fresh at video_t."""
        out, dead = [], []
        with self.lock:
            for tid, st in self.tracks.items():
                age = float(video_t) - st["t"]
                if age > self.fresh:
                    dead.append(tid)
                    continue
                age = max(0.0, age)
                fade = 1.0 - age / self.fresh if self.fresh else 1.0
                out.append((tid, _predict_box(st, video_t), st["score"], fade, st["t"]))
            for tid in dead:
                del self.tracks[tid]
        return out


class FrameSlot:
    """Single latest frame. Publishing replaces; nothing is queued."""

    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.video_t = 0.0
        self.captured_mono = 0.0
        self.done = False

    def publish(self, frame, frame_id, video_t, captured_mono):
        with self.lock:
            # OpenCV reuses the decode buffer, so the slot keeps its own copy.
            self.frame = frame.copy()
            self.frame_id = int(frame_id)
            self.video_t = float(video_t)
            self.captured_mono = float(captured_mono)

    def snapshot(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame_id, self.video_t, self.captured_mono, self.frame.copy()

    def mark_done(self):
        with self.lock:
            self.done = True

    def is_done(self):
        with self.lock:
            return self.done


def drain_buffered(cap, max_drop=30):
    """Drop frames that are already queued, then stop.

    Used once when a camera is opened. Calling it on every frame would wait
    for the next picture as well and show the stream at half rate.
    """
    ok, frame = cap.read() if cap is not None and cap.isOpened() else (False, None)
    if not ok or frame is None:
        return False, None
    for _ in range(max_drop):
        started = time.perf_counter()
        if not cap.grab():
            break
        waited = time.perf_counter() - started
        ok2, newer = cap.retrieve()
        if ok2 and newer is not None:
            frame = newer
        # A grab that blocked is the live edge, not a backlog.
        if waited > 0.008:
            break
    return True, frame


class FileSource:
    """Video file. Realtime mode plays at the file fps on the wall clock.

    Fast mode yields sampled frames in order, as quickly as decode allows.
    video_t is always seconds into the file, not time spent processing.
    """
    live = False

    def __init__(self, path, realtime, stop_event):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open video {path}")
        self.src_fps = _sane_fps(self.cap.get(cv2.CAP_PROP_FPS))
        self.realtime = bool(realtime)
        self.stop_event = stop_event
        self.slot = FrameSlot()
        self.running = True
        self.failed = None
        self.thread = None
        self.clock_t0 = time.monotonic()
        if self.realtime:
            self.thread = threading.Thread(target=self._play, daemon=True)
            self.thread.start()

    def _play(self):
        idx = 0
        self.clock_t0 = time.monotonic()
        t0 = self.clock_t0
        try:
            while self.running and not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    break
                video_t = idx / self.src_fps
                idx += 1
                if not _wait_until(t0 + video_t, self.stop_event):
                    break
                self.slot.publish(frame, idx, video_t, time.monotonic())
        finally:
            self.slot.mark_done()

    def samples(self):
        """Fast path: every process_fps frame, in file order, with no wall-clock wait."""
        step = max(1, round(self.src_fps / P["process_fps"]))
        idx = 0
        try:
            while self.running and not self.stop_event.is_set():
                ok, frame = self.cap.read()
                if not ok or frame is None:
                    break
                video_t = idx / self.src_fps
                if idx % step == 0:
                    self.slot.publish(frame, idx + 1, video_t, time.monotonic())
                    yield video_t, frame
                idx += 1
        finally:
            self.slot.mark_done()

    def close(self):
        self.running = False
        self.stop_event.set()
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class StreamSource:
    """Phone / webcam. The reader keeps only the newest frame and drains the decoder buffer."""
    live = True

    def __init__(self, uri, stop_event):
        self.uri = int(uri) if str(uri).isdigit() else uri
        self.stop_event = stop_event
        self.slot = FrameSlot()
        self.running = True
        self.failed = None
        self.t0 = time.monotonic()
        self._opened = False
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _open(self):
        # Low-latency options apply only while this capture is created.
        # A tiny probesize breaks some files, so it is not set for the whole process.
        opts = "fflags;nobuffer|flags;low_delay|framedrop;1|max_delay;500000"
        with _STREAM_OPEN:
            previous = os.environ.get("OPENCV_FFMPEG_CAPTURE_OPTIONS")
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = opts
            try:
                if isinstance(self.uri, int):
                    cap = cv2.VideoCapture(self.uri)
                else:
                    cap = cv2.VideoCapture(self.uri, cv2.CAP_FFMPEG)
            finally:
                if previous is None:
                    os.environ.pop("OPENCV_FFMPEG_CAPTURE_OPTIONS", None)
                else:
                    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = previous
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _reader(self):
        S = CFG["stream"]
        tries = 0
        cap = self._open()
        try:
            while self.running and not self.stop_event.is_set():
                # One read per loop keeps the decoder empty without skipping live frames.
                # A fresh connection still drops whatever ffmpeg already queued.
                if not self._opened:
                    ok, frame = drain_buffered(cap)
                else:
                    ok, frame = cap.read() if cap.isOpened() else (False, None)
                if ok and frame is not None:
                    tries = 0
                    self._opened = True
                    captured = time.monotonic()
                    self.slot.publish(frame, self.slot.frame_id + 1, captured - self.t0, captured)
                    continue
                tries += 1
                if not self._opened and tries > 2:
                    self.failed = (f"Cannot open {self.uri}. Check the URL and that the phone "
                                   "and laptop are on the same Wi-Fi.")
                    break
                if tries > S["max_reconnects"]:
                    self.failed = f"Stream lost: could not reconnect to {self.uri}"
                    break
                print(f"[stream] no frames from {self.uri}, reconnect {tries}/{S['max_reconnects']}")
                cap.release()
                if not _wait_until(time.monotonic() + S["reconnect_seconds"], self.stop_event):
                    break
                cap = self._open()
        finally:
            self.slot.mark_done()
            cap.release()

    def close(self):
        self.running = False
        self.stop_event.set()


# ------------------------------------------------------------------ pipeline
class Pipeline(threading.Thread):
    def __init__(self, job_id, kind, uri, notify, realtime=None):
        super().__init__(daemon=True)
        self.job_id, self.kind, self.uri = job_id, kind, uri
        self.notify = notify                  # callback(dict) -> pushed to WebSocket clients
        if kind == "stream":
            self.realtime = True
        elif realtime is None:
            self.realtime = bool(P.get("realtime", True))
        else:
            self.realtime = bool(realtime)
        self.stop_event = threading.Event()
        self.preview = None                   # latest annotated JPEG (bytes)
        self.engine = FlagEngine()
        self.history = None                   # TrackHistory, created once the model is loaded
        self.recorder = ClipRecorder(P["process_fps"])
        self._frames_seen = 0
        self.source = None
        self.display_fps = 0.0
        self.infer_fps = 0.0
        self.latency_ms = None
        self.frame_t = None
        self.det_t = None
        self.det_age_ms = None
        self._overlay = OverlayTracks(P.get("overlay_fresh_seconds", 1.0))
        self._overlay_active = set()
        self._overlay_thr = None
        self._state_lock = threading.Lock()
        self._t_last = 0.0
        self._n = 0
        self._busy = 0.0
        self._t_start = None

    def stop(self):
        self.stop_event.set()
        if self.source is not None:
            self.source.close()

    def stats(self):
        frame_t = self.frame_t
        with self._state_lock:
            det_t = self.det_t
        if frame_t is None or det_t is None:
            det_age = self.det_age_ms
        else:
            det_age = max(0.0, (float(frame_t) - float(det_t)) * 1000.0)
        return {
            "display_fps": round(float(self.display_fps), 1),
            "infer_fps": round(float(self.infer_fps), 1),
            "latency_ms": None if self.latency_ms is None else round(float(self.latency_ms), 1),
            "frame_t": None if frame_t is None else round(float(frame_t), 3),
            "det_t": None if det_t is None else round(float(det_t), 3),
            "det_age_ms": None if det_age is None else round(float(det_age), 1),
            "realtime": self.realtime,
        }

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
        preview_thread = None
        self._update_job(status="running", started_at=now())
        self.notify({"type": "job", "job_id": self.job_id, "status": "running"})
        mode = "realtime" if self.realtime else "as fast as possible"
        print(f"[pipeline] job {self.job_id} {self.kind} {mode}, "
              f"sample {P['process_fps']} fps, preview width {P.get('preview_max_width') or 'full'}")
        try:
            configure_compute_threads()
            yolo = YOLO(D["weights"])
            clf = get_classifier()
            self.engine.threshold = clf.threshold
            self.recorder.threshold = clf.threshold
            self._overlay_thr = clf.threshold
            # Live view keeps a short window; a recorded file keeps every point.
            self.history = TrackHistory(
                bounded=self.kind == "stream",
                max_seconds=CFG["flagging"].get("score_history_seconds", 180),
                threshold=clf.threshold,
            )
            device = pick_device()
            source = StreamSource(self.uri, self.stop_event) if self.kind == "stream" \
                else FileSource(self.uri, self.realtime, self.stop_event)
            self.source = source
            self._t_start = time.monotonic()
            preview_thread = threading.Thread(target=self._preview_loop, args=(source, clf), daemon=True)
            preview_thread.start()
            if self.kind == "file" and not self.realtime:
                self._infer_fast(source, yolo, clf, device, D)
            else:
                self._infer_live(source, yolo, clf, device, D)
            for ev in self.engine.close_all(self._t_last):
                self._on_event(ev)
            self.recorder.flush()
            status = "stopped" if self.stop_event.is_set() else "done"
            self._finish_metrics(status)
        except Exception as e:
            status = "error"
            print(f"[pipeline] job {self.job_id} error: {e}")
            self._update_job(status="error", error=str(e), finished_at=now())
        finally:
            self.stop_event.set()
            if source:
                source.close()
            if preview_thread is not None:
                preview_thread.join(timeout=2)
            self._write_series()
        self.notify({"type": "job", "job_id": self.job_id, "status": status})

    def _finish_metrics(self, status):
        elapsed = max(time.monotonic() - (self._t_start or time.monotonic()), 1e-6)
        n = self._n
        self._update_job(status=status, finished_at=now(), frames_processed=n,
                         video_seconds=self._t_last, proc_seconds=self._busy,
                         avg_fps=n / elapsed,
                         avg_latency_ms=1000 * self._busy / max(n, 1))

    def _note_infer(self, video_t, spent):
        self._busy += spent
        self._n += 1
        self._t_last = video_t
        now_m = time.monotonic()
        marks = getattr(self, "_infer_marks", None)
        if marks is None:
            marks = deque()
            self._infer_marks = marks
        marks.append(now_m)
        self.infer_fps = _rate(marks, now_m)
        if self._n % 5 == 0 and self._t_start:
            elapsed = max(now_m - self._t_start, 1e-6)
            self._update_job(frames_processed=self._n, video_seconds=video_t, proc_seconds=self._busy,
                             avg_fps=self._n / elapsed, avg_latency_ms=1000 * self._busy / self._n)

    def _infer_live(self, source, yolo, clf, device, D):
        """Newest frame only, at process_fps. A slow call skips anything that arrived while it ran."""
        clock = InferClock(P["process_fps"])
        while not self.stop_event.is_set():
            if source.failed:
                raise RuntimeError(source.failed)
            snap = source.slot.snapshot()
            if snap is None:
                if source.slot.is_done():
                    break
                time.sleep(0.005)
                continue
            fid, video_t, _captured, frame = snap
            if not clock.due(fid, video_t):
                if source.slot.is_done():
                    self._t_last = max(self._t_last, video_t)
                    break
                time.sleep(0.004)
                continue
            clock.mark(fid, video_t)
            tic = time.perf_counter()
            self._process(video_t, frame, yolo, clf, device, D)
            self._note_infer(video_t, time.perf_counter() - tic)

    def _infer_fast(self, source, yolo, clf, device, D):
        """Every sampled frame, in order. Used when a file should be scored as fast as possible."""
        for video_t, frame in source.samples():
            if self.stop_event.is_set():
                break
            tic = time.perf_counter()
            self._process(video_t, frame, yolo, clf, device, D)
            self._note_infer(video_t, time.perf_counter() - tic)

    def _preview_loop(self, source, clf):
        last_id = -1
        last_enc = 0.0
        last_sync = 0.0
        disp_marks = deque()
        quality = int(P.get("preview_jpeg_quality", 60))
        max_w = int(P.get("preview_max_width") or 0)
        max_fps = float(P.get("preview_max_fps") or 0)
        min_gap = 1.0 / max_fps if max_fps > 0 else 0.0
        while not self.stop_event.is_set():
            snap = source.slot.snapshot()
            if snap is None:
                if source.slot.is_done() or source.failed:
                    break
                time.sleep(0.004)
                continue
            fid, video_t, captured, frame = snap
            if fid == last_id:
                if source.slot.is_done():
                    break
                time.sleep(0.002)
                continue
            now_m = time.monotonic()
            if min_gap and now_m - last_enc < min_gap:
                time.sleep(min(0.003, min_gap - (now_m - last_enc)))
                continue
            last_id = fid
            last_enc = now_m
            dets, active, thr = self._overlay_for(video_t)
            if source.live or not getattr(source, "realtime", True):
                latency_ms = max(0.0, (now_m - captured) * 1000.0)
            else:
                # How far this picture is behind a 1x clock (0 means it is on time).
                latency_ms = max(0.0, (now_m - (source.clock_t0 + video_t)) * 1000.0)
            with self._state_lock:
                det_t = self.det_t
            det_age_ms = None if det_t is None else max(0.0, (video_t - det_t) * 1000.0)
            self.latency_ms = latency_ms
            self.frame_t = video_t
            self.det_age_ms = det_age_ms
            disp_marks.append(now_m)
            self.display_fps = _rate(disp_marks, now_m)
            hud = (f"t={video_t:6.2f}s  display {self.display_fps:4.1f} fps  "
                   f"infer {self.infer_fps:4.1f} fps  latency {latency_ms:4.0f} ms")
            if det_t is not None:
                hud += f"  det {det_t:6.2f}s"
            label = "STUB MODEL - scores not real" if clf.is_stub else "Flags are for review only"
            img = render_preview(frame, dets, active, thr, f"{hud}  {label}", max_w)
            self.preview = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()
            if now_m - last_sync >= 0.25:
                last_sync = now_m
                self.notify({
                    "type": "sync", "job_id": self.job_id,
                    "frame_t": round(float(video_t), 3),
                    "det_t": None if det_t is None else round(float(det_t), 3),
                    "display_fps": round(float(self.display_fps), 1),
                    "infer_fps": round(float(self.infer_fps), 1),
                    "latency_ms": round(float(latency_ms), 1),
                    "det_age_ms": None if det_age_ms is None else round(float(det_age_ms), 1),
                })

    def _overlay_for(self, video_t):
        with self._state_lock:
            thr = self._overlay_thr if self._overlay_thr is not None else self.engine.threshold
            active = set(self._overlay_active)
        return self._overlay.visible(video_t), active, thr

    def _set_overlay(self, dets, active, thr, det_t):
        self._overlay.update(dets, det_t)
        with self._state_lock:
            self._overlay_active = set(active)
            self._overlay_thr = thr
            self.det_t = float(det_t)

    def _process(self, t, frame, yolo, clf, device, D):
        # t is the captured frame's video time. Inference may downscale a copy;
        # overlay boxes are mapped back onto the full frame the preview shows.
        scale = 1.0
        infer = frame
        max_w = int(P.get("max_frame_width") or 0)
        if max_w > 0 and frame.shape[1] > max_w:
            scale = max_w / frame.shape[1]
            infer = cv2.resize(frame, None, fx=scale, fy=scale)
        h, w = infer.shape[:2]

        res = yolo.track(infer, persist=True, classes=[0], conf=D["conf"], imgsz=D["imgsz"],
                         tracker=D["tracker"], device=device, verbose=False)[0]
        boxes, ids = [], []
        if res.boxes is not None and res.boxes.id is not None:
            for b, tid in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.id.int().cpu().tolist()):
                if (b[2] - b[0]) >= P["min_crop_px"] and (b[3] - b[1]) >= P["min_crop_px"]:
                    boxes.append(b)
                    ids.append(tid)

        crops, keep_boxes, keep_ids = [], [], []
        for box, tid in zip(boxes, ids):
            crop = cut_crop(infer, box, CROP_PAD)
            if crop is None:
                continue
            crops.append(crop)
            keep_boxes.append(box)
            keep_ids.append(tid)
        probs = clf.predict(crops)
        scores = clf.cheat_score(probs) if len(crops) else np.zeros(0)

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

        self.recorder.add(t, infer, dets)
        for ev in events:
            self._on_event(ev)
        self._sync_open_flags()
        self._frames_seen += 1
        if self._frames_seen % 15 == 0:
            self._write_series()
        self._publish_scores(t)
        full = dets if scale == 1.0 else [
            (tid, [c / scale for c in box], s) for tid, box, s in dets]
        active = {tid for tid, st in self.engine.tracks.items() if st.active_flag}
        self._set_overlay(full, active, clf.threshold, t)

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
            "type": "scores", "job_id": self.job_id, "t": t_key, "frame_t": t_key,
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


def render_preview(frame, dets, active, threshold, hud, max_w):
    """Resize for the JPEG only. Box coordinates stay aligned with this image."""
    img = frame
    if max_w and frame.shape[1] > max_w:
        s = max_w / frame.shape[1]
        nw = max(2, int(frame.shape[1] * s) // 2 * 2)
        nh = max(2, int(frame.shape[0] * s) // 2 * 2)
        img = cv2.resize(frame, (nw, nh))
        scaled = []
        for item in dets:
            tid, box, score = item[0], item[1], item[2]
            fade = item[3] if len(item) > 3 else 1.0
            det_t = item[4] if len(item) > 4 else None
            scaled.append((tid, [c * s for c in box], score, fade, det_t))
        dets = scaled
    img = draw(img, dets, threshold=threshold, active=active)
    cv2.putText(img, hud, (8, img.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    return img


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
