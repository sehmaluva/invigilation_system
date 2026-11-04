"""Ring buffer of recent frames + clip/thumbnail export for each flag."""
import threading
from collections import deque

import cv2

from .config import CFG, CLIP_DIR, THUMB_DIR

try:
    import imageio_ffmpeg                       # bundled ffmpeg -> browser-playable H.264 mp4
except ImportError:
    imageio_ffmpeg = None


def draw(frame, dets, highlight=None, threshold=None, active=None):
    """Draw person boxes.

    Each det is (track_id, bbox, smoothed_score) or
    (track_id, bbox, score, fade, det_time). fade is 1 when the detection
    matches this frame and falls toward 0 as it gets older. The optional
    det_time is the frame the score was measured on, in the same clock as the video.
    Red = flagged student, orange = score at or above the threshold, green = below it.
    """
    thr = CFG["flagging"]["threshold"] if threshold is None else threshold
    active = set(active or ())
    if highlight is not None:
        active.add(highlight)
    img = frame.copy()
    for item in dets:
        fade, det_t = 1.0, None
        if len(item) >= 5:
            tid, (x1, y1, x2, y2), s, fade, det_t = item[:5]
        elif len(item) == 4:
            tid, (x1, y1, x2, y2), s, fade = item
        else:
            tid, (x1, y1, x2, y2), s = item
        fade = max(0.0, min(1.0, float(fade)))
        hot = tid in active
        base = (0, 0, 255) if hot else (0, 165, 255) if s >= thr else (80, 200, 80)
        gain = 0.25 + 0.75 * fade
        color = tuple(int(c * gain) for c in base)
        thick = 3 if hot else 2
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
        label = f"ID {tid}  {s:.2f}"
        if det_t is not None:
            label += f"  {det_t:.1f}s"
        cv2.putText(img, label, (int(x1), max(int(y1) - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return img


def write_video(path, frames, fps):
    """Write BGR frames to mp4. Uses H.264 (plays in browsers) when ffmpeg is available."""
    h, w = frames[0].shape[:2]
    w2, h2 = w - w % 2, h - h % 2                          # H.264 needs even sizes
    if imageio_ffmpeg:
        gen = imageio_ffmpeg.write_frames(str(path), (w2, h2), fps=fps, codec="libx264",
                                          pix_fmt_out="yuv420p", macro_block_size=1,
                                          output_params=["-movflags", "+faststart"])
        gen.send(None)
        for f in frames:
            gen.send(cv2.cvtColor(f[:h2, :w2], cv2.COLOR_BGR2RGB).tobytes())
        gen.close()
    else:                                                 # fallback (may not play in every browser)
        vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w2, h2))
        for f in frames:
            vw.write(f[:h2, :w2])
        vw.release()


class ClipRecorder:
    """Keeps the last `pre_seconds` of frames; records `post_seconds` more after a flag."""

    def __init__(self, fps):
        self.fps = fps
        self.pre = CFG["clips"]["pre_seconds"]
        self.post = CFG["clips"]["post_seconds"]
        self.buffer = deque()                             # (t, frame, dets)
        self.pending = []                                 # clips still collecting frames
        self.threshold = None                             # set from the loaded checkpoint

    def add(self, t, frame, dets):
        self.buffer.append((t, frame, dets))
        while self.buffer and self.buffer[0][0] < t - self.pre:
            self.buffer.popleft()
        done = []
        for clip in self.pending:
            clip["frames"].append((t, frame, dets))
            if t >= clip["t_flag"] + self.post:
                done.append(clip)
        for clip in done:
            self.pending.remove(clip)
            self._save_async(clip)

    def start(self, flag_id, track_id, t_flag, on_saved):
        """Begin a clip for a new flag; returns the thumbnail path (written now)."""
        frames = list(self.buffer)
        t, frame, dets = frames[-1]
        thumb = THUMB_DIR / f"flag_{flag_id}.jpg"
        img = draw(frame, dets, highlight=track_id, threshold=self.threshold)
        scale = 480 / img.shape[1]
        cv2.imwrite(str(thumb), cv2.resize(img, None, fx=scale, fy=scale))
        self.pending.append({"flag_id": flag_id, "track_id": track_id, "t_flag": t_flag,
                             "frames": frames, "on_saved": on_saved})
        return str(thumb)

    def flush(self):
        """Source ended: save whatever has been collected."""
        for clip in self.pending:
            self._save_async(clip)
        self.pending = []

    def _save_async(self, clip):
        def work():
            path = CLIP_DIR / f"flag_{clip['flag_id']}.mp4"
            frames = [draw(f, d, highlight=clip["track_id"], threshold=self.threshold)
                      for _, f, d in clip["frames"]]
            try:
                write_video(path, frames, self.fps)
                clip["on_saved"](clip["flag_id"], str(path))
            except Exception as e:                        # never crash the pipeline over a clip
                print(f"[clips] failed to write {path}: {e}")
        threading.Thread(target=work, daemon=True).start()
