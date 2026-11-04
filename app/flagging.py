"""Temporal smoothing + flag rules (one state per tracked student).

The window, minimum duration and cooldown match the notebook Try-it video
section (WINDOW_S=3, MIN_DURATION_S=2, COOLDOWN_S=20). The threshold is the
one resolved from the checkpoint (see Classifier), not a hard-coded 0.6.
"""
import threading
from collections import deque

from .config import CFG

F = CFG["flagging"]


class TrackState:
    def __init__(self):
        self.scores = deque()          # (t, score) inside the rolling window
        self.above_since = None        # when the smoothed score went above threshold
        self.active_flag = None        # dict of the open flag event, if any
        self.cooldown_until = -1.0
        self.last_seen = 0.0


class FlagEngine:
    """Feed per-frame scores; get back 'open' and 'close' events."""

    def __init__(self, threshold=None):
        self.tracks = {}
        self.threshold = float(F["threshold"] if threshold is None else threshold)

    def update(self, t, track_id, score, extra):
        """extra: dict with bbox, zone, top_class for this frame. Returns (smooth, events)."""
        st = self.tracks.setdefault(track_id, TrackState())
        st.last_seen = t
        st.scores.append((t, score))
        while st.scores and st.scores[0][0] < t - F["window_seconds"]:
            st.scores.popleft()
        smooth = sum(s for _, s in st.scores) / len(st.scores)
        events = []

        if smooth >= self.threshold:
            if st.above_since is None:
                st.above_since = t
            if st.active_flag:                                 # flag already open: update it
                fl = st.active_flag
                fl["end_s"] = t
                fl["peak"] = max(fl["peak"], smooth)
                fl["sum"] += smooth
                fl["n"] += 1
            elif (t - st.above_since >= F["min_duration_seconds"] and t >= st.cooldown_until):
                st.active_flag = {"track_id": track_id, "start_s": st.above_since, "end_s": t,
                                  "peak": smooth, "sum": smooth, "n": 1, **extra}
                events.append(("open", st.active_flag))
        else:
            st.above_since = None
            if st.active_flag:
                events.append(("close", self._close(st, t)))
        return smooth, events

    def _close(self, st, t):
        fl = st.active_flag
        st.active_flag = None
        st.cooldown_until = t + F["cooldown_seconds"]
        return fl

    def expire(self, t):
        """Close flags of students that are no longer visible."""
        events = []
        for tid in list(self.tracks):
            st = self.tracks[tid]
            if t - st.last_seen > F["track_lost_seconds"]:
                if st.active_flag:
                    events.append(("close", self._close(st, st.last_seen)))
                del self.tracks[tid]
        return events

    def close_all(self, t):
        return [("close", self._close(st, t)) for st in self.tracks.values() if st.active_flag]


class TrackHistory:
    """Per-track score series for the dashboard chart.

    Live streams keep a bounded window (score_history_seconds). Recorded videos
    keep the whole series so GET /jobs/{id}/scores can return it after the job.
    """

    def __init__(self, bounded, max_seconds, threshold):
        self.bounded = bool(bounded)
        self.max_seconds = float(max_seconds)
        self.threshold = float(threshold)
        self.tracks = {}               # track_id -> deque|list of (t, raw, smooth)
        self.flags = {}                # flag id -> {id, track_id, start_s, end_s}
        self.lock = threading.Lock()

    def add(self, track_id, t, raw, smooth):
        point = (round(float(t), 3), round(float(raw), 4), round(float(smooth), 4))
        with self.lock:
            seq = self.tracks.get(track_id)
            if seq is None:
                seq = deque() if self.bounded else []
                self.tracks[track_id] = seq
            seq.append(point)
            if self.bounded:
                cutoff = t - self.max_seconds
                while seq and seq[0][0] < cutoff:
                    seq.popleft()
                for tid in list(self.tracks):
                    s = self.tracks[tid]
                    if s and s[-1][0] < cutoff:
                        del self.tracks[tid]

    def window(self, track_id, t0, t1):
        with self.lock:
            seq = list(self.tracks.get(track_id, ()))
        return [[t, sm] for t, _raw, sm in seq if t0 <= t <= t1]

    def upsert_flag(self, flag_id, track_id, start_s, end_s):
        with self.lock:
            self.flags[flag_id] = {
                "id": flag_id, "track_id": int(track_id),
                "start_s": round(float(start_s), 3), "end_s": round(float(end_s), 3),
            }

    def snapshot(self):
        with self.lock:
            tracks = [{
                "track_id": int(tid),
                "points": [{"t": t, "raw": raw, "smooth": sm} for t, raw, sm in seq],
            } for tid, seq in self.tracks.items() if seq]
            flags = list(self.flags.values())
        tracks.sort(key=lambda row: row["track_id"])
        flags.sort(key=lambda row: (row["start_s"], row["track_id"]))
        return {
            "threshold": self.threshold,
            "bounded": self.bounded,
            "history_seconds": self.max_seconds,
            "tracks": tracks,
            "flags": flags,
        }


def zone_of(bbox, w, h):
    """3x3 hall grid from the box centre: rows A-C (top->bottom), cols 1-3 (left->right)."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    col = min(int(3 * cx / w), 2)
    row = min(int(3 * cy / h), 2)
    return "ABC"[row] + str(col + 1)
