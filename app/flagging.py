"""Temporal smoothing + flag rules (one state per tracked student)."""
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

    def __init__(self):
        self.tracks = {}

    def update(self, t, track_id, score, extra):
        """extra: dict with bbox, zone, top_class for this frame. Returns list of events."""
        st = self.tracks.setdefault(track_id, TrackState())
        st.last_seen = t
        st.scores.append((t, score))
        while st.scores and st.scores[0][0] < t - F["window_seconds"]:
            st.scores.popleft()
        smooth = sum(s for _, s in st.scores) / len(st.scores)
        events = []

        if smooth >= F["threshold"]:
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


def zone_of(bbox, w, h):
    """3x3 hall grid from the box centre: rows A-C (top->bottom), cols 1-3 (left->right)."""
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    col = min(int(3 * cx / w), 2)
    row = min(int(3 * cy / h), 2)
    return "ABC"[row] + str(col + 1)
