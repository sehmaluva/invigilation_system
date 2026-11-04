"""Keeps track of running pipelines and pushes events to WebSocket clients."""
import asyncio
import json

from .config import SERIES_DIR
from .pipeline import Pipeline


class Manager:
    def __init__(self):
        self.pipelines = {}           # job_id -> Pipeline
        self.clients = set()          # open WebSockets
        self.loop = None              # FastAPI event loop (set on startup)

    def start(self, job_id, kind, uri, realtime=None):
        p = Pipeline(job_id, kind, uri, self.notify, realtime=realtime)
        self.pipelines[job_id] = p
        p.start()
        return p

    def stop(self, job_id):
        p = self.pipelines.get(job_id)
        if p:
            p.stop()
        return p is not None

    def preview(self, job_id):
        p = self.pipelines.get(job_id)
        return p.preview if p else None

    def is_alive(self, job_id):
        p = self.pipelines.get(job_id)
        return bool(p and p.is_alive())

    def stats(self, job_id):
        p = self.pipelines.get(job_id)
        return p.stats() if p else None

    def scores(self, job_id):
        """In-memory series while the job is loaded, otherwise the file written at the end."""
        p = self.pipelines.get(job_id)
        if p is not None:
            snap = p.score_snapshot()
            if snap is not None:
                return snap
        path = SERIES_DIR / f"{job_id}.json"
        if path.is_file():
            return json.loads(path.read_text(encoding="utf-8"))
        return None

    # called from pipeline threads -> hand over to the asyncio loop
    def notify(self, msg):
        if self.loop:
            asyncio.run_coroutine_threadsafe(self._broadcast(json.dumps(msg)), self.loop)

    async def _broadcast(self, text):
        for ws in list(self.clients):
            try:
                await ws.send_text(text)
            except Exception:
                self.clients.discard(ws)


manager = Manager()
