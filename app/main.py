"""FastAPI app: REST API + WebSocket + live preview + the dashboard (static/)."""
import asyncio
import json
import shutil
import time
from datetime import datetime
from pathlib import Path

from fastapi import (Depends, FastAPI, File, Form, HTTPException, Request, UploadFile,
                     WebSocket, WebSocketDisconnect)
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import func

from .auth import admin_user, check_password, current_user, ensure_default_admin, hash_password, make_token
from .classifier import get_classifier, get_classifier_info
from .config import CFG, ROOT, UPLOAD_DIR
from .db import AuditLog, Flag, Job, Review, SessionLocal, Source, User, audit, init_db, now
from .manager import manager
from .pipeline import flag_dict
from .retention import purge_old_media

app = FastAPI(title="AI-Assisted Exam Invigilation Support System")


@app.on_event("startup")
async def startup():
    init_db()
    ensure_default_admin()
    manager.loop = asyncio.get_running_loop()
    with SessionLocal() as db:                       # jobs interrupted by a restart
        for job in db.query(Job).filter(Job.status.in_(["queued", "running"])):
            job.status, job.error = "stopped", "server restarted"
        db.commit()
    purge_old_media()
    get_classifier()                                 # log model, classes, cheating index, threshold


# ------------------------------------------------------------------ auth
class LoginIn(BaseModel):
    username: str
    password: str


@app.post("/auth/login")
def login(body: LoginIn):
    with SessionLocal() as db:
        user = db.query(User).filter_by(username=body.username).first()
        if not user or not check_password(body.password, user.password_hash):
            audit(db, None, "login_failed", body.username)
            raise HTTPException(401, "Wrong username or password")
        audit(db, user.id, "login", user.username)
        return {"token": make_token(user), "username": user.username, "role": user.role}


class UserIn(BaseModel):
    username: str
    password: str
    role: str = "invigilator"


@app.post("/users")
def create_user(body: UserIn, admin: User = Depends(admin_user)):
    with SessionLocal() as db:
        if db.query(User).filter_by(username=body.username).first():
            raise HTTPException(400, "User exists")
        u = User(username=body.username, password_hash=hash_password(body.password), role=body.role)
        db.add(u)
        db.commit()
        audit(db, admin.id, "create_user", u.username, u.role)
        return {"id": u.id, "username": u.username, "role": u.role}


# ------------------------------------------------------------------ sources & jobs
def _as_bool(value, default):
    if value is None or value == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _new_job(kind, uri, name, user, realtime=None):
    with SessionLocal() as db:
        src = Source(kind=kind, uri=uri, name=name)
        db.add(src)
        db.commit()
        job = Job(source_id=src.id, created_by=user.id)
        db.add(job)
        db.commit()
        audit(db, user.id, f"start_{kind}", f"job {job.id}", name)
        job_id = job.id
    manager.start(job_id, kind, uri, realtime=realtime)
    return {"job_id": job_id, "source_id": src.id}


@app.post("/videos")
def upload_video(file: UploadFile = File(...), realtime: str = Form(None),
                 user: User = Depends(current_user)):
    """Upload a recorded exam video; it is processed in the background.

    realtime=true (the default) plays the preview at 1x. realtime=false analyses
    sampled frames as fast as the CPU allows, for evaluation.
    """
    dest = UPLOAD_DIR / f"{int(time.time())}_{Path(file.filename).name}"
    with open(dest, "wb") as f:
        shutil.copyfileobj(file.file, f)
    use_rt = _as_bool(realtime, CFG["pipeline"].get("realtime", True))
    return _new_job("file", str(dest), file.filename, user, realtime=use_rt)


class StreamIn(BaseModel):
    url: str              # e.g. http://192.168.1.5:8080/video  or "0" for the laptop webcam
    name: str = "Live session"


@app.post("/streams")
def start_stream(body: StreamIn, user: User = Depends(current_user)):
    return _new_job("stream", body.url.strip(), body.name, user)


@app.post("/streams/{job_id}/stop")
def stop_stream(job_id: int, user: User = Depends(current_user)):
    if not manager.stop(job_id):
        raise HTTPException(404, "No running job with that id")
    with SessionLocal() as db:
        audit(db, user.id, "stop_stream", f"job {job_id}")
    return {"ok": True}


def job_dict(job, src):
    info = {"id": job.id, "status": job.status, "kind": src.kind, "name": src.name,
            "created_at": job.created_at.isoformat() + "Z",
            "started_at": job.started_at.isoformat() + "Z" if job.started_at else None,
            "finished_at": job.finished_at.isoformat() + "Z" if job.finished_at else None,
            "frames_processed": job.frames_processed, "video_seconds": round(job.video_seconds or 0, 1),
            "proc_seconds": round(job.proc_seconds or 0, 1), "avg_fps": round(job.avg_fps or 0, 2),
            "avg_latency_ms": round(job.avg_latency_ms or 0, 1), "error": job.error,
            "live": manager.is_alive(job.id)}
    stats = manager.stats(job.id)
    if stats:
        info.update(stats)
    return info


@app.get("/jobs")
def list_jobs(user: User = Depends(current_user)):
    with SessionLocal() as db:
        rows = db.query(Job, Source).join(Source, Job.source_id == Source.id).order_by(Job.id.desc()).limit(50)
        return [job_dict(j, s) for j, s in rows]


@app.get("/jobs/{job_id}")
def get_job(job_id: int, user: User = Depends(current_user)):
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return job_dict(job, db.get(Source, job.source_id))


@app.get("/jobs/{job_id}/preview.mjpg")
async def preview(job_id: int, user: User = Depends(current_user)):
    """Live annotated preview (boxes, track IDs, smoothed scores) as an MJPEG stream."""
    async def gen():
        last = None
        while True:
            frame = manager.preview(job_id)
            if frame is not None and frame is not last:
                last = frame
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            if not manager.is_alive(job_id) and frame is last:
                break
            await asyncio.sleep(0.02)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


# ------------------------------------------------------------------ flags & review
@app.get("/flags")
def list_flags(status: str = "pending", limit: int = None, job_id: int = None,
               user: User = Depends(current_user)):
    """Ranked by peak score. `limit` is the alert cap (defaults to review.alert_cap)."""
    limit = limit or CFG["review"]["alert_cap"]
    with SessionLocal() as db:
        q = db.query(Flag)
        if status != "all":
            q = q.filter(Flag.status == status)
        if job_id:
            q = q.filter(Flag.job_id == job_id)
        return [flag_dict(f) for f in q.order_by(Flag.peak_score.desc()).limit(limit)]


@app.get("/flags/{flag_id}")
def get_flag(flag_id: int, user: User = Depends(current_user)):
    """Opening a flag records when review started (for average review time)."""
    with SessionLocal() as db:
        f = db.get(Flag, flag_id)
        if not f:
            raise HTTPException(404, "Flag not found")
        if not f.first_opened_at:
            f.first_opened_at = now()
            db.commit()
        reviews = [{"decision": r.decision, "note": r.note, "decided_at": r.decided_at.isoformat() + "Z",
                    "user": db.get(User, r.user_id).username}
                   for r in db.query(Review).filter_by(flag_id=flag_id)]
        thr = f.threshold if f.threshold is not None else get_classifier().threshold
        return {**flag_dict(f), "reviews": reviews,
                "threshold": thr, "score_series": _score_series(f.score_series)}


def _media(path, kind):
    if not path or not Path(path).exists():
        raise HTTPException(404, f"No {kind} (not ready yet, or deleted by retention policy)")
    return FileResponse(path)


@app.get("/flags/{flag_id}/clip")
def get_clip(flag_id: int, user: User = Depends(current_user)):
    with SessionLocal() as db:
        f = db.get(Flag, flag_id)
        audit(db, user.id, "view_clip", f"flag {flag_id}")
        return _media(f.clip_path if f else None, "clip")


@app.get("/flags/{flag_id}/thumbnail")
def get_thumb(flag_id: int, user: User = Depends(current_user)):
    with SessionLocal() as db:
        f = db.get(Flag, flag_id)
        return _media(f.thumb_path if f else None, "thumbnail")


class ReviewIn(BaseModel):
    decision: str         # confirmed | dismissed
    note: str = ""


@app.post("/flags/{flag_id}/review")
def review_flag(flag_id: int, body: ReviewIn, user: User = Depends(current_user)):
    if body.decision not in ("confirmed", "dismissed"):
        raise HTTPException(400, "decision must be 'confirmed' or 'dismissed'")
    with SessionLocal() as db:
        f = db.get(Flag, flag_id)
        if not f:
            raise HTTPException(404, "Flag not found")
        f.status = body.decision
        db.add(Review(flag_id=f.id, user_id=user.id, decision=body.decision, note=body.note,
                      opened_at=f.first_opened_at or now(), decided_at=now()))
        db.commit()
        audit(db, user.id, "review", f"flag {flag_id}", f"{body.decision}: {body.note}")
        manager.notify({"type": "flag_update", "flag": flag_dict(f)})
        return flag_dict(f)


# ------------------------------------------------------------------ stats
@app.get("/stats")
def stats(job_id: int = None, user: User = Depends(current_user)):
    with SessionLocal() as db:
        fq, jq = db.query(Flag), db.query(Job)
        if job_id:
            fq, jq = fq.filter(Flag.job_id == job_id), jq.filter(Job.id == job_id)
        flags, jobs = fq.all(), jq.all()
        counts = {s: sum(f.status == s for f in flags) for s in ("pending", "confirmed", "dismissed")}
        reviewed = counts["confirmed"] + counts["dismissed"]
        hours = sum(j.video_seconds or 0 for j in jobs) / 3600
        rq = db.query(Review)
        if job_id:
            rq = rq.join(Flag, Review.flag_id == Flag.id).filter(Flag.job_id == job_id)
        review_secs = [(r.decided_at - r.opened_at).total_seconds() for r in rq if r.opened_at]
        timed = [j for j in jobs if j.frames_processed]
        return {
            "total_flags": len(flags), **counts,
            "footage_hours": round(hours, 3),
            "flags_per_hour": round(len(flags) / hours, 2) if hours else None,
            "precision_reviewed": round(counts["confirmed"] / reviewed, 3) if reviewed else None,
            "avg_review_seconds": round(sum(review_secs) / len(review_secs), 1) if review_secs else None,
            "avg_processing_fps": round(sum(j.avg_fps for j in timed) / len(timed), 2) if timed else None,
            "avg_latency_ms": round(sum(j.avg_latency_ms for j in timed) / len(timed), 1) if timed else None,
            "jobs": len(jobs),
            "model": get_classifier_info(),
        }


@app.get("/audit")
def audit_log(limit: int = 100, admin: User = Depends(admin_user)):
    with SessionLocal() as db:
        rows = db.query(AuditLog).order_by(AuditLog.id.desc()).limit(limit)
        return [{"at": a.at.isoformat() + "Z", "user_id": a.user_id, "action": a.action,
                 "target": a.target, "detail": a.detail} for a in rows]


@app.post("/admin/retention/run")
def run_retention(days: int = None, admin: User = Depends(admin_user)):
    deleted = purge_old_media(days, admin.id)
    return {"deleted_files": len(deleted), "days": days if days is not None else CFG["retention"]["days"]}


@app.get("/jobs/{job_id}/scores")
def job_scores(job_id: int, user: User = Depends(current_user)):
    """Full per-track cheating-score series for a recorded video, or the live window for a stream."""
    with SessionLocal() as db:
        job = db.get(Job, job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        kind = db.get(Source, job.source_id).kind
    data = manager.scores(job_id)
    if data is None:
        info = get_classifier_info()
        data = {"job_id": job_id, "kind": kind, "threshold": info["threshold"],
                "bounded": kind == "stream",
                "history_seconds": CFG["flagging"].get("score_history_seconds", 180),
                "tracks": [], "flags": []}
    return data


@app.get("/config")
def public_config(user: User = Depends(current_user)):
    info = get_classifier_info()
    flagging = dict(CFG["flagging"])
    flagging["threshold"] = info["threshold"]
    return {"classes": info["class_names"], "cheating_classes": info["cheating_classes"],
            "cheating_index": info["cheating_index"], "flagging": flagging,
            "threshold": info["threshold"], "threshold_source": info["threshold_source"],
            "activation": info["activation"], "arch": info["arch"],
            "alert_cap": CFG["review"]["alert_cap"],
            "process_fps": CFG["pipeline"]["process_fps"],
            "score_history_seconds": CFG["flagging"].get("score_history_seconds", 180),
            "retention_days": CFG["retention"]["days"]}


def _score_series(raw):
    if not raw:
        return []
    try:
        return json.loads(raw)
    except ValueError:
        return []


# ------------------------------------------------------------------ websocket
@app.websocket("/ws")
async def ws(websocket: WebSocket):
    """Live push of new/updated flags and job status. Connect with ?token=..."""
    try:
        current_user(websocket)                       # same token check
    except HTTPException:
        await websocket.close(code=4401)
        return
    await websocket.accept()
    manager.clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()            # keep-alive pings from the page
    except WebSocketDisconnect:
        manager.clients.discard(websocket)


# ------------------------------------------------------------------ dashboard
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.get("/")
def home():
    return RedirectResponse("/static/index.html")
