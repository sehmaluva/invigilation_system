"""Database tables (SQLite + SQLAlchemy)."""
from datetime import datetime, timezone

from sqlalchemy import (Column, DateTime, Float, ForeignKey, Integer, String,
                        Text, create_engine)
from sqlalchemy.orm import declarative_base, sessionmaker

from .config import DATA_DIR

engine = create_engine(f"sqlite:///{DATA_DIR / 'invigilation.db'}",
                       connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)
Base = declarative_base()


def now():
    return datetime.now(timezone.utc).replace(tzinfo=None)   # stored as UTC


class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String(64), unique=True, nullable=False)
    password_hash = Column(String(256), nullable=False)
    role = Column(String(20), default="invigilator")         # admin | invigilator
    created_at = Column(DateTime, default=now)


class Source(Base):
    """A video input: an uploaded file or a live stream (phone camera / webcam)."""
    __tablename__ = "sources"
    id = Column(Integer, primary_key=True)
    kind = Column(String(10))                                 # file | stream
    uri = Column(Text)                                        # file path or stream URL
    name = Column(String(200))
    created_at = Column(DateTime, default=now)


class Job(Base):
    """One processing run over a source, with timing metrics for evaluation."""
    __tablename__ = "jobs"
    id = Column(Integer, primary_key=True)
    source_id = Column(Integer, ForeignKey("sources.id"))
    status = Column(String(20), default="queued")             # queued|running|done|stopped|error
    created_by = Column(Integer, ForeignKey("users.id"), nullable=True)
    created_at = Column(DateTime, default=now)
    started_at = Column(DateTime)
    finished_at = Column(DateTime)
    frames_processed = Column(Integer, default=0)
    video_seconds = Column(Float, default=0.0)                # footage covered
    proc_seconds = Column(Float, default=0.0)                 # time spent analysing
    avg_fps = Column(Float, default=0.0)                      # frames analysed per second
    avg_latency_ms = Column(Float, default=0.0)               # per-frame detect+classify time
    error = Column(Text)


class Flag(Base):
    """A 'flagged for review' event for one tracked student."""
    __tablename__ = "flags"
    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id"))
    track_id = Column(Integer)
    start_s = Column(Float)                                   # seconds into video/session
    end_s = Column(Float)
    peak_score = Column(Float)
    mean_score = Column(Float)
    top_class = Column(String(50))
    zone = Column(String(10))                                 # 3x3 grid, e.g. "B2"
    clip_path = Column(Text)
    thumb_path = Column(Text)
    status = Column(String(20), default="pending")            # pending|confirmed|dismissed
    created_at = Column(DateTime, default=now)
    first_opened_at = Column(DateTime)                        # when an invigilator first opened it


class Review(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True)
    flag_id = Column(Integer, ForeignKey("flags.id"))
    user_id = Column(Integer, ForeignKey("users.id"))
    decision = Column(String(20))                             # confirmed | dismissed
    note = Column(Text)
    opened_at = Column(DateTime)
    decided_at = Column(DateTime, default=now)


class AuditLog(Base):
    __tablename__ = "audit_log"
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    action = Column(String(50))
    target = Column(String(100))
    detail = Column(Text)
    at = Column(DateTime, default=now)


def init_db():
    Base.metadata.create_all(engine)


def audit(db, user_id, action, target="", detail=""):
    db.add(AuditLog(user_id=user_id, action=action, target=str(target), detail=detail))
    db.commit()
