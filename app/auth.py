"""Simple login: salted PBKDF2 password hashes + signed bearer tokens."""
import hashlib
import hmac
import os
import secrets

from fastapi import Depends, HTTPException, Request
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from .config import CFG, DATA_DIR
from .db import SessionLocal, User, audit

TOKEN_HOURS = 12

# Secret key kept in data/secret.key (created once)
_key_file = DATA_DIR / "secret.key"
if not _key_file.exists():
    _key_file.write_text(secrets.token_hex(32))
_serializer = URLSafeTimedSerializer(_key_file.read_text().strip(), salt="invig")


def hash_password(pw: str) -> str:
    salt = os.urandom(16)
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 200_000)
    return salt.hex() + "$" + h.hex()


def check_password(pw: str, stored: str) -> bool:
    salt_hex, h_hex = stored.split("$")
    h = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt_hex), 200_000)
    return hmac.compare_digest(h.hex(), h_hex)


def make_token(user: User) -> str:
    return _serializer.dumps({"uid": user.id, "role": user.role})


def ensure_default_admin():
    db = SessionLocal()
    if db.query(User).count() == 0:
        u = User(username=CFG["server"]["default_admin_user"],
                 password_hash=hash_password(CFG["server"]["default_admin_password"]),
                 role="admin")
        db.add(u)
        db.commit()
        audit(db, u.id, "create_default_admin", u.username)
        print(f"[auth] Created default admin '{u.username}' - change the password!")
    db.close()


def current_user(request: Request) -> User:
    """Token from 'Authorization: Bearer ...' or ?token=... (needed for <img>/<video> tags)."""
    token = request.query_params.get("token")
    header = request.headers.get("authorization", "")
    if header.lower().startswith("bearer "):
        token = header[7:]
    if not token:
        raise HTTPException(401, "Not logged in")
    try:
        data = _serializer.loads(token, max_age=TOKEN_HOURS * 3600)
    except (BadSignature, SignatureExpired):
        raise HTTPException(401, "Session expired - log in again")
    db = SessionLocal()
    user = db.get(User, data["uid"])
    db.close()
    if not user:
        raise HTTPException(401, "Unknown user")
    return user


def admin_user(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Admin only")
    return user
