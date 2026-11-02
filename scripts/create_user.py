"""Add a user or reset a password:  python scripts/create_user.py NAME PASSWORD [admin|invigilator]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.auth import hash_password                 # noqa: E402
from app.db import SessionLocal, User, audit, init_db  # noqa: E402

if len(sys.argv) < 3:
    sys.exit(__doc__)
name, pw = sys.argv[1], sys.argv[2]
role = sys.argv[3] if len(sys.argv) > 3 else "invigilator"
init_db()
with SessionLocal() as db:
    u = db.query(User).filter_by(username=name).first()
    if u:
        u.password_hash, u.role = hash_password(pw), role
    else:
        u = User(username=name, password_hash=hash_password(pw), role=role)
        db.add(u)
    db.commit()
    audit(db, None, "create_user_cli", name, role)
print(f"User '{name}' saved with role '{role}'")
