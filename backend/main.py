"""IntelliScreen AI: FastAPI app (auth, REST APIs, business logic) that also serves the web UI."""
import hashlib, hmac, os, re, secrets, sqlite3, time
from pathlib import Path
import jwt
from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field
import screening

ROOT = Path(__file__).resolve().parent.parent
DB = os.getenv("DB_PATH", str(ROOT / "intelliscreen.db"))
SECRET = os.getenv("SECRET_KEY") or secrets.token_hex(32)  # set SECRET_KEY so logins survive restarts
MAX_BATCH = 500
app = FastAPI(title="IntelliScreen AI")
bearer = HTTPBearer(auto_error=False)


def db():
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS users (email TEXT PRIMARY KEY, salt BLOB, hash BLOB)")
    return con


def pbkdf2(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 200_000)


def make_token(email):
    return jwt.encode({"sub": email, "exp": int(time.time()) + 8 * 3600}, SECRET, algorithm="HS256")


def current_user(cred: HTTPAuthorizationCredentials = Depends(bearer)):
    try:
        return jwt.decode(cred.credentials, SECRET, algorithms=["HS256"])["sub"]
    except Exception:
        raise HTTPException(401, "Please log in again")


class Credentials(BaseModel):
    email: str = Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
    password: str = Field(min_length=8, max_length=128)


@app.post("/api/register")
def register(c: Credentials):
    salt = secrets.token_bytes(16)
    try:
        with db() as con:
            con.execute("INSERT INTO users VALUES (?,?,?)", (c.email.lower(), salt, pbkdf2(c.password, salt)))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "An account with this email already exists")
    return {"token": make_token(c.email.lower())}


@app.post("/api/login")
def login(c: Credentials):
    row = db().execute("SELECT salt, hash FROM users WHERE email=?", (c.email.lower(),)).fetchone()
    if not row or not hmac.compare_digest(row[1], pbkdf2(c.password, row[0])):
        raise HTTPException(401, "Email or password is incorrect")
    return {"token": make_token(c.email.lower())}


@app.get("/api/health")
def health():
    return {"mode": "claude" if os.getenv("ANTHROPIC_API_KEY") else "offline"}


@app.post("/api/screen")
def screen(jd_text: str = Form(""), threshold: int = Form(40), resumes: list[UploadFile] = File(...),
           jd_file: UploadFile | None = File(None), user: str = Depends(current_user)):
    if jd_file and jd_file.filename:
        jd_text = screening.extract_text(jd_file.filename, jd_file.file.read())
    if len(jd_text.split()) < 10:
        raise HTTPException(422, "Add a job description (at least 10 words)")
    if len(resumes) > MAX_BATCH:
        raise HTTPException(422, f"Upload at most {MAX_BATCH} resumes per batch")
    files = [(r.filename, r.file.read()) for r in resumes]
    return screening.screen(jd_text, files, max(0, min(100, threshold)))


@app.get("/")
def index():
    return FileResponse(ROOT / "frontend" / "index.html")
