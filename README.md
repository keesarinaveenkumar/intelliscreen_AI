# IntelliScreen AI

AI Resume Screening Agent Platform using Generative AI. Python 3.11, FastAPI, HTML5/CSS3/JavaScript, Anthropic Claude API, pdfplumber, python-docx, JWT + PBKDF2.

## Run it
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...                     # Windows: set ANTHROPIC_API_KEY=...
export SECRET_KEY=any-long-random-string                # keeps logins valid after restart
cd backend && uvicorn main:app --reload
```
Open http://127.0.0.1:8000. Without an API key the app runs in offline keyword mode.
Set `ANTHROPIC_MODEL` to change the Claude model (default `claude-sonnet-5-5`).

## Pipeline (matches the slides)
1. Ingest PDF, DOCX, TXT. 2. Parse into Experience, Education, Skills, Projects. 3. Claude turns the JD into must-have skills (JSON).
4. Chain-of-thought reasoning per resume. 5. Score = 10 x (0.40 T + 0.35 E + 0.15 Ed + 0.10 X). 6. Rank, filter by threshold (default 40), explain.
Rules: skip unsupported files, exclude resumes under 50 words, re-prompt once on invalid JSON, retry API calls up to 3 times.

## Files
- `backend/main.py` auth (JWT, PBKDF2, SQLite) and REST API, serves the UI
- `backend/screening.py` parsing, Claude reasoning, scoring, ranking
- `frontend/index.html` login, upload, ranked results, CSV/JSON export
