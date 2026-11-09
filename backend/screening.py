"""IntelliScreen AI pipeline: ingest -> parse -> structure JD -> reason -> score -> rank & explain."""
import io, json, os, re, time
import docx, pdfplumber

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
WEIGHTS = {"technical": 0.40, "experience": 0.35, "education": 0.15, "extras": 0.10}
ALLOWED = {".pdf", ".docx", ".txt"}
ALIAS = {"work experience": "experience", "employment": "experience", "academics": "education",
         "technical skills": "skills", "certification": "extras", "certifications": "extras"}
HEADING = re.compile(r"^\s*(work experience|experience|employment|education|academics|technical skills|"
                     r"skills|projects|certifications?)\s*:?\s*$", re.I)
VOCAB = ["python", "java", "javascript", "typescript", "react", "fastapi", "django", "flask", "sql",
         "postgresql", "mysql", "mongodb", "pandas", "numpy", "spark", "airflow", "kafka", "aws", "azure",
         "gcp", "docker", "kubernetes", "git", "rest", "machine learning", "nlp", "llm", "pytorch",
         "tensorflow", "etl", "data pipelines", "tableau", "power bi", "html", "css", "linux"]


def client():
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        return None  # offline mode: keyword rubric instead of Claude reasoning
    from anthropic import Anthropic
    return Anthropic(api_key=key)


def extract_text(name, data):
    ext = os.path.splitext(name.lower())[1]
    if ext == ".pdf":
        with pdfplumber.open(io.BytesIO(data)) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)
    if ext == ".docx":
        return "\n".join(p.text for p in docx.Document(io.BytesIO(data)).paragraphs)
    return data.decode("utf-8", "ignore")


def split_sections(text):
    sections, cur = {"summary": []}, "summary"
    for line in text.splitlines():
        m = HEADING.match(line)
        if m:
            key = m.group(1).lower()
            cur = ALIAS.get(key, key)
            sections.setdefault(cur, [])
        else:
            sections[cur].append(line)
    return {k: "\n".join(v).strip() for k, v in sections.items()}


def ask_json(c, system, prompt):
    """Retry API calls up to 3 times; re-prompt once if the reply is not valid JSON."""
    for _ in range(2):
        for i in range(3):
            try:
                r = c.messages.create(model=MODEL, max_tokens=1500, system=system,
                                      messages=[{"role": "user", "content": prompt}])
                break
            except Exception:
                if i == 2:
                    raise
                time.sleep(2 ** i)
        try:
            return json.loads(re.search(r"\{.*\}", r.content[0].text, re.S).group(0))
        except Exception:
            prompt += "\n\nYour last reply was not valid JSON. Reply with one JSON object only."
    raise ValueError("Model did not return valid JSON")


def parse_jd(c, jd):
    if c:
        return ask_json(c, "You convert job descriptions to JSON.",
                        'Return {"title": str, "must_have": [skills], "nice_to_have": [skills], '
                        '"min_years": int} for this job description:\n' + jd)
    low = jd.lower()
    years = re.search(r"(\d+)\+?\s*years", low)
    return {"title": jd.strip().splitlines()[0][:80] if jd.strip() else "Role",
            "must_have": [s for s in VOCAB if s in low], "nice_to_have": [],
            "min_years": int(years.group(1)) if years else 2}


def reason_and_score(c, resume, jd):
    """Chain-of-thought evaluation. Returns 0-10 scores per category plus the explanation."""
    if c:
        out = ask_json(
            c, "You are a careful recruiter. Use only facts present in the resume and job description. "
               "Think step by step about transferable skills, then justify every score.",
            f"JOB (JSON): {json.dumps(jd)}\n\nRESUME SECTIONS: {json.dumps(resume['sections'])}\n\n"
            'Return {"reasoning": str, "scores": {"technical": 0-10, "experience": 0-10, '
            '"education": 0-10, "extras": 0-10}, "strengths": [str], "gaps": [str], "recommendation": str}')
        return out
    low = resume["text"].lower()
    need = jd["must_have"]
    hit = [s for s in need if s in low]
    years = max([int(y) for y in re.findall(r"(\d+)\+?\s*years", low)] or [1])
    edu = 10 if re.search(r"m\.?tech|master|ph\.?d|mba", low) else 8 if re.search(r"b\.?tech|b\.?e\b|bachelor|b\.?sc", low) else 4
    extras = min(10, 3 * len(re.findall(r"certif|award|publication|open.?source|hackathon|github", low)))
    tech = 10 * len(hit) / len(need) if need else 5
    exp = min(10, 10 * years / max(jd["min_years"], 1))
    miss = [s for s in need if s not in low]
    score = 10 * (WEIGHTS["technical"] * tech + WEIGHTS["experience"] * exp + WEIGHTS["education"] * edu + WEIGHTS["extras"] * extras)
    return {"reasoning": "Offline keyword rubric (no Claude API key set).",
            "scores": {"technical": tech, "experience": exp, "education": edu, "extras": extras},
            "strengths": [f"Matches {len(hit)} of {len(need)} required skills: {', '.join(hit[:6])}"] if hit else ["No required skills found"],
            "gaps": [f"Missing: {', '.join(miss[:6])}"] if miss else ["No major gaps found"],
            "recommendation": "Interview" if score >= 60 else "Consider" if score >= 40 else "Reject"}


def score_total(scores):
    """Score = 10 x (0.40 T + 0.35 E + 0.15 Ed + 0.10 X), normalised to 0-100."""
    return round(10 * sum(WEIGHTS[k] * max(0, min(10, float(scores[k]))) for k in WEIGHTS), 1)


def screen(jd_text, files, threshold=40):
    c = client()
    jd = parse_jd(c, jd_text)
    results, skipped = [], []
    for name, data in files:
        if os.path.splitext(name.lower())[1] not in ALLOWED:
            skipped.append({"name": name, "reason": "Unsupported file type (use PDF, DOCX or TXT)"})
            continue
        try:
            text = extract_text(name, data)
        except Exception:
            skipped.append({"name": name, "reason": "File could not be read"})
            continue
        if len(text.split()) < 50:
            skipped.append({"name": name, "reason": "Under 50 words (scanned or empty; OCR is not supported)"})
            continue
        resume = {"name": name, "text": text, "sections": split_sections(text)}
        out = reason_and_score(c, resume, jd)
        total = score_total(out["scores"])
        results.append({"name": name, "score": total, "categories": {k: round(float(v), 1) for k, v in out["scores"].items()},
                        "strengths": out["strengths"], "gaps": out["gaps"], "recommendation": out["recommendation"],
                        "reasoning": out["reasoning"], "status": "Shortlisted" if total >= threshold else "Below threshold"})
    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results, 1):
        r["rank"] = i
    return {"job": jd, "mode": "claude" if c else "offline", "threshold": threshold,
            "results": results, "skipped": skipped}
