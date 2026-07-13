"""Stateless /ask backend for the portfolio chatbot.

Serves the deploy-safe bundle only. The LLM key lives server-side (env / .env);
it is never exposed to the client. Public-endpoint guardrails per the brief:
per-IP rate limit, request-size cap, CORS locked to configured origins.

Run locally:
  uvicorn server.main:app --reload --port 8000

Env:
  PORTFOLIO_ALLOWED_ORIGINS   comma-separated origins (default "*")
  PORTFOLIO_RATE_LIMIT        requests per window per IP (default 20)
  PORTFOLIO_RATE_WINDOW_SEC   window seconds (default 300)
  PORTFOLIO_MAX_QUESTION_LEN  max chars in a question (default 600)
  PORTFOLIO_TURNSTILE_SECRET  Cloudflare Turnstile secret key (optional; when set,
                              /ask requires a valid turnstile_token -> bot protection)
  + one LLM key (ANTHROPIC_API_KEY / OPENAI_API_KEY / GROQ_API_KEY / XAI_API_KEY)
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
import urllib.request
from collections import defaultdict, deque

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from app.ask import answer, load_bundle
from ingest.llm import load_dotenv

load_dotenv()

_ORIGINS = [o.strip() for o in os.environ.get("PORTFOLIO_ALLOWED_ORIGINS", "*").split(",") if o.strip()]
_RATE_LIMIT = int(os.environ.get("PORTFOLIO_RATE_LIMIT", "20"))
_RATE_WINDOW = int(os.environ.get("PORTFOLIO_RATE_WINDOW_SEC", "300"))
_MAX_Q = int(os.environ.get("PORTFOLIO_MAX_QUESTION_LEN", "600"))
# Cloudflare Turnstile: when this secret is set, /ask requires a valid token.
# Unset -> verification is skipped (local dev works with no keys).
_TURNSTILE_SECRET = os.environ.get("PORTFOLIO_TURNSTILE_SECRET", "").strip()
_TURNSTILE_VERIFY_URL = "https://challenges.cloudflare.com/turnstile/v0/siteverify"

# Friendly, in-voice messages for transient provider problems (one per kind).
_BUSY_MESSAGES = {
    "rate_limit": "I'm getting a lot of questions right now, give me a sec and try again.",
    "unavailable": "Welp, my machinery behind the scenes just showed its true colors while I was putting your answer together. Give it another go in a moment?",
    "internal": "Aha, hit a snag while generating that one. I'm pinging the real Ali to look into it. Mind trying again in a bit?",
    "timeout": "Hmm, that took a little longer than I'd like. Mind giving it another shot?",
}


def _verify_turnstile(token: str, ip: str) -> bool:
    """Verify a Cloudflare Turnstile token server-side. Open when no secret is
    configured (dev). Fails CLOSED on a missing/invalid token, but OPEN on a
    verify-service network error so a Cloudflare hiccup can't take the bot down."""
    if not _TURNSTILE_SECRET:
        return True
    if not token:
        return False
    data = urllib.parse.urlencode(
        {"secret": _TURNSTILE_SECRET, "response": token, "remoteip": ip}).encode()
    try:
        req = urllib.request.Request(_TURNSTILE_VERIFY_URL, data=data)
        with urllib.request.urlopen(req, timeout=5) as r:
            return bool(json.loads(r.read().decode()).get("success"))
    except Exception:
        return True  # verify service unreachable -> don't block real visitors


def _busy_kind(exc: Exception) -> str | None:
    """Classify a TRANSIENT provider error into a friendly-message category, or
    None if it's a genuine (non-transient) error we shouldn't sugarcoat.
    Matches on class name + HTTP status so we don't import each provider SDK."""
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None) or getattr(exc, "status", None)
    if "ratelimit" in name or "overloaded" in name or status in (429, 529):
        return "rate_limit"
    if "timeout" in name or status in (408, 504):
        return "timeout"
    if "serviceunavailable" in name or "apiconnection" in name or status in (502, 503):
        return "unavailable"
    if "internalservererror" in name or status == 500:
        return "internal"
    return None

app = FastAPI(title="Portfolio Chatbot", version="1.0", docs_url="/docs")
app.add_middleware(
    CORSMiddleware, allow_origins=_ORIGINS or ["*"],
    allow_methods=["POST", "GET", "OPTIONS"], allow_headers=["*"],
)

# --- simple in-memory sliding-window rate limit (per IP) -------------------
_hits: dict[str, deque] = defaultdict(deque)


def _rate_limited(ip: str) -> bool:
    now = time.time()
    dq = _hits[ip]
    while dq and dq[0] < now - _RATE_WINDOW:
        dq.popleft()
    if len(dq) >= _RATE_LIMIT:
        return True
    dq.append(now)
    return False


class AskIn(BaseModel):
    question: str = Field(..., description="The visitor's question for the bot.")
    session_id: str | None = Field(None, description="Optional client id (not used server-side yet).")
    turnstile_token: str | None = Field(None, description="Cloudflare Turnstile token (required in prod when configured).")


class AskOut(BaseModel):
    answer: str
    projects_used: list[str]
    model: str


@app.get("/health")
def health() -> dict:
    b = load_bundle()
    return {"status": "ok", "projects": len(b["projects"]), "skills": len(b["skills"])}


@app.post("/ask", response_model=AskOut)
def ask(body: AskIn, request: Request):
    ip = (request.headers.get("x-forwarded-for", "").split(",")[0].strip()
          or (request.client.host if request.client else "unknown"))
    if _rate_limited(ip):
        return JSONResponse(status_code=429,
                            content={"detail": "Rate limit exceeded. Please slow down."})
    if not _verify_turnstile((body.turnstile_token or "").strip(), ip):
        return JSONResponse(
            status_code=403,
            content={"detail": "turnstile",
                     "message": "Quick bot check didn't pass. Refresh the page and try again."})

    q = (body.question or "").strip()
    if not q:
        return JSONResponse(status_code=400, content={"detail": "Question is required."})
    if len(q) > _MAX_Q:
        return JSONResponse(status_code=413,
                            content={"detail": f"Question too long (max {_MAX_Q} chars)."})
    try:
        return answer(q)
    except Exception as e:
        kind = _busy_kind(e)
        if kind:
            # Transient provider issue -> friendly, retryable response.
            return JSONResponse(
                status_code=503,
                headers={"Retry-After": "20"},
                content={"detail": kind, "message": _BUSY_MESSAGES[kind]},
            )
        return JSONResponse(status_code=502,
                            content={"detail": f"Upstream LLM error: {type(e).__name__}"})