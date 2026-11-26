"""Security Helpers: Session Cookie & Bearer Auth, Redis-Capable Rate Limiter, Upload Validation & PII Redaction."""

from __future__ import annotations

import hashlib
import hmac
import importlib.util
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Optional

from fastapi import Cookie, Header, HTTPException, Request, WebSocket

from app.config import get_runtime_secret, settings


def _cookie_signature(token: str) -> str:
    secret = token.encode("utf-8")
    return hmac.new(secret, b"fortivex-session-v2", hashlib.sha256).hexdigest()


class RateLimiter:
    """Thread-safe sliding-window rate limiter with optional Redis backend for multi-worker deployments."""

    def __init__(self, max_per_minute: int = 120, redis_url: str = "") -> None:
        self.max_per_minute = max_per_minute
        self._lock = threading.Lock()
        self.buckets: dict[str, deque[float]] = defaultdict(deque)
        self._redis = None
        url = (redis_url or settings.redis_url).strip()
        if url and importlib.util.find_spec("redis") is not None:
            try:
                import redis

                self._redis = redis.Redis.from_url(url, socket_timeout=1.0)
                self._redis.ping()
            except Exception:
                self._redis = None

    def check(self, key: str, limit_override: Optional[int] = None) -> bool:
        limit = limit_override if limit_override is not None else self.max_per_minute
        if limit <= 0:
            return True

        if self._redis is not None:
            try:
                r_key = f"fortivex:rl:{key}"
                now_ms = int(time.time() * 1000)
                cutoff = now_ms - 60_000
                pipe = self._redis.pipeline()
                pipe.zremrangebyscore(r_key, 0, cutoff)
                pipe.zcard(r_key)
                pipe.zadd(r_key, {f"{now_ms}-{time.monotonic_ns()}": now_ms})
                pipe.expire(r_key, 65)
                _, current_count, _, _ = pipe.execute()
                return int(current_count) < limit
            except Exception:
                pass

        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            q = self.buckets[key]
            while q and q[0] < cutoff:
                q.popleft()
            if len(q) >= limit:
                return False
            q.append(now)
            return True


rate_limiter = RateLimiter(max_per_minute=settings.rate_limit_rpm)


def enforce_rate_limit(request: Request) -> None:
    """FastAPI dependency rejecting clients exceeding configured RPM."""
    client_ip = request.client.host if request.client else "unknown"
    limit = settings.rate_limit_rpm
    if not rate_limiter.check(client_ip, limit_override=limit):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait before sending more requests.",
        )


def is_valid_token_or_cookie(
    authorization: Optional[str],
    x_api_key: Optional[str],
    session_cookie: Optional[str],
) -> bool:
    expected = get_runtime_secret("FORTIVEX_API_TOKEN", settings.fortivex_api_token)
    if not expected:
        return True

    if authorization and authorization.lower().startswith("bearer "):
        if hmac.compare_digest(authorization[7:].strip(), expected):
            return True
    if x_api_key and hmac.compare_digest(x_api_key.strip(), expected):
        return True
    if session_cookie and hmac.compare_digest(session_cookie.strip(), _cookie_signature(expected)):
        return True
    return False


def require_api_auth(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
    fortivex_session: Optional[str] = Cookie(default=None),
) -> None:
    """Require Bearer, X-API-Key, or signed HttpOnly session cookie when FORTIVEX_API_TOKEN is set."""
    if not is_valid_token_or_cookie(authorization, x_api_key, fortivex_session):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API token.")


def verify_ws_auth(websocket: WebSocket) -> bool:
    """Verify WebSocket connection token or signed session cookie when FORTIVEX_API_TOKEN is configured."""
    expected = get_runtime_secret("FORTIVEX_API_TOKEN", settings.fortivex_api_token)
    if not expected:
        return True
    auth = websocket.headers.get("authorization")
    x_key = websocket.headers.get("x-api-key") or websocket.query_params.get("token")
    cookie_val = websocket.cookies.get("fortivex_session")
    return is_valid_token_or_cookie(auth, x_key, cookie_val)


def validate_upload_file(filename: str, raw_bytes: bytes) -> None:
    """Enforce upload size limit and verify file extension + magic bytes."""
    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(raw_bytes) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum allowed size ({settings.max_upload_bytes // (1024 * 1024)} MB).",
        )
    ext = Path(filename).suffix.lower()
    if ext not in {".pdf", ".docx", ".txt", ".md"}:
        raise HTTPException(status_code=400, detail=f"Unsupported file format '{ext}'. Allowed: .pdf, .docx, .txt, .md")

    if ext == ".pdf" and not raw_bytes.lstrip().startswith(b"%PDF-"):
        raise HTTPException(status_code=400, detail="Invalid PDF file signature.")
    if ext == ".docx" and not raw_bytes.startswith(b"PK"):
        raise HTTPException(status_code=400, detail="Invalid DOCX archive signature.")


def mask_pii(text: str) -> str:
    """Redact emails and phone numbers for privacy-aware logging when enabled."""
    if not settings.pii_masking_enabled:
        return text
    masked = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[REDACTED_EMAIL]", text)
    masked = re.sub(r"(?<!\$)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\b\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b", "[REDACTED_PHONE]", masked)
    return masked
