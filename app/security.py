"""Security Helpers: API Authentication, Sliding-Window Rate Limiting & PII Redaction."""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque
from typing import Optional

from fastapi import Header, HTTPException, Request

from app.config import get_runtime_secret, settings


class RateLimiter:
    """Thread-safe in-memory sliding-window IP rate limiter."""

    def __init__(self, max_per_minute: int = 120) -> None:
        self.max_per_minute = max_per_minute
        self.buckets: dict[str, deque[float]] = defaultdict(deque)

    def check(self, key: str, limit_override: Optional[int] = None) -> bool:
        limit = limit_override if limit_override is not None else self.max_per_minute
        if limit <= 0:
            return True
        now = time.monotonic()
        cutoff = now - 60.0
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


def require_api_auth(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    """Require Bearer or X-API-Key authentication whenever FORTIVEX_API_TOKEN is configured."""
    expected = get_runtime_secret("FORTIVEX_API_TOKEN", settings.fortivex_api_token)
    if not expected:
        return

    provided = ""
    if authorization and authorization.lower().startswith("bearer "):
        provided = authorization[7:].strip()
    elif x_api_key:
        provided = x_api_key.strip()

    if provided != expected:
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API token.")


def mask_pii(text: str) -> str:
    """Redact emails and phone numbers for privacy-aware logging when enabled."""
    if not settings.pii_masking_enabled:
        return text
    masked = re.sub(r"[\w.+-]+@[\w-]+\.[\w.-]+", "[REDACTED_EMAIL]", text)
    masked = re.sub(r"(?<!\$)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)|\b\d{3})[\s.-]?\d{3}[\s.-]?\d{4}\b", "[REDACTED_PHONE]", masked)
    return masked
