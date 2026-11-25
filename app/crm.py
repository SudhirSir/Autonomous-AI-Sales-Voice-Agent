"""Modular CRM Integration Layer (SQLite, Webhook, HubSpot, Salesforce, Mock) with Retries."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import httpx

from app.config import get_runtime_secret, settings
from app.database import DatabaseStore
from app.models import AppointmentRecord, CallSummary, LeadRecord

logger = logging.getLogger(__name__)


@dataclass
class CRMSyncResult:
    adapter: str
    operation: str
    success: bool
    mode: str  # "local-sqlite" | "live-external" | "mock"
    status_code: Optional[int]
    attempts: int
    detail: str
    idempotency_key: str


class ExternalCRMAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        ...

    @abstractmethod
    def dispatch(
        self, operation: str, payload: dict[str, Any], idempotency_key: str
    ) -> CRMSyncResult:
        ...


def _post_with_retries(
    url: str,
    headers: dict[str, str],
    payload: dict[str, Any],
    max_attempts: int = 3,
    timeout_sec: float = 6.0,
    base_backoff: float = 0.25,
) -> tuple[bool, Optional[int], int, str]:
    """Send HTTP POST with exponential backoff retries and timeout handling."""
    last_err = ""
    last_code: Optional[int] = None
    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=timeout_sec) as client:
                resp = client.post(url, headers=headers, json=payload)
                last_code = resp.status_code
                if 200 <= resp.status_code < 300:
                    return True, resp.status_code, attempt, f"HTTP {resp.status_code} OK"
                last_err = f"HTTP {resp.status_code}: {resp.text[:160]}"
        except Exception as exc:
            last_err = str(exc)

        if attempt < max_attempts:
            time.sleep(base_backoff * (2 ** (attempt - 1)))

    return False, last_code, max_attempts, last_err


class WebhookCRMAdapter(ExternalCRMAdapter):
    name = "webhook"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("CRM_WEBHOOK_URL", settings.crm_webhook_url))

    def dispatch(
        self, operation: str, payload: dict[str, Any], idempotency_key: str
    ) -> CRMSyncResult:
        url = get_runtime_secret("CRM_WEBHOOK_URL", settings.crm_webhook_url)
        secret = get_runtime_secret("CRM_WEBHOOK_SECRET", settings.crm_webhook_secret)
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
            "X-Fortivex-Event": operation,
        }
        if secret:
            sig = hmac.new(secret.encode(), json.dumps(payload, sort_keys=True).encode(), hashlib.sha256).hexdigest()
            headers["X-Fortivex-Signature"] = f"sha256={sig}"

        ok, code, attempts, detail = _post_with_retries(url, headers, payload)
        return CRMSyncResult(
            adapter=self.name,
            operation=operation,
            success=ok,
            mode="live-external",
            status_code=code,
            attempts=attempts,
            detail=detail,
            idempotency_key=idempotency_key,
        )


class HubSpotCRMAdapter(ExternalCRMAdapter):
    name = "hubspot"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("HUBSPOT_ACCESS_TOKEN", settings.hubspot_access_token))

    def dispatch(
        self, operation: str, payload: dict[str, Any], idempotency_key: str
    ) -> CRMSyncResult:
        token = get_runtime_secret("HUBSPOT_ACCESS_TOKEN", settings.hubspot_access_token)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
        }
        props = payload.get("properties", {})
        hubspot_body = {
            "properties": {
                "email": props.get("email") or "",
                "firstname": (props.get("firstname_lastname") or "").split(" ")[0],
                "lastname": " ".join((props.get("firstname_lastname") or "").split(" ")[1:]) or "Prospect",
                "company": props.get("company") or "",
                "jobtitle": props.get("jobtitle") or "",
                "phone": props.get("phone") or "",
                "hs_lead_status": props.get("hs_lead_status") or "NEW",
            }
        }
        ok, code, attempts, detail = _post_with_retries(
            "https://api.hubapi.com/crm/v3/objects/contacts",
            headers,
            hubspot_body,
        )
        return CRMSyncResult(
            adapter=self.name,
            operation=operation,
            success=ok,
            mode="live-external",
            status_code=code,
            attempts=attempts,
            detail=detail,
            idempotency_key=idempotency_key,
        )


class SalesforceCRMAdapter(ExternalCRMAdapter):
    name = "salesforce"

    def is_configured(self) -> bool:
        return bool(
            get_runtime_secret("SALESFORCE_INSTANCE_URL", settings.salesforce_instance_url)
            and get_runtime_secret("SALESFORCE_ACCESS_TOKEN", settings.salesforce_access_token)
        )

    def dispatch(
        self, operation: str, payload: dict[str, Any], idempotency_key: str
    ) -> CRMSyncResult:
        inst = get_runtime_secret("SALESFORCE_INSTANCE_URL", settings.salesforce_instance_url).rstrip("/")
        token = get_runtime_secret("SALESFORCE_ACCESS_TOKEN", settings.salesforce_access_token)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        props = payload.get("properties", {})
        sf_body = {
            "LastName": props.get("firstname_lastname") or "Voice Prospect",
            "Company": props.get("company") or "Unknown Company",
            "Email": props.get("email"),
            "Phone": props.get("phone"),
            "Title": props.get("jobtitle"),
            "Status": props.get("lifecyclestage") or "Discovery",
        }
        ok, code, attempts, detail = _post_with_retries(
            f"{inst}/services/data/v59.0/sobjects/Lead",
            headers,
            sf_body,
        )
        return CRMSyncResult(
            adapter=self.name,
            operation=operation,
            success=ok,
            mode="live-external",
            status_code=code,
            attempts=attempts,
            detail=detail,
            idempotency_key=idempotency_key,
        )


class MockCRMAdapter(ExternalCRMAdapter):
    name = "mock-crm"

    def is_configured(self) -> bool:
        return settings.crm_mock_mode

    def dispatch(
        self, operation: str, payload: dict[str, Any], idempotency_key: str
    ) -> CRMSyncResult:
        return CRMSyncResult(
            adapter=self.name,
            operation=operation,
            success=True,
            mode="mock",
            status_code=200,
            attempts=1,
            detail="Simulated mock CRM sync (not sent to an external CRM endpoint).",
            idempotency_key=idempotency_key,
        )


class CRMIntegrationManager:
    """Coordinates internal SQLite CRM persistence + optional external CRM adapters."""

    def __init__(self, db: DatabaseStore) -> None:
        self.db = db
        self.adapters: list[ExternalCRMAdapter] = [
            WebhookCRMAdapter(),
            HubSpotCRMAdapter(),
            SalesforceCRMAdapter(),
            MockCRMAdapter(),
        ]

    def active_mode(self) -> str:
        live = [a.name for a in self.adapters if a.name != "mock-crm" and a.is_configured()]
        if live:
            return f"local-sqlite + {','.join(live)}"
        if settings.crm_mock_mode:
            return "local-sqlite + mock-crm"
        return "local-sqlite"

    def get_status(self) -> dict[str, Any]:
        return {
            "mode": self.active_mode(),
            "adapters": [
                {
                    "name": a.name,
                    "configured": a.is_configured(),
                    "kind": "mock" if a.name == "mock-crm" else "live-external",
                }
                for a in self.adapters
            ],
            "recent_sync_logs": self.db.list_crm_sync_logs(limit=15),
        }

    def sync_lead(
        self, session_id: str, lead: LeadRecord, event_type: str
    ) -> tuple[list[dict[str, Any]], list[CRMSyncResult]]:
        payload = {
            "event": event_type,
            "session_id": session_id,
            "lead_id": lead.id,
            "properties": {
                "firstname_lastname": lead.name,
                "email": lead.email,
                "phone": lead.phone,
                "company": lead.company,
                "jobtitle": lead.role,
                "hs_lead_status": lead.tier,
                "lifecyclestage": lead.stage,
                "fortivex_bant_score": lead.score,
                "amount": lead.estimated_acv_usd,
                "preferred_language": lead.preferred_language,
                "pain_points": lead.pain_points,
                "objections": lead.objections,
            },
        }
        return self._dispatch_all(event_type, payload)

    def sync_appointment(self, appt: AppointmentRecord, event_type: str = "calendar.appointment.booked") -> list[dict[str, Any]]:
        webhooks, _ = self._dispatch_all(event_type, appt.model_dump())
        return webhooks

    def sync_call_summary(self, summary: CallSummary) -> list[dict[str, Any]]:
        webhooks, _ = self._dispatch_all("crm.call_summary.created", summary.model_dump())
        return webhooks

    def _dispatch_all(
        self, operation: str, payload: dict[str, Any]
    ) -> tuple[list[dict[str, Any]], list[CRMSyncResult]]:
        webhooks: list[dict[str, Any]] = []
        results: list[CRMSyncResult] = []
        active_external = [a for a in self.adapters if a.is_configured()]

        if not active_external:
            # Record internal SQLite CRM audit event honestly as internal persistence
            evt = self.db.record_webhook(
                event_type=operation,
                target="Internal SQLite CRM (no external CRM configured)",
                payload=payload,
                delivered=True,
            )
            webhooks.append(evt)
            return webhooks, results

        for adapter in active_external:
            idem_key = f"{adapter.name}-{operation}-{uuid.uuid4().hex[:10]}"
            res = adapter.dispatch(operation, payload, idem_key)
            results.append(res)
            self.db.record_crm_sync_log(
                adapter=res.adapter,
                operation=operation,
                idempotency_key=res.idempotency_key,
                status="success" if res.success else "failed",
                http_status=res.status_code,
                attempts=res.attempts,
                detail=res.detail,
            )
            evt = self.db.record_webhook(
                event_type=operation,
                target=f"{res.adapter} ({res.mode})",
                payload=payload,
                delivered=res.success,
            )
            webhooks.append(evt)
        return webhooks, results
