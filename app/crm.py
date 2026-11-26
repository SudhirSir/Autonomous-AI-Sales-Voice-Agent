"""Modular CRM Integration Layer (SQLite, Webhook, HubSpot Upsert, Salesforce Upsert, Mock) with Idempotency & DLQ."""

from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Optional
from urllib.parse import quote

import httpx

from app.config import get_runtime_secret, settings
from app.database import DatabaseStore
from app.models import (
    AppointmentRecord,
    CRMSyncStatusRecord,
    CallSummary,
    LeadRecord,
    utc_now_iso,
)

logger = logging.getLogger(__name__)

TRANSIENT_HTTP_CODES = {408, 425, 429, 500, 502, 503, 504}
PERMANENT_HTTP_CODES = {400, 401, 403, 404, 409, 422}


def compute_idempotency_key(adapter: str, operation: str, entity_id: str, payload: dict[str, Any]) -> str:
    """Compute a deterministic idempotency key so retries never create duplicate writes."""
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    digest = hashlib.sha256(f"{adapter}:{operation}:{entity_id}:{canonical}".encode("utf-8")).hexdigest()[:24]
    return f"{adapter}-{operation}-{digest}"


def _request_with_retries(
    method: str,
    url: str,
    headers: dict[str, str],
    json_body: Optional[dict[str, Any]] = None,
    max_attempts: int = 3,
    timeout_sec: float = 6.0,
    base_backoff: float = 0.25,
) -> tuple[bool, Optional[int], int, Optional[dict[str, Any]], Optional[str], Optional[str]]:
    """Execute HTTP request with exponential backoff for transient errors only.

    Returns: (success, http_status, attempts, response_json, error_code, failure_reason)
    """
    last_err = ""
    last_code: Optional[int] = None
    last_error_code: Optional[str] = None

    for attempt in range(1, max_attempts + 1):
        try:
            with httpx.Client(timeout=timeout_sec) as client:
                if method.upper() == "GET":
                    resp = client.get(url, headers=headers)
                elif method.upper() == "PATCH":
                    resp = client.patch(url, headers=headers, json=json_body)
                else:
                    resp = client.post(url, headers=headers, json=json_body)

                last_code = resp.status_code
                if 200 <= resp.status_code < 300:
                    data = None
                    try:
                        if resp.text and resp.text.strip():
                            data = resp.json()
                    except Exception:
                        data = None
                    return True, resp.status_code, attempt, data, None, None

                last_error_code = f"HTTP_{resp.status_code}"
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                if resp.status_code in PERMANENT_HTTP_CODES:
                    # Do NOT retry permanent validation or auth failures
                    return False, resp.status_code, attempt, None, last_error_code, last_err
        except httpx.TimeoutException as exc:
            last_error_code = "TIMEOUT"
            last_err = f"Request timeout: {exc}"
        except Exception as exc:
            last_error_code = "NETWORK_ERROR"
            last_err = str(exc)

        if attempt < max_attempts:
            time.sleep(base_backoff * (2 ** (attempt - 1)))

    return False, last_code, max_attempts, None, last_error_code, last_err


class ExternalCRMAdapter(ABC):
    name: str = "base"

    @abstractmethod
    def is_configured(self) -> bool:
        ...

    @abstractmethod
    def dispatch(
        self,
        operation: str,
        entity_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
        existing_external_id: Optional[str] = None,
    ) -> CRMSyncStatusRecord:
        ...


class WebhookCRMAdapter(ExternalCRMAdapter):
    name = "webhook"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("CRM_WEBHOOK_URL", settings.crm_webhook_url))

    def dispatch(
        self,
        operation: str,
        entity_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
        existing_external_id: Optional[str] = None,
    ) -> CRMSyncStatusRecord:
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

        ok, code, attempts, data, err_code, err_reason = _request_with_retries("POST", url, headers, payload)
        ext_id = (data or {}).get("id") or existing_external_id
        return CRMSyncStatusRecord(
            provider=self.name,
            operation=operation,
            local_persistence_status="saved",
            external_sync_status="synced" if ok else "failed",
            external_record_id=str(ext_id) if ext_id else None,
            http_status=code,
            retry_count=max(0, attempts - 1),
            error_code=err_code,
            failure_reason=err_reason,
            last_sync_timestamp=utc_now_iso(),
            idempotency_key=idempotency_key,
        )


class HubSpotCRMAdapter(ExternalCRMAdapter):
    """HubSpot Contact Search -> Create or Update Upsert Adapter."""

    name = "hubspot"

    def is_configured(self) -> bool:
        return bool(get_runtime_secret("HUBSPOT_ACCESS_TOKEN", settings.hubspot_access_token))

    def dispatch(
        self,
        operation: str,
        entity_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
        existing_external_id: Optional[str] = None,
    ) -> CRMSyncStatusRecord:
        token = get_runtime_secret("HUBSPOT_ACCESS_TOKEN", settings.hubspot_access_token)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
            "Idempotency-Key": idempotency_key,
        }
        props = payload.get("properties", {})
        email = (props.get("email") or "").strip()
        full_name = (props.get("firstname_lastname") or "").strip()
        parts = full_name.split(" ") if full_name else []
        hs_props = {
            "firstname": parts[0] if parts else "Voice",
            "lastname": " ".join(parts[1:]) if len(parts) > 1 else "Prospect",
            "company": props.get("company") or "",
            "jobtitle": props.get("jobtitle") or "",
            "phone": props.get("phone") or "",
            "hs_lead_status": props.get("hs_lead_status") or "NEW",
        }
        if email:
            hs_props["email"] = email

        record_id = existing_external_id
        total_attempts = 0

        # Step 1: Search contact by email when no record ID cached yet
        if not record_id and email:
            search_body = {
                "filterGroups": [
                    {"filters": [{"propertyName": "email", "operator": "EQ", "value": email}]}
                ],
                "limit": 1,
            }
            s_ok, s_code, s_att, s_data, s_err_code, s_err = _request_with_retries(
                "POST",
                "https://api.hubapi.com/crm/v3/objects/contacts/search",
                headers,
                search_body,
            )
            total_attempts += s_att
            if not s_ok and s_code in PERMANENT_HTTP_CODES:
                return CRMSyncStatusRecord(
                    provider=self.name,
                    operation=operation,
                    local_persistence_status="saved",
                    external_sync_status="failed",
                    http_status=s_code,
                    retry_count=max(0, total_attempts - 1),
                    error_code=s_err_code,
                    failure_reason=s_err,
                    last_sync_timestamp=utc_now_iso(),
                    idempotency_key=idempotency_key,
                )
            if s_ok and s_data and s_data.get("results"):
                record_id = str(s_data["results"][0].get("id"))

        # Step 2: Update if existing record_id found, else Create
        if record_id:
            ok, code, att, data, err_code, err_reason = _request_with_retries(
                "PATCH",
                f"https://api.hubapi.com/crm/v3/objects/contacts/{record_id}",
                headers,
                {"properties": hs_props},
            )
        else:
            ok, code, att, data, err_code, err_reason = _request_with_retries(
                "POST",
                "https://api.hubapi.com/crm/v3/objects/contacts",
                headers,
                {"properties": hs_props},
            )
            if ok and data and data.get("id"):
                record_id = str(data["id"])

        total_attempts += att
        return CRMSyncStatusRecord(
            provider=self.name,
            operation=operation,
            local_persistence_status="saved",
            external_sync_status="synced" if ok else "failed",
            external_record_id=record_id,
            http_status=code,
            retry_count=max(0, total_attempts - 1),
            error_code=err_code,
            failure_reason=err_reason,
            last_sync_timestamp=utc_now_iso(),
            idempotency_key=idempotency_key,
        )


class SalesforceCRMAdapter(ExternalCRMAdapter):
    """Salesforce Lead SOQL Search -> Create or Update Upsert Adapter."""

    name = "salesforce"

    def is_configured(self) -> bool:
        return bool(
            get_runtime_secret("SALESFORCE_INSTANCE_URL", settings.salesforce_instance_url)
            and get_runtime_secret("SALESFORCE_ACCESS_TOKEN", settings.salesforce_access_token)
        )

    def dispatch(
        self,
        operation: str,
        entity_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
        existing_external_id: Optional[str] = None,
    ) -> CRMSyncStatusRecord:
        inst = get_runtime_secret("SALESFORCE_INSTANCE_URL", settings.salesforce_instance_url).rstrip("/")
        token = get_runtime_secret("SALESFORCE_ACCESS_TOKEN", settings.salesforce_access_token)
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }
        props = payload.get("properties", {})
        email = (props.get("email") or "").strip()
        sf_body = {
            "LastName": props.get("firstname_lastname") or "Voice Prospect",
            "Company": props.get("company") or "Unknown Company",
            "Email": email or None,
            "Phone": props.get("phone"),
            "Title": props.get("jobtitle"),
            "Status": props.get("lifecyclestage") or "Discovery",
        }

        record_id = existing_external_id
        total_attempts = 0

        if not record_id and email:
            safe_email = email.replace("'", "\\'")
            soql = quote(f"SELECT Id FROM Lead WHERE Email = '{safe_email}' LIMIT 1")
            s_ok, s_code, s_att, s_data, s_err_code, s_err = _request_with_retries(
                "GET",
                f"{inst}/services/data/v59.0/query?q={soql}",
                headers,
            )
            total_attempts += s_att
            if not s_ok and s_code in PERMANENT_HTTP_CODES:
                return CRMSyncStatusRecord(
                    provider=self.name,
                    operation=operation,
                    local_persistence_status="saved",
                    external_sync_status="failed",
                    http_status=s_code,
                    retry_count=max(0, total_attempts - 1),
                    error_code=s_err_code,
                    failure_reason=s_err,
                    last_sync_timestamp=utc_now_iso(),
                    idempotency_key=idempotency_key,
                )
            if s_ok and s_data and s_data.get("records"):
                record_id = str(s_data["records"][0].get("Id"))

        if record_id:
            ok, code, att, data, err_code, err_reason = _request_with_retries(
                "PATCH",
                f"{inst}/services/data/v59.0/sobjects/Lead/{record_id}",
                headers,
                sf_body,
            )
        else:
            ok, code, att, data, err_code, err_reason = _request_with_retries(
                "POST",
                f"{inst}/services/data/v59.0/sobjects/Lead",
                headers,
                sf_body,
            )
            if ok and data and data.get("id"):
                record_id = str(data["id"])

        total_attempts += att
        return CRMSyncStatusRecord(
            provider=self.name,
            operation=operation,
            local_persistence_status="saved",
            external_sync_status="synced" if ok else "failed",
            external_record_id=record_id,
            http_status=code,
            retry_count=max(0, total_attempts - 1),
            error_code=err_code,
            failure_reason=err_reason,
            last_sync_timestamp=utc_now_iso(),
            idempotency_key=idempotency_key,
        )


class MockCRMAdapter(ExternalCRMAdapter):
    name = "mock-crm"

    def is_configured(self) -> bool:
        return settings.crm_mock_mode

    def dispatch(
        self,
        operation: str,
        entity_id: str,
        payload: dict[str, Any],
        idempotency_key: str,
        existing_external_id: Optional[str] = None,
    ) -> CRMSyncStatusRecord:
        mock_id = existing_external_id or f"mock-{entity_id}"
        return CRMSyncStatusRecord(
            provider=self.name,
            operation=operation,
            local_persistence_status="saved",
            external_sync_status="mock",
            external_record_id=mock_id,
            http_status=200,
            retry_count=0,
            error_code=None,
            failure_reason="Simulated mock CRM sync (explicitly labelled mock; no external API called).",
            last_sync_timestamp=utc_now_iso(),
            idempotency_key=idempotency_key,
        )


class CRMIntegrationManager:
    """Coordinates internal SQLite CRM persistence + external CRM upserts, idempotency, retries & DLQ."""

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
            "failed_jobs": self.db.list_crm_sync_jobs(status="failed", limit=20),
            "all_jobs": self.db.list_crm_sync_jobs(limit=20),
        }

    def sync_lead(
        self, session_id: str, lead: LeadRecord, event_type: str
    ) -> tuple[list[dict[str, Any]], list[CRMSyncStatusRecord]]:
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
        return self._dispatch_all(event_type, lead.id, payload, lead=lead)

    def sync_appointment(
        self, appt: AppointmentRecord, event_type: str = "calendar.appointment.booked"
    ) -> list[dict[str, Any]]:
        webhooks, _ = self._dispatch_all(event_type, appt.id, appt.model_dump())
        return webhooks

    def sync_call_summary(self, summary: CallSummary) -> list[dict[str, Any]]:
        webhooks, _ = self._dispatch_all("crm.call_summary.created", summary.session_id, summary.model_dump())
        return webhooks

    def retry_failed_job(self, job_id: str) -> Optional[CRMSyncStatusRecord]:
        """Replay a failed/dead-letter CRM sync job using its original stable idempotency key."""
        job = self.db.get_crm_sync_job(job_id)
        if not job:
            return None
        adapter = next((a for a in self.adapters if a.name == job["adapter"] and a.is_configured()), None)
        if not adapter:
            return CRMSyncStatusRecord(
                provider=job["adapter"],
                operation=job["operation"],
                local_persistence_status="saved",
                external_sync_status="failed",
                error_code="ADAPTER_UNCONFIGURED",
                failure_reason=f"Adapter '{job['adapter']}' is not currently configured.",
                idempotency_key=job["idempotency_key"],
            )
        payload = json.loads(job["payload_json"])
        res = adapter.dispatch(
            operation=job["operation"],
            entity_id=job["entity_id"],
            payload=payload,
            idempotency_key=job["idempotency_key"],
            existing_external_id=job.get("external_record_id"),
        )
        self._persist_sync_result(job["entity_id"], payload, res)
        return res

    def _persist_sync_result(
        self, entity_id: str, payload: dict[str, Any], res: CRMSyncStatusRecord, lead: Optional[LeadRecord] = None
    ) -> dict[str, Any]:
        ok = res.external_sync_status in {"synced", "mock"}
        if lead and res.external_record_id and res.external_sync_status == "synced":
            self.db.set_lead_external_id(lead.id, res.provider, res.external_record_id)

        self.db.record_crm_sync_log(
            adapter=res.provider,
            operation=res.operation,
            idempotency_key=res.idempotency_key,
            status="success" if ok else "failed",
            http_status=res.http_status,
            attempts=res.retry_count + 1,
            detail=res.failure_reason or f"HTTP {res.http_status or 200} OK",
            external_record_id=res.external_record_id,
            error_code=res.error_code,
        )
        self.db.upsert_crm_sync_job(
            adapter=res.provider,
            operation=res.operation,
            entity_id=entity_id,
            idempotency_key=res.idempotency_key,
            payload=payload,
            status="synced" if res.external_sync_status == "synced" else ("mock" if res.external_sync_status == "mock" else "failed"),
            retry_count=res.retry_count,
            http_status=res.http_status,
            external_record_id=res.external_record_id,
            error_code=res.error_code,
            failure_reason=res.failure_reason,
        )
        return self.db.record_webhook(
            event_type=res.operation,
            target=f"{res.provider} ({res.external_sync_status})",
            payload=payload,
            delivered=ok,
        )

    def _dispatch_all(
        self,
        operation: str,
        entity_id: str,
        payload: dict[str, Any],
        lead: Optional[LeadRecord] = None,
    ) -> tuple[list[dict[str, Any]], list[CRMSyncStatusRecord]]:
        webhooks: list[dict[str, Any]] = []
        results: list[CRMSyncStatusRecord] = []
        active_external = [a for a in self.adapters if a.is_configured()]

        if not active_external:
            results.append(
                CRMSyncStatusRecord(
                    provider="local-sqlite",
                    operation=operation,
                    local_persistence_status="saved",
                    external_sync_status="not_configured",
                    last_sync_timestamp=utc_now_iso(),
                    idempotency_key=compute_idempotency_key("local-sqlite", operation, entity_id, payload),
                )
            )
            evt = self.db.record_webhook(
                event_type=operation,
                target="local-sqlite (external CRM not configured)",
                payload=payload,
                delivered=True,
            )
            webhooks.append(evt)
            return webhooks, results

        for adapter in active_external:
            idem_key = compute_idempotency_key(adapter.name, operation, entity_id, payload)
            existing_success = self.db.get_successful_crm_sync(idem_key)
            if existing_success:
                results.append(
                    CRMSyncStatusRecord(
                        provider=adapter.name,
                        operation=operation,
                        local_persistence_status="saved",
                        external_sync_status="mock" if adapter.name == "mock-crm" else "synced",
                        external_record_id=existing_success.get("external_record_id"),
                        http_status=existing_success.get("http_status") or 200,
                        retry_count=0,
                        failure_reason="Deduplicated: identical idempotency key already synced.",
                        last_sync_timestamp=existing_success.get("created_at") or utc_now_iso(),
                        idempotency_key=idem_key,
                    )
                )
                continue

            existing_ext_id = lead.external_crm_ids.get(adapter.name) if lead else None
            res = adapter.dispatch(
                operation=operation,
                entity_id=entity_id,
                payload=payload,
                idempotency_key=idem_key,
                existing_external_id=existing_ext_id,
            )
            results.append(res)
            evt = self._persist_sync_result(entity_id, payload, res, lead=lead)
            webhooks.append(evt)

        return webhooks, results
