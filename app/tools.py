"""Structured Agent Tool Registry with Schema Validation, Permissions & Audit Logging."""

from __future__ import annotations

import time
from typing import Any, Callable, Optional
from pydantic import BaseModel, Field

from app.crm import CRMIntegrationManager
from app.database import BookingConflictError, BookingValidationError, DatabaseStore
from app.models import (
    AppointmentCreate,
    AppointmentRecord,
    AppointmentReschedule,
    LeadMemory,
    LeadQualification,
    RetrievedChunk,
    ThreeTierMemorySnapshot,
    ToolExecutionLog,
)
from app.rag import HybridRAGRetriever


class RetrieveMemoryArgs(BaseModel):
    session_id: str


class RetrieveKnowledgeArgs(BaseModel):
    query: str
    top_k: int = Field(default=3, ge=1, le=8)


class CheckAvailabilityArgs(BaseModel):
    count: int = Field(default=3, ge=1, le=10)
    target_date: Optional[str] = None


class BookAppointmentArgs(BaseModel):
    session_id: str
    lead_id: Optional[str] = None
    lead_name: str
    lead_email: Optional[str] = None
    company: Optional[str] = None
    slot_iso: str
    timezone: str = "UTC"
    topic: str = "Fortivex Product Demo"


class RescheduleAppointmentArgs(BaseModel):
    appt_id: str
    slot_iso: str
    timezone: str = "UTC"


class CancelAppointmentArgs(BaseModel):
    appt_id: str


class SyncCRMArgs(BaseModel):
    session_id: str
    lead_id: str
    event_type: str


ALLOWED_TOOLS = {
    "retrieve_customer_memory",
    "retrieve_knowledge",
    "check_availability",
    "book_appointment",
    "reschedule_appointment",
    "cancel_appointment",
    "sync_crm",
}


class ToolRegistry:
    """Validated, audited tool execution layer isolating LLM reasoning from DB/CRM mutations."""

    def __init__(
        self,
        db: DatabaseStore,
        rag: HybridRAGRetriever,
        crm: CRMIntegrationManager,
    ) -> None:
        self.db = db
        self.rag = rag
        self.crm = crm

    def execute(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        extra_context: Optional[dict[str, Any]] = None,
    ) -> tuple[Any, ToolExecutionLog]:
        t0 = time.perf_counter()
        if tool_name not in ALLOWED_TOOLS:
            log = ToolExecutionLog(
                tool_name=tool_name,
                arguments=args,
                status="denied",
                summary=f"Tool '{tool_name}' is not in the permitted tool allowlist.",
                duration_ms=0,
            )
            self.db.log_tool_execution(session_id, log)
            raise PermissionError(log.summary)

        try:
            result, summary_txt = self._dispatch(tool_name, args, extra_context or {})
            log = ToolExecutionLog(
                tool_name=tool_name,
                arguments=args,
                status="success",
                summary=summary_txt,
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            self.db.log_tool_execution(session_id, log)
            return result, log
        except Exception as exc:
            log = ToolExecutionLog(
                tool_name=tool_name,
                arguments=args,
                status="error",
                summary=str(exc),
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            self.db.log_tool_execution(session_id, log)
            raise

    def _dispatch(
        self, tool_name: str, args: dict[str, Any], ctx: dict[str, Any]
    ) -> tuple[Any, str]:
        if tool_name == "retrieve_customer_memory":
            parsed = RetrieveMemoryArgs(**args)
            mem: LeadMemory = ctx["memory"]
            snap: ThreeTierMemorySnapshot = self.db.get_three_tier_memory(parsed.session_id, mem)
            return snap, f"Retrieved 3-tier memory (returning={mem.returning_lead}, prior_calls={len(snap.previous_interactions)})"

        if tool_name == "retrieve_knowledge":
            parsed = RetrieveKnowledgeArgs(**args)
            chunks: list[RetrievedChunk] = self.rag.retrieve(parsed.query, top_k=parsed.top_k)
            return chunks, f"Hybrid RAG returned {len(chunks)} chunks"

        if tool_name == "check_availability":
            parsed = CheckAvailabilityArgs(**args)
            slots = self.db.get_available_slots(count=parsed.count, target_date=parsed.target_date)
            return slots, f"Found {len(slots)} conflict-free slots"

        if tool_name == "book_appointment":
            parsed = BookAppointmentArgs(**args)
            appt: AppointmentRecord = self.db.book_appointment(
                AppointmentCreate(
                    session_id=parsed.session_id,
                    lead_id=parsed.lead_id,
                    lead_name=parsed.lead_name,
                    lead_email=parsed.lead_email,
                    company=parsed.company,
                    slot_iso=parsed.slot_iso,
                    timezone=parsed.timezone,
                    topic=parsed.topic,
                )
            )
            hooks = self.crm.sync_appointment(appt, "calendar.appointment.booked")
            return (appt, hooks), f"Confirmed appointment {appt.id} at {appt.slot_iso}"

        if tool_name == "reschedule_appointment":
            parsed = RescheduleAppointmentArgs(**args)
            updated = self.db.reschedule_appointment(
                parsed.appt_id, AppointmentReschedule(slot_iso=parsed.slot_iso, timezone=parsed.timezone)
            )
            if not updated:
                raise BookingValidationError(f"Appointment '{parsed.appt_id}' not found.")
            hooks = self.crm.sync_appointment(updated, "calendar.appointment.rescheduled")
            return (updated, hooks), f"Rescheduled appointment {updated.id} to {updated.slot_iso}"

        if tool_name == "cancel_appointment":
            parsed = CancelAppointmentArgs(**args)
            ok = self.db.cancel_appointment(parsed.appt_id)
            if not ok:
                raise BookingValidationError(f"Appointment '{parsed.appt_id}' not found.")
            return True, f"Cancelled appointment {parsed.appt_id}"

        if tool_name == "sync_crm":
            parsed = SyncCRMArgs(**args)
            lead = self.db.get_lead(parsed.lead_id)
            if not lead:
                raise ValueError(f"Lead '{parsed.lead_id}' not found.")
            hooks, _ = self.crm.sync_lead(parsed.session_id, lead, parsed.event_type)
            return hooks, f"Synced lead {lead.id} ({parsed.event_type}) via {self.crm.active_mode()}"

        raise PermissionError(f"Unknown tool: {tool_name}")
