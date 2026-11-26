"""Structured Agent Tool Registry (11 Tools) with Pydantic Validation, JSON Schemas, Timeouts & Audit Logs."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from datetime import timedelta
from typing import Any, Optional
from pydantic import BaseModel, Field

from app.config import settings
from app.crm import CRMIntegrationManager
from app.database import DatabaseStore
from app.models import (
    AppointmentCreate,
    AppointmentReschedule,
    BANTProfile,
    LeadMemory,
    ToolExecutionLog,
    utc_now,
)
from app.nlp import (
    analyze_sentiment,
    classify_intent,
    compute_qualification,
    extract_memory_and_bant,
)
from app.rag import HybridRAGRetriever


# ---------------- Strict Pydantic Tool Input Schemas ----------------

class RetrieveMemoryArgs(BaseModel):
    session_id: str = Field(..., description="Active call session identifier")


class RetrieveKnowledgeArgs(BaseModel):
    query: str = Field(..., description="Buyer query to search in the Fortivex knowledge base")
    top_k: int = Field(default=3, ge=1, le=8, description="Number of knowledge chunks to retrieve")


class QualifyLeadArgs(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    utterance: str = Field(..., description="Latest buyer utterance containing BANT or profile details")
    has_booking: bool = Field(default=False, description="Whether a demo is already scheduled")


class AnalyzeSentimentArgs(BaseModel):
    utterance: str = Field(..., description="Buyer utterance to analyze for emotion, frustration, and buying intent")


class CheckAvailabilityArgs(BaseModel):
    count: int = Field(default=4, ge=1, le=10, description="Number of open slots to retrieve")
    target_date: Optional[str] = Field(default=None, description="Optional ISO date filter YYYY-MM-DD")
    timezone: str = Field(default="UTC", description="Customer IANA timezone")


class BookAppointmentArgs(BaseModel):
    session_id: str = Field(..., description="Active call session ID")
    lead_id: Optional[str] = Field(default=None, description="Persisted CRM lead ID")
    lead_name: str = Field(..., description="Prospect full name")
    lead_email: Optional[str] = Field(default=None, description="Prospect email address")
    company: Optional[str] = Field(default=None, description="Prospect company name")
    slot_iso: str = Field(..., description="ISO-8601 datetime slot to book")
    timezone: str = Field(default="UTC", description="Customer timezone")
    topic: str = Field(default="Fortivex Product Demo", description="Appointment topic")


class RescheduleAppointmentArgs(BaseModel):
    appt_id: str = Field(..., description="Existing appointment ID")
    slot_iso: str = Field(..., description="New ISO-8601 slot")
    timezone: str = Field(default="UTC", description="Customer timezone")


class CancelAppointmentArgs(BaseModel):
    appt_id: str = Field(..., description="Appointment ID to cancel")


class SyncCRMArgs(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    lead_id: str = Field(..., description="CRM lead ID to synchronize")
    event_type: str = Field(default="crm.lead.updated", description="CRM sync event type")


class GenerateFollowupArgs(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    channel: str = Field(default="email", description="Follow-up channel: email | call | calendar")
    notes: Optional[str] = Field(default=None, description="Optional context notes")


class GetCustomerHistoryArgs(BaseModel):
    email: Optional[str] = Field(default=None, description="Customer email address")
    phone: Optional[str] = Field(default=None, description="Customer phone number")
    name: Optional[str] = Field(default=None, description="Customer name")
    company: Optional[str] = Field(default=None, description="Customer company")


TOOL_SCHEMAS: dict[str, type[BaseModel]] = {
    "retrieve_customer_memory": RetrieveMemoryArgs,
    "retrieve_knowledge": RetrieveKnowledgeArgs,
    "qualify_lead": QualifyLeadArgs,
    "analyze_sentiment": AnalyzeSentimentArgs,
    "check_appointment_availability": CheckAvailabilityArgs,
    "check_availability": CheckAvailabilityArgs,  # Backward-compatible alias
    "book_appointment": BookAppointmentArgs,
    "reschedule_appointment": RescheduleAppointmentArgs,
    "cancel_appointment": CancelAppointmentArgs,
    "sync_crm": SyncCRMArgs,
    "generate_followup": GenerateFollowupArgs,
    "get_customer_history": GetCustomerHistoryArgs,
}

# Tools exposed to Gemini Function Calling (11 distinct canonical tools)
CANONICAL_TOOLS: list[tuple[str, str, type[BaseModel]]] = [
    ("retrieve_customer_memory", "Retrieve 3-tier short-term, session, and long-term customer memory.", RetrieveMemoryArgs),
    ("retrieve_knowledge", "Search Fortivex's hybrid RAG knowledge base for grounded pricing, security, product, or integration facts.", RetrieveKnowledgeArgs),
    ("qualify_lead", "Extract BANT fields and compute an explainable lead qualification score and tier.", QualifyLeadArgs),
    ("analyze_sentiment", "Analyze emotional polarity, frustration, objections, and purchase intent from buyer text.", AnalyzeSentimentArgs),
    ("check_appointment_availability", "Check conflict-free business-hour demo slots in the database.", CheckAvailabilityArgs),
    ("book_appointment", "Atomically book a confirmed demo appointment in the database after buyer intent and slot are clear.", BookAppointmentArgs),
    ("reschedule_appointment", "Atomically reschedule an existing demo appointment to a new valid slot.", RescheduleAppointmentArgs),
    ("cancel_appointment", "Cancel an existing scheduled appointment.", CancelAppointmentArgs),
    ("sync_crm", "Upsert and synchronize the lead record to internal SQLite and configured external CRM adapters.", SyncCRMArgs),
    ("generate_followup", "Generate a structured follow-up action plan and target date for the buyer.", GenerateFollowupArgs),
    ("get_customer_history", "Look up returning customer interactions, prior call summaries, and appointment history.", GetCustomerHistoryArgs),
]

ALLOWED_TOOLS = set(TOOL_SCHEMAS.keys())


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
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fortivex-tool")

    @staticmethod
    def json_schemas() -> dict[str, dict[str, Any]]:
        """Export strict JSON Schemas for all 11 canonical tools."""
        return {
            name: {
                "name": name,
                "description": desc,
                "parameters": model.model_json_schema(),
            }
            for name, desc, model in CANONICAL_TOOLS
        }

    def execute(
        self,
        session_id: str,
        tool_name: str,
        args: dict[str, Any],
        extra_context: Optional[dict[str, Any]] = None,
        timeout_seconds: Optional[float] = None,
    ) -> tuple[Any, ToolExecutionLog]:
        t0 = time.perf_counter()
        if tool_name not in ALLOWED_TOOLS:
            log = ToolExecutionLog(
                tool_name=tool_name,
                arguments=args,
                status="denied",
                summary=f"Tool '{tool_name}' is not in the permitted tool allowlist.",
                error="TOOL_NOT_ALLOWED",
                duration_ms=0,
            )
            self.db.log_tool_execution(session_id, log)
            raise PermissionError(log.summary)

        timeout = timeout_seconds if timeout_seconds is not None else settings.tool_timeout_seconds
        try:
            future = self._executor.submit(self._dispatch, tool_name, args, extra_context or {})
            result, summary_txt = future.result(timeout=timeout)
            preview = str(summary_txt)[:220]
            log = ToolExecutionLog(
                tool_name=tool_name,
                arguments=args,
                status="success",
                summary=summary_txt,
                result_preview=preview,
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            self.db.log_tool_execution(session_id, log)
            return result, log
        except FuturesTimeoutError:
            msg = f"Tool '{tool_name}' timed out after {timeout}s."
            log = ToolExecutionLog(
                tool_name=tool_name,
                arguments=args,
                status="error",
                summary=msg,
                error="TIMEOUT",
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            self.db.log_tool_execution(session_id, log)
            raise TimeoutError(msg)
        except Exception as exc:
            log = ToolExecutionLog(
                tool_name=tool_name,
                arguments=args,
                status="error",
                summary=f"{tool_name} failed: {exc}",
                error=str(exc),
                duration_ms=int((time.perf_counter() - t0) * 1000),
            )
            self.db.log_tool_execution(session_id, log)
            raise

    def _dispatch(
        self, tool_name: str, raw_args: dict[str, Any], ctx: dict[str, Any]
    ) -> tuple[Any, str]:
        schema_cls = TOOL_SCHEMAS[tool_name]
        validated = schema_cls(**raw_args)

        if tool_name == "retrieve_customer_memory":
            a: RetrieveMemoryArgs = validated  # type: ignore[assignment]
            memory: LeadMemory = ctx.get("memory") or self.db.get_or_create_session(a.session_id)["memory"]
            snap = self.db.get_three_tier_memory(a.session_id, memory)
            return (
                snap,
                f"Retrieved 3-tier memory (turns={len(snap.short_term_turns)}, returning={bool(snap.long_term_profile)})",
            )

        if tool_name == "retrieve_knowledge":
            a_rk: RetrieveKnowledgeArgs = validated  # type: ignore[assignment]
            chunks = self.rag.retrieve(a_rk.query, top_k=a_rk.top_k)
            top = f"{chunks[0].title} ({chunks[0].score:.2f})" if chunks else "no matching docs"
            return chunks, f"Hybrid RAG returned {len(chunks)} chunks (top={top})"

        if tool_name == "qualify_lead":
            a_ql: QualifyLeadArgs = validated  # type: ignore[assignment]
            sess = self.db.get_or_create_session(a_ql.session_id)
            mem: LeadMemory = ctx.get("memory") or sess["memory"]
            bant: BANTProfile = ctx.get("bant") or sess["qualification"].bant
            upd_mem, upd_bant = extract_memory_and_bant(a_ql.utterance, mem, bant)
            qual = compute_qualification(
                upd_bant, upd_mem, has_booking=a_ql.has_booking, last_utterance=a_ql.utterance
            )
            return (upd_mem, qual), f"Qualified lead: {qual.score}/100 ({qual.tier}, {qual.stage})"

        if tool_name == "analyze_sentiment":
            a_as: AnalyzeSentimentArgs = validated  # type: ignore[assignment]
            intent_cls = classify_intent(a_as.utterance)
            sent = analyze_sentiment(a_as.utterance, intent_cls=intent_cls)
            return sent, f"Sentiment={sent.label} ({sent.polarity}), Intent={sent.intent_score}/100"

        if tool_name in {"check_appointment_availability", "check_availability"}:
            a_ca: CheckAvailabilityArgs = validated  # type: ignore[assignment]
            slots = self.db.get_available_slots(
                count=a_ca.count, target_date=a_ca.target_date, tz_name=a_ca.timezone
            )
            return slots, f"Found {len(slots)} open conflict-free business-hour slots"

        if tool_name == "book_appointment":
            a_ba: BookAppointmentArgs = validated  # type: ignore[assignment]
            record = self.db.book_appointment(
                AppointmentCreate(
                    session_id=a_ba.session_id,
                    lead_id=a_ba.lead_id,
                    lead_name=a_ba.lead_name,
                    lead_email=a_ba.lead_email,
                    company=a_ba.company,
                    slot_iso=a_ba.slot_iso,
                    timezone=a_ba.timezone,
                    topic=a_ba.topic,
                )
            )
            hooks = self.crm.sync_appointment(record, "calendar.appointment.booked")
            return (record, hooks), f"Booked appointment {record.id} at {record.slot_iso}"

        if tool_name == "reschedule_appointment":
            a_ra: RescheduleAppointmentArgs = validated  # type: ignore[assignment]
            updated = self.db.reschedule_appointment(
                a_ra.appt_id,
                AppointmentReschedule(slot_iso=a_ra.slot_iso, timezone=a_ra.timezone),
            )
            if not updated:
                raise ValueError(f"Appointment '{a_ra.appt_id}' not found.")
            hooks = self.crm.sync_appointment(updated, "calendar.appointment.rescheduled")
            return (updated, hooks), f"Rescheduled {updated.id} to {updated.slot_iso}"

        if tool_name == "cancel_appointment":
            a_xc: CancelAppointmentArgs = validated  # type: ignore[assignment]
            ok = self.db.cancel_appointment(a_xc.appt_id)
            if not ok:
                raise ValueError(f"Appointment '{a_xc.appt_id}' not found.")
            return True, f"Cancelled appointment {a_xc.appt_id}"

        if tool_name == "sync_crm":
            a_sc: SyncCRMArgs = validated  # type: ignore[assignment]
            lead = self.db.get_lead(a_sc.lead_id)
            if not lead:
                raise ValueError(f"Lead '{a_sc.lead_id}' not found for CRM sync.")
            hooks, results = self.crm.sync_lead(a_sc.session_id, lead, a_sc.event_type)
            ext_states = ", ".join(f"{r.provider}:{r.external_sync_status}" for r in results) or "local-sqlite"
            return (hooks, results), f"CRM sync for {lead.id} ({ext_states})"

        if tool_name == "generate_followup":
            a_gf: GenerateFollowupArgs = validated  # type: ignore[assignment]
            sess = self.db.get_or_create_session(a_gf.session_id)
            qual = sess["qualification"]
            mem = sess["memory"]
            appt = self.db.get_session_appointment(a_gf.session_id)
            follow_date = appt.slot_iso[:10] if appt else (utc_now() + timedelta(days=2)).strftime("%Y-%m-%d")
            plan = {
                "channel": a_gf.channel,
                "follow_up_date": follow_date,
                "recipient": mem.email or mem.phone or mem.name or a_gf.session_id,
                "priority": qual.lead_priority,
                "recommended_actions": [
                    f"Share grounded summary addressing {', '.join(mem.objections) or 'evaluation goals'}",
                    f"Confirm next milestone for {qual.stage}",
                ],
                "notes": a_gf.notes or "",
            }
            return plan, f"Generated {a_gf.channel} follow-up plan for {follow_date}"

        if tool_name == "get_customer_history":
            a_gh: GetCustomerHistoryArgs = validated  # type: ignore[assignment]
            lead = self.db.find_returning_lead(a_gh.email, a_gh.phone, a_gh.name, a_gh.company)
            if not lead:
                return {"found": False, "previous_interactions": [], "appointments": []}, "No prior customer history found"
            appts = [a.model_dump() for a in self.db.list_appointments() if a.lead_id == lead.id]
            summaries = [s.model_dump() for s in self.db.list_call_summaries() if s.lead_id == lead.id]
            return {
                "found": True,
                "lead": lead.model_dump(),
                "previous_interactions": [s["executive_summary"] for s in summaries[:5]],
                "appointments": appts[:5],
            }, f"Loaded customer history for {lead.name} ({len(summaries)} prior summaries, {len(appts)} appointments)"

        raise ValueError(f"Unhandled tool: {tool_name}")
