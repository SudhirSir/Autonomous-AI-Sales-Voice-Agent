"""SQLite persistence, schema migrations, atomic booking, 3-tier memory, and analytics."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.models import (
    AppointmentCreate,
    AppointmentRecord,
    AppointmentReschedule,
    BANTProfile,
    BusinessInsights,
    CallSummary,
    ConversationMessage,
    LeadMemory,
    LeadQualification,
    LeadRecord,
    SentimentResult,
    SessionMetrics,
    SupportedLang,
    ThreeTierMemorySnapshot,
    ToolExecutionLog,
    utc_now,
    utc_now_iso,
)
from app.nlp import resolve_zoneinfo, validate_business_slot


class BookingConflictError(Exception):
    """Raised when an appointment slot is already booked."""


class BookingValidationError(Exception):
    """Raised when an appointment date/time violates business hours or is in the past."""


class DatabaseStore:
    """Thread-safe SQLite store with schema migrations and atomic transactions."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or settings.fortivex_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._run_migrations()
        self._seed_demo_data_if_empty()

    def _connect(self, isolation_level: str | None = "") -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10.0,
            isolation_level=isolation_level,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA foreign_keys=ON;")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version INTEGER PRIMARY KEY,
                    applied_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS leads (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
                    phone TEXT,
                    company TEXT,
                    role TEXT,
                    preferred_language TEXT DEFAULT 'en',
                    score INTEGER DEFAULT 0,
                    tier TEXT DEFAULT 'Cold',
                    stage TEXT DEFAULT 'Discovery',
                    estimated_acv_usd INTEGER DEFAULT 0,
                    bant_json TEXT NOT NULL,
                    pain_points_json TEXT NOT NULL,
                    objections_json TEXT NOT NULL,
                    last_sentiment REAL DEFAULT 0.0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    lead_id TEXT,
                    language TEXT DEFAULT 'en',
                    memory_json TEXT NOT NULL,
                    qualification_json TEXT NOT NULL,
                    awaiting_slot_choice INTEGER DEFAULT 0,
                    proposed_slots_json TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    language TEXT DEFAULT 'en',
                    sentiment_json TEXT,
                    retrieved_chunks_json TEXT DEFAULT '[]',
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS appointments (
                    id TEXT PRIMARY KEY,
                    lead_id TEXT,
                    session_id TEXT,
                    lead_name TEXT NOT NULL,
                    lead_email TEXT,
                    company TEXT,
                    slot_iso TEXT NOT NULL,
                    duration_minutes INTEGER DEFAULT 30,
                    topic TEXT NOT NULL,
                    timezone TEXT DEFAULT 'UTC',
                    status TEXT DEFAULT 'Confirmed',
                    meeting_link TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS score_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    lead_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    score INTEGER NOT NULL,
                    tier TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    sentiment_polarity REAL DEFAULT 0.0,
                    intent_score INTEGER DEFAULT 0,
                    recorded_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS tool_execution_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    arguments_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    duration_ms INTEGER DEFAULT 0,
                    executed_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS webhook_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    delivered INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS crm_sync_logs (
                    id TEXT PRIMARY KEY,
                    adapter TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    http_status INTEGER,
                    attempts INTEGER DEFAULT 1,
                    external_record_id TEXT,
                    error_code TEXT,
                    detail TEXT,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS crm_sync_jobs (
                    id TEXT PRIMARY KEY,
                    adapter TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    entity_id TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    payload_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    retry_count INTEGER DEFAULT 0,
                    http_status INTEGER,
                    external_record_id TEXT,
                    error_code TEXT,
                    failure_reason TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS call_summaries (
                    session_id TEXT PRIMARY KEY,
                    summary_json TEXT NOT NULL,
                    generated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS custom_docs (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )

    def _run_migrations(self) -> None:
        """Preserve existing DB records while upgrading legacy schemas safely."""
        with self._connect() as conn:
            lead_cols = {r["name"] for r in conn.execute("PRAGMA table_info(leads)").fetchall()}
            if "phone" not in lead_cols:
                conn.execute("ALTER TABLE leads ADD COLUMN phone TEXT")
            if "external_crm_ids_json" not in lead_cols:
                conn.execute("ALTER TABLE leads ADD COLUMN external_crm_ids_json TEXT DEFAULT '{}'")

            sess_cols = {r["name"] for r in conn.execute("PRAGMA table_info(sessions)").fetchall()}
            if "metrics_json" not in sess_cols:
                conn.execute("ALTER TABLE sessions ADD COLUMN metrics_json TEXT")

            sync_cols = {r["name"] for r in conn.execute("PRAGMA table_info(crm_sync_logs)").fetchall()}
            if "external_record_id" not in sync_cols:
                conn.execute("ALTER TABLE crm_sync_logs ADD COLUMN external_record_id TEXT")
            if "error_code" not in sync_cols:
                conn.execute("ALTER TABLE crm_sync_logs ADD COLUMN error_code TEXT")

            applied = {r["version"] for r in conn.execute("SELECT version FROM schema_migrations").fetchall()}
            if 1 not in applied:
                conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (1, utc_now_iso()),
                )
            if 2 not in applied:
                conn.execute(
                    "INSERT OR IGNORE INTO schema_migrations (version, applied_at) VALUES (?, ?)",
                    (2, utc_now_iso()),
                )
            conn.commit()

    def _seed_demo_data_if_empty(self) -> None:
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
            if count > 0:
                return
            now = utc_now()
            tomorrow = now + timedelta(days=1)
            if tomorrow.weekday() >= 5:
                tomorrow += timedelta(days=(7 - tomorrow.weekday()))
            tomorrow_15 = tomorrow.replace(hour=15, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")

            lead1_id = "lead-demo-001"
            lead2_id = "lead-demo-002"
            bant1 = BANTProfile(
                budget="~$42,000", budget_score=25,
                authority="VP of Sales (Executive Decision Maker)", authority_score=25,
                need="Manual CRM data entry overhead; Multilingual buyer coverage gap", need_score=22,
                timeline="Near-term (1–3 months)", timeline_score=22,
            )
            bant2 = BANTProfile(
                budget="Evaluating Fortivex paid tier", budget_score=18,
                authority="RevOps Manager (Team Manager / Evaluator)", authority_score=16,
                need="Low lead-to-demo conversion rate", need_score=18,
                timeline="Mid-term (1–2 quarters)", timeline_score=17,
            )

            conn.execute(
                """
                INSERT INTO leads (id, name, email, phone, company, role, preferred_language, score, tier, stage,
                                   estimated_acv_usd, bant_json, pain_points_json, objections_json, last_sentiment, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead1_id, "Elena Rostova", "elena@novascale.io", "+34 612 555 019", "NovaScale Inc", "VP of Sales",
                    "es", 94, "Hot", "Demo Scheduled", 42000,
                    bant1.model_dump_json(),
                    json.dumps(["Manual CRM data entry overhead", "Multilingual buyer coverage gap"]),
                    json.dumps(["GDPR data residency concern"]),
                    0.72, (now - timedelta(hours=5)).isoformat(), (now - timedelta(hours=4)).isoformat(),
                ),
            )
            conn.execute(
                """
                INSERT INTO leads (id, name, email, phone, company, role, preferred_language, score, tier, stage,
                                   estimated_acv_usd, bant_json, pain_points_json, objections_json, last_sentiment, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead2_id, "Marcus Weber", "m.weber@cloudwerk.de", "+49 30 5550 182", "CloudWerk GmbH", "RevOps Manager",
                    "de", 69, "Warm", "Proposal", 17988,
                    bant2.model_dump_json(),
                    json.dumps(["Low lead-to-demo conversion rate"]),
                    json.dumps(["Pricing / budget concern"]),
                    0.28, (now - timedelta(days=1)).isoformat(), (now - timedelta(hours=20)).isoformat(),
                ),
            )

            conn.execute(
                """
                INSERT INTO appointments (id, lead_id, session_id, lead_name, lead_email, company, slot_iso,
                                          duration_minutes, topic, timezone, status, meeting_link, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    "appt-demo-001", lead1_id, "sess-demo-001", "Elena Rostova", "elena@novascale.io",
                    "NovaScale Inc", tomorrow_15, 30,
                    "Fortivex Enterprise Multilingual & SOC2 Walkthrough", "UTC", "Confirmed",
                    "https://meet.fortivex.ai/demo/novascale-001", (now - timedelta(hours=4)).isoformat(),
                ),
            )

            summary1 = CallSummary(
                session_id="sess-demo-001",
                lead_id=lead1_id,
                lead_name="Elena Rostova",
                customer_email="elena@novascale.io",
                customer_phone="+34 612 555 019",
                company="NovaScale Inc",
                language="es",
                turn_count=6,
                duration_estimate_sec=210,
                executive_summary="Elena Rostova (VP of Sales at NovaScale Inc) evaluated Fortivex in Spanish for a 65-rep EMEA/LATAM team. Key drivers are eliminating manual CRM logging and supporting multilingual buyers under GDPR EU residency. Lead scored 94/100 (Hot) and confirmed an architecture demo.",
                key_pain_points=["Manual CRM data entry overhead", "Multilingual buyer coverage gap"],
                buying_signals=["buying_signal", "booking_intent"],
                objections_raised=["GDPR data residency concern"],
                competitor_mentions=[],
                knowledge_sources_used=["Security, Privacy & Regulatory Compliance", "Fortivex Pricing Tiers & ROI Benchmarks"],
                qualification=LeadQualification(
                    score=94, tier="Hot", stage="Demo Scheduled", bant=bant1,
                    missing_dimensions=[], estimated_acv_usd=42000,
                ),
                average_sentiment=0.68,
                sentiment_trend="Improving",
                appointment_outcome=f"Confirmed for {tomorrow_15}",
                next_steps=[
                    f"Host confirmed 30-min Enterprise Demo at {tomorrow_15}",
                    "Send Frankfurt EU data residency & SOC2 Type II pack to elena@novascale.io",
                ],
                follow_up_date=tomorrow_15[:10],
                generated_at=(now - timedelta(hours=4)).isoformat(),
            )
            conn.execute(
                "INSERT INTO call_summaries (session_id, summary_json, generated_at) VALUES (?, ?, ?)",
                ("sess-demo-001", summary1.model_dump_json(), (now - timedelta(hours=4)).isoformat()),
            )
            conn.commit()

    # ---------------- Sessions, Messages & 3-Tier Memory ----------------

    def get_or_create_session(self, session_id: str, language: SupportedLang = "en") -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row:
                metrics_raw = row["metrics_json"] if "metrics_json" in row.keys() else None
                metrics = (
                    SessionMetrics.model_validate_json(metrics_raw)
                    if metrics_raw
                    else SessionMetrics(call_started_at=row["created_at"], call_connected_at=row["created_at"])
                )
                return {
                    "session_id": row["session_id"],
                    "lead_id": row["lead_id"],
                    "language": row["language"],
                    "memory": LeadMemory.model_validate_json(row["memory_json"]),
                    "qualification": LeadQualification.model_validate_json(row["qualification_json"]),
                    "awaiting_slot_choice": bool(row["awaiting_slot_choice"]),
                    "proposed_slots": json.loads(row["proposed_slots_json"]),
                    "metrics": metrics,
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }

            now = utc_now_iso()
            mem = LeadMemory(preferred_language=language)
            qual = LeadQualification()
            metrics = SessionMetrics(call_started_at=now, call_connected_at=now)
            conn.execute(
                """
                INSERT INTO sessions (session_id, lead_id, language, memory_json, qualification_json,
                                      awaiting_slot_choice, proposed_slots_json, metrics_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, None, language, mem.model_dump_json(), qual.model_dump_json(),
                    0, "[]", metrics.model_dump_json(), now, now,
                ),
            )
            conn.commit()
            return {
                "session_id": session_id,
                "lead_id": None,
                "language": language,
                "memory": mem,
                "qualification": qual,
                "awaiting_slot_choice": False,
                "proposed_slots": [],
                "metrics": metrics,
                "created_at": now,
                "updated_at": now,
            }

    def list_sessions(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT s.session_id, s.lead_id, s.language, s.memory_json, s.qualification_json,
                       s.created_at, s.updated_at,
                       (SELECT COUNT(*) FROM messages m WHERE m.session_id = s.session_id) AS msg_count
                FROM sessions s
                ORDER BY s.updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
            out: list[dict[str, Any]] = []
            for r in rows:
                mem = json.loads(r["memory_json"])
                qual = json.loads(r["qualification_json"])
                out.append(
                    {
                        "session_id": r["session_id"],
                        "lead_id": r["lead_id"],
                        "lead_name": mem.get("name") or f"Prospect ({r['session_id'][-6:]})",
                        "company": mem.get("company"),
                        "language": r["language"],
                        "score": qual.get("score", 0),
                        "tier": qual.get("tier", "Cold"),
                        "stage": qual.get("stage", "Discovery"),
                        "message_count": r["msg_count"],
                        "created_at": r["created_at"],
                        "updated_at": r["updated_at"],
                    }
                )
            return out

    def save_session_state(
        self,
        session_id: str,
        lead_id: Optional[str],
        language: SupportedLang,
        memory: LeadMemory,
        qualification: LeadQualification,
        awaiting_slot_choice: bool,
        proposed_slots: list[str],
        metrics: Optional[SessionMetrics] = None,
    ) -> None:
        now = utc_now_iso()
        with self._connect() as conn:
            if metrics is not None:
                conn.execute(
                    """
                    UPDATE sessions
                    SET lead_id = ?, language = ?, memory_json = ?, qualification_json = ?,
                        awaiting_slot_choice = ?, proposed_slots_json = ?, metrics_json = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (
                        lead_id,
                        language,
                        memory.model_dump_json(),
                        qualification.model_dump_json(),
                        1 if awaiting_slot_choice else 0,
                        json.dumps(proposed_slots),
                        metrics.model_dump_json(),
                        now,
                        session_id,
                    ),
                )
            else:
                conn.execute(
                    """
                    UPDATE sessions
                    SET lead_id = ?, language = ?, memory_json = ?, qualification_json = ?,
                        awaiting_slot_choice = ?, proposed_slots_json = ?, updated_at = ?
                    WHERE session_id = ?
                    """,
                    (
                        lead_id,
                        language,
                        memory.model_dump_json(),
                        qualification.model_dump_json(),
                        1 if awaiting_slot_choice else 0,
                        json.dumps(proposed_slots),
                        now,
                        session_id,
                    ),
                )
            conn.commit()

    def get_session_metrics(self, session_id: str) -> SessionMetrics:
        sess = self.get_or_create_session(session_id)
        return sess["metrics"]

    def record_session_event(
        self,
        session_id: str,
        *,
        connected: bool = False,
        ended: bool = False,
        interruption: bool = False,
        user_turn: bool = False,
        agent_turn: bool = False,
        response_latency_ms: int = 0,
        stt_ms: int = 0,
        tts_ms: int = 0,
        speaking_seconds: float = 0.0,
    ) -> SessionMetrics:
        """Update real event-driven call metrics from live call timestamps and events."""
        sess = self.get_or_create_session(session_id)
        m: SessionMetrics = sess["metrics"]
        now_dt = utc_now()
        now_iso = now_dt.isoformat()

        if not m.call_started_at:
            m.call_started_at = sess["created_at"]
        if connected and not m.call_connected_at:
            m.call_connected_at = now_iso
        if interruption:
            m.interruption_count += 1
        if user_turn:
            m.user_turn_count += 1
        if agent_turn:
            prev_n = m.agent_turn_count
            m.agent_turn_count += 1
            if response_latency_ms > 0:
                m.average_response_latency_ms = round(
                    ((m.average_response_latency_ms * prev_n) + response_latency_ms) / m.agent_turn_count, 1
                )
        if stt_ms > 0:
            m.stt_processing_ms += stt_ms
        if tts_ms > 0:
            m.tts_processing_ms += tts_ms
        if speaking_seconds > 0:
            m.speaking_duration_seconds = round(m.speaking_duration_seconds + speaking_seconds, 2)
        if ended:
            m.call_ended_at = now_iso

        try:
            start_dt = datetime.fromisoformat(m.call_started_at.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat((m.call_ended_at or now_iso).replace("Z", "+00:00"))
            elapsed = max(0.0, (end_dt - start_dt).total_seconds())
        except Exception:
            elapsed = 0.0
        m.actual_duration_seconds = round(elapsed, 2)
        m.silence_duration_seconds = round(max(0.0, elapsed - m.speaking_duration_seconds), 2)

        with self._connect() as conn:
            conn.execute(
                "UPDATE sessions SET metrics_json = ?, updated_at = ? WHERE session_id = ?",
                (m.model_dump_json(), now_iso, session_id),
            )
            conn.commit()
        return m

    def add_message(self, msg: ConversationMessage) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (session_id, role, text, language, sentiment_json, retrieved_chunks_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    msg.session_id,
                    msg.role,
                    msg.text,
                    msg.language,
                    msg.sentiment.model_dump_json() if msg.sentiment else None,
                    json.dumps([c.model_dump() for c in msg.retrieved_chunks]),
                    msg.created_at,
                ),
            )
            conn.commit()

    def get_messages(self, session_id: str) -> list[ConversationMessage]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
            out: list[ConversationMessage] = []
            for r in rows:
                sent = SentimentResult.model_validate_json(r["sentiment_json"]) if r["sentiment_json"] else None
                out.append(
                    ConversationMessage(
                        id=r["id"],
                        session_id=r["session_id"],
                        role=r["role"],
                        text=r["text"],
                        language=r["language"],
                        sentiment=sent,
                        retrieved_chunks=json.loads(r["retrieved_chunks_json"]),
                        created_at=r["created_at"],
                    )
                )
            return out

    def get_three_tier_memory(self, session_id: str, memory: LeadMemory) -> ThreeTierMemorySnapshot:
        """Assemble short-term, session, and long-term cross-call customer memory."""
        messages = self.get_messages(session_id)
        short_term = [{"role": m.role, "text": m.text, "language": m.language} for m in messages[-6:]]
        session_facts = {
            "name": memory.name,
            "company": memory.company,
            "role": memory.role,
            "email": memory.email,
            "phone": memory.phone,
            "team_size": memory.team_size,
            "preferred_language": memory.preferred_language,
            "pain_points": memory.pain_points,
            "objections": memory.objections,
        }

        long_term_profile: Optional[dict[str, Any]] = None
        prev_interactions: list[str] = []
        appt_history: list[dict[str, Any]] = []

        if memory.lead_id:
            lead = self.get_lead(memory.lead_id)
            if lead:
                long_term_profile = {
                    "lead_id": lead.id,
                    "name": lead.name,
                    "company": lead.company,
                    "email": lead.email,
                    "phone": lead.phone,
                    "tier": lead.tier,
                    "stage": lead.stage,
                    "score": lead.score,
                }
                with self._connect() as conn:
                    sums = conn.execute(
                        "SELECT summary_json FROM call_summaries ORDER BY generated_at DESC LIMIT 10"
                    ).fetchall()
                    for row in sums:
                        s = json.loads(row["summary_json"])
                        if s.get("lead_id") == lead.id or (lead.email and s.get("customer_email") == lead.email):
                            if s.get("session_id") != session_id:
                                prev_interactions.append(s.get("executive_summary", ""))

                    appts = conn.execute(
                        "SELECT * FROM appointments WHERE lead_id = ? ORDER BY slot_iso DESC LIMIT 5",
                        (lead.id,),
                    ).fetchall()
                    appt_history = [dict(a) for a in appts]

        return ThreeTierMemorySnapshot(
            short_term_turns=short_term,
            session_facts=session_facts,
            long_term_profile=long_term_profile,
            previous_interactions=prev_interactions,
            appointment_history=appt_history,
        )

    # ---------------- CRM Leads, Score History & Tool Logs ----------------

    def find_returning_lead(
        self, email: Optional[str], phone: Optional[str], name: Optional[str], company: Optional[str]
    ) -> Optional[LeadRecord]:
        with self._connect() as conn:
            row = None
            if email:
                row = conn.execute("SELECT * FROM leads WHERE LOWER(email) = LOWER(?)", (email,)).fetchone()
            if not row and phone:
                row = conn.execute("SELECT * FROM leads WHERE phone = ?", (phone,)).fetchone()
            if not row and name and company:
                row = conn.execute(
                    "SELECT * FROM leads WHERE LOWER(name) = LOWER(?) AND LOWER(company) = LOWER(?)",
                    (name, company),
                ).fetchone()
            if not row:
                return None
            return self._row_to_lead(row)

    def upsert_lead_from_session(
        self,
        session_id: str,
        memory: LeadMemory,
        qualification: LeadQualification,
        last_sentiment: float,
        intent_score: int = 0,
    ) -> tuple[LeadRecord, str]:
        """Create or update a CRM Lead (preventing duplicates) and record score history."""
        now = utc_now_iso()
        existing = None
        if memory.lead_id:
            existing = self.get_lead(memory.lead_id)
        if not existing:
            existing = self.find_returning_lead(memory.email, memory.phone, memory.name, memory.company)

        lead_name = memory.name or (f"Prospect ({memory.company})" if memory.company else f"Voice Lead {session_id[-6:]}")

        with self._connect() as conn:
            if existing:
                lead_id = existing.id
                merged_pains = list(dict.fromkeys(existing.pain_points + memory.pain_points))
                merged_objs = list(dict.fromkeys(existing.objections + memory.objections))
                effective_score = (
                    qualification.score
                    if qualification.tier == "Disqualified"
                    else (max(existing.score, qualification.score) if qualification.score < 30 else qualification.score)
                )
                conn.execute(
                    """
                    UPDATE leads
                    SET name = ?, email = COALESCE(?, email), phone = COALESCE(?, phone),
                        company = COALESCE(?, company), role = COALESCE(?, role),
                        preferred_language = ?, score = ?, tier = ?, stage = ?,
                        estimated_acv_usd = ?, bant_json = ?, pain_points_json = ?,
                        objections_json = ?, last_sentiment = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        lead_name if memory.name else existing.name,
                        memory.email,
                        memory.phone,
                        memory.company,
                        memory.role,
                        memory.preferred_language,
                        effective_score,
                        qualification.tier,
                        qualification.stage,
                        qualification.estimated_acv_usd,
                        qualification.bant.model_dump_json(),
                        json.dumps(merged_pains),
                        json.dumps(merged_objs),
                        last_sentiment,
                        now,
                        lead_id,
                    ),
                )
                event_type = "crm.lead.updated"
            else:
                lead_id = f"lead-{uuid.uuid4().hex[:8]}"
                conn.execute(
                    """
                    INSERT INTO leads (id, name, email, phone, company, role, preferred_language, score, tier, stage,
                                       estimated_acv_usd, bant_json, pain_points_json, objections_json, last_sentiment,
                                       created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lead_id, lead_name, memory.email, memory.phone, memory.company, memory.role,
                        memory.preferred_language, qualification.score, qualification.tier,
                        qualification.stage, qualification.estimated_acv_usd,
                        qualification.bant.model_dump_json(),
                        json.dumps(memory.pain_points),
                        json.dumps(memory.objections),
                        last_sentiment, now, now,
                    ),
                )
                event_type = "crm.lead.created"

            conn.execute(
                """
                INSERT INTO score_history (lead_id, session_id, score, tier, stage, sentiment_polarity, intent_score, recorded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead_id, session_id, qualification.score, qualification.tier,
                    qualification.stage, last_sentiment, intent_score, now,
                ),
            )
            conn.commit()

        lead = self.get_lead(lead_id)
        assert lead is not None
        return lead, event_type

    def get_score_history(self, lead_id: str) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM score_history WHERE lead_id = ? ORDER BY id ASC", (lead_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def log_tool_execution(self, session_id: str, log: ToolExecutionLog) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO tool_execution_logs (session_id, tool_name, arguments_json, status, summary, duration_ms, executed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id, log.tool_name, json.dumps(log.arguments),
                    log.status, log.summary, log.duration_ms, log.executed_at,
                ),
            )
            conn.commit()

    def get_tool_logs(self, session_id: str) -> list[ToolExecutionLog]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tool_execution_logs WHERE session_id = ? ORDER BY id ASC", (session_id,)
            ).fetchall()
            return [
                ToolExecutionLog(
                    tool_name=r["tool_name"],
                    arguments=json.loads(r["arguments_json"]),
                    status=r["status"],
                    summary=r["summary"],
                    duration_ms=r["duration_ms"],
                    executed_at=r["executed_at"],
                )
                for r in rows
            ]

    def _row_to_lead(self, row: sqlite3.Row) -> LeadRecord:
        keys = row.keys()
        ext_ids_raw = row["external_crm_ids_json"] if "external_crm_ids_json" in keys and row["external_crm_ids_json"] else "{}"
        return LeadRecord(
            id=row["id"],
            name=row["name"],
            email=row["email"],
            phone=row["phone"] if "phone" in keys else None,
            company=row["company"],
            role=row["role"],
            preferred_language=row["preferred_language"],
            score=row["score"],
            tier=row["tier"],
            stage=row["stage"],
            estimated_acv_usd=row["estimated_acv_usd"],
            bant=BANTProfile.model_validate_json(row["bant_json"]),
            pain_points=json.loads(row["pain_points_json"]),
            objections=json.loads(row["objections_json"]),
            last_sentiment=row["last_sentiment"],
            external_crm_ids=json.loads(ext_ids_raw),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def set_lead_external_id(self, lead_id: str, provider: str, external_id: str) -> None:
        """Persist external CRM record ID (e.g., HubSpot contact ID or Salesforce Lead ID) for upserts."""
        with self._connect() as conn:
            row = conn.execute("SELECT external_crm_ids_json FROM leads WHERE id = ?", (lead_id,)).fetchone()
            if not row:
                return
            current = json.loads(row["external_crm_ids_json"] or "{}")
            current[provider] = external_id
            conn.execute(
                "UPDATE leads SET external_crm_ids_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(current), utc_now_iso(), lead_id),
            )
            conn.commit()

    def get_lead(self, lead_id: str) -> Optional[LeadRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
            return self._row_to_lead(row) if row else None

    def list_leads(self) -> list[LeadRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM leads ORDER BY updated_at DESC").fetchall()
            return [self._row_to_lead(r) for r in rows]

    # ---------------- Atomic Appointment Booking, Rescheduling & Cancellation ----------------

    def get_available_slots(
        self, count: int = 4, target_date: Optional[str] = None, tz_name: str = "UTC"
    ) -> list[str]:
        with self._connect() as conn:
            booked = {
                r["slot_iso"]
                for r in conn.execute(
                    "SELECT slot_iso FROM appointments WHERE status IN ('Confirmed', 'Rescheduled')"
                ).fetchall()
            }

        now_utc = utc_now()
        biz_tz = resolve_zoneinfo(settings.business_timezone)
        now_biz = now_utc.astimezone(biz_tz)
        candidates: list[str] = []
        start_day = 1
        if target_date:
            try:
                parsed = datetime.fromisoformat(target_date).date()
                delta = (parsed - now_biz.date()).days
                start_day = max(0, delta)
            except Exception:
                pass

        hours = [
            h
            for h in (10, 14, 16)
            if settings.business_hours_start <= h < settings.business_hours_end
        ] or [settings.business_hours_start]

        day_offset = start_day
        while len(candidates) < count and day_offset <= start_day + 14:
            day_biz = now_biz + timedelta(days=day_offset)
            if day_biz.weekday() < 5:
                for hr in hours:
                    slot_biz = day_biz.replace(hour=hr, minute=0, second=0, microsecond=0)
                    slot_utc = slot_biz.astimezone(timezone.utc)
                    if slot_utc <= now_utc + timedelta(minutes=15):
                        continue
                    slot_iso = slot_utc.isoformat().replace("+00:00", "Z")
                    if slot_iso not in booked:
                        candidates.append(slot_iso)
                        if len(candidates) >= count:
                            break
            day_offset += 1
        return candidates

    def book_appointment(self, req: AppointmentCreate) -> AppointmentRecord:
        """Atomically book an appointment using BEGIN IMMEDIATE lock to block concurrent double-booking."""
        normalized_slot, val_err = validate_business_slot(req.slot_iso, tz_name=req.timezone)
        if val_err or not normalized_slot:
            raise BookingValidationError(val_err or "Invalid appointment slot.")

        appt_id = f"appt-{uuid.uuid4().hex[:8]}"
        now = utc_now_iso()
        meeting_link = f"https://meet.fortivex.ai/demo/{appt_id}"

        conn = self._connect(isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conflict = conn.execute(
                "SELECT id FROM appointments WHERE slot_iso = ? AND status IN ('Confirmed', 'Rescheduled')",
                (normalized_slot,),
            ).fetchone()
            if conflict:
                conn.execute("ROLLBACK")
                raise BookingConflictError(
                    f"Slot {normalized_slot} is already booked. Please choose another available time."
                )

            conn.execute(
                """
                INSERT INTO appointments (id, lead_id, session_id, lead_name, lead_email, company, slot_iso,
                                          duration_minutes, topic, timezone, status, meeting_link, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    appt_id, req.lead_id, req.session_id, req.lead_name, req.lead_email,
                    req.company, normalized_slot, req.duration_minutes, req.topic, req.timezone,
                    "Confirmed", meeting_link, now,
                ),
            )
            if req.lead_id:
                conn.execute(
                    "UPDATE leads SET stage = 'Demo Scheduled', updated_at = ? WHERE id = ?",
                    (now, req.lead_id),
                )
            conn.execute("COMMIT")
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

        return AppointmentRecord(
            id=appt_id,
            lead_id=req.lead_id,
            session_id=req.session_id,
            lead_name=req.lead_name,
            lead_email=req.lead_email,
            company=req.company,
            slot_iso=normalized_slot,
            duration_minutes=req.duration_minutes,
            topic=req.topic,
            timezone=req.timezone,
            status="Confirmed",
            meeting_link=meeting_link,
            created_at=now,
        )

    def reschedule_appointment(self, appt_id: str, payload: AppointmentReschedule) -> Optional[AppointmentRecord]:
        """Atomically reschedule an existing appointment after conflict & business-hours checks."""
        tz = payload.timezone or "UTC"
        normalized_slot, val_err = validate_business_slot(payload.slot_iso, tz_name=tz)
        if val_err or not normalized_slot:
            raise BookingValidationError(val_err or "Invalid reschedule slot.")

        conn = self._connect(isolation_level=None)
        try:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT * FROM appointments WHERE id = ?", (appt_id,)).fetchone()
            if not existing:
                conn.execute("ROLLBACK")
                return None

            conflict = conn.execute(
                "SELECT id FROM appointments WHERE slot_iso = ? AND id != ? AND status IN ('Confirmed', 'Rescheduled')",
                (normalized_slot, appt_id),
            ).fetchone()
            if conflict:
                conn.execute("ROLLBACK")
                raise BookingConflictError(f"Slot {normalized_slot} is already occupied.")

            conn.execute(
                "UPDATE appointments SET slot_iso = ?, timezone = ?, status = 'Rescheduled' WHERE id = ?",
                (normalized_slot, tz, appt_id),
            )
            conn.execute("COMMIT")
            updated = conn.execute("SELECT * FROM appointments WHERE id = ?", (appt_id,)).fetchone()
            return AppointmentRecord(**dict(updated))
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()

    def get_session_appointment(self, session_id: str) -> Optional[AppointmentRecord]:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT * FROM appointments WHERE session_id = ? AND status IN ('Confirmed', 'Rescheduled') ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            return AppointmentRecord(**dict(r)) if r else None

    def list_appointments(self) -> list[AppointmentRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM appointments ORDER BY slot_iso ASC").fetchall()
            return [AppointmentRecord(**dict(r)) for r in rows]

    def cancel_appointment(self, appt_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("UPDATE appointments SET status = 'Cancelled' WHERE id = ?", (appt_id,))
            conn.commit()
            return cur.rowcount > 0

    # ---------------- Webhooks, CRM Sync Logs, Summaries & Custom Docs ----------------

    def record_webhook(self, event_type: str, target: str, payload: dict[str, Any], delivered: bool = True) -> dict[str, Any]:
        evt = {
            "id": f"wh-{uuid.uuid4().hex[:8]}",
            "event_type": event_type,
            "target": target,
            "payload": payload,
            "delivered": delivered,
            "created_at": utc_now_iso(),
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO webhook_events (id, event_type, target, payload_json, delivered, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (evt["id"], event_type, target, json.dumps(payload), 1 if delivered else 0, evt["created_at"]),
            )
            conn.commit()
        return evt

    def record_crm_sync_log(
        self,
        adapter: str,
        operation: str,
        idempotency_key: str,
        status: str,
        http_status: Optional[int],
        attempts: int,
        detail: str,
        external_record_id: Optional[str] = None,
        error_code: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO crm_sync_logs (id, adapter, operation, idempotency_key, status, http_status,
                                           attempts, external_record_id, error_code, detail, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    f"sync-{uuid.uuid4().hex[:8]}",
                    adapter,
                    operation,
                    idempotency_key,
                    status,
                    http_status,
                    attempts,
                    external_record_id,
                    error_code,
                    detail,
                    utc_now_iso(),
                ),
            )
            conn.commit()

    def get_successful_crm_sync(self, idempotency_key: str) -> Optional[dict[str, Any]]:
        """Check if an idempotent CRM sync operation already succeeded."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM crm_sync_logs WHERE idempotency_key = ? AND status = 'success' ORDER BY created_at DESC LIMIT 1",
                (idempotency_key,),
            ).fetchone()
            return dict(row) if row else None

    def upsert_crm_sync_job(
        self,
        adapter: str,
        operation: str,
        entity_id: str,
        idempotency_key: str,
        payload: dict[str, Any],
        status: str,
        retry_count: int,
        http_status: Optional[int],
        external_record_id: Optional[str],
        error_code: Optional[str],
        failure_reason: Optional[str],
    ) -> str:
        """Persist CRM synchronization job state (pending / synced / failed dead-letter) safely."""
        now = utc_now_iso()
        job_id = f"job-{uuid.uuid4().hex[:8]}"
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO crm_sync_jobs (id, adapter, operation, entity_id, idempotency_key, payload_json,
                                           status, retry_count, http_status, external_record_id,
                                           error_code, failure_reason, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(idempotency_key) DO UPDATE SET
                    status = excluded.status,
                    retry_count = excluded.retry_count,
                    http_status = excluded.http_status,
                    external_record_id = COALESCE(excluded.external_record_id, crm_sync_jobs.external_record_id),
                    error_code = excluded.error_code,
                    failure_reason = excluded.failure_reason,
                    updated_at = excluded.updated_at
                """,
                (
                    job_id, adapter, operation, entity_id, idempotency_key, json.dumps(payload),
                    status, retry_count, http_status, external_record_id, error_code, failure_reason, now, now,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT id FROM crm_sync_jobs WHERE idempotency_key = ?", (idempotency_key,)).fetchone()
            return row["id"] if row else job_id

    def list_crm_sync_jobs(self, status: Optional[str] = None, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM crm_sync_jobs WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM crm_sync_jobs ORDER BY updated_at DESC LIMIT ?", (limit,)
                ).fetchall()
            return [dict(r) for r in rows]

    def get_crm_sync_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM crm_sync_jobs WHERE id = ?", (job_id,)).fetchone()
            return dict(row) if row else None

    def list_crm_sync_logs(self, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM crm_sync_logs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    def list_webhooks(self, limit: int = 25) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM webhook_events ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [
                {
                    "id": r["id"],
                    "event_type": r["event_type"],
                    "target": r["target"],
                    "payload": json.loads(r["payload_json"]),
                    "delivered": bool(r["delivered"]),
                    "created_at": r["created_at"],
                }
                for r in rows
            ]

    def save_call_summary(self, summary: CallSummary) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO call_summaries (session_id, summary_json, generated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET summary_json = excluded.summary_json, generated_at = excluded.generated_at
                """,
                (summary.session_id, summary.model_dump_json(), summary.generated_at),
            )
            conn.commit()

    def get_call_summary(self, session_id: str) -> Optional[CallSummary]:
        with self._connect() as conn:
            r = conn.execute("SELECT summary_json FROM call_summaries WHERE session_id = ?", (session_id,)).fetchone()
            return CallSummary.model_validate_json(r["summary_json"]) if r else None

    def list_call_summaries(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        tier: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> list[CallSummary]:
        with self._connect() as conn:
            rows = conn.execute("SELECT summary_json FROM call_summaries ORDER BY generated_at DESC").fetchall()
        summaries = [CallSummary.model_validate_json(r["summary_json"]) for r in rows]
        filtered: list[CallSummary] = []
        for s in summaries:
            gen_date = s.generated_at[:10]
            if date_from and gen_date < date_from:
                continue
            if date_to and gen_date > date_to:
                continue
            if tier and s.qualification.tier.lower() != tier.lower():
                continue
            if outcome == "booked" and not s.booked_appointment and "Confirmed" not in s.appointment_outcome:
                continue
            if outcome == "unbooked" and (s.booked_appointment or "Confirmed" in s.appointment_outcome):
                continue
            filtered.append(s)
        return filtered

    def save_custom_doc(self, title: str, category: str, content: str, doc_id: Optional[str] = None) -> str:
        did = doc_id or f"doc-{uuid.uuid4().hex[:8]}"
        now = utc_now_iso()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO custom_docs (id, title, category, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (did, title, category, content, now),
            )
            conn.commit()
        return did

    def delete_custom_doc(self, doc_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM custom_docs WHERE id = ?", (doc_id,))
            conn.commit()
            return cur.rowcount > 0

    def list_custom_docs(self) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM custom_docs ORDER BY created_at ASC").fetchall()
            return [dict(r) for r in rows]

    def get_business_insights(
        self,
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        tier: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> BusinessInsights:
        leads = self.list_leads()
        if date_from:
            leads = [l for l in leads if l.updated_at[:10] >= date_from]
        if date_to:
            leads = [l for l in leads if l.updated_at[:10] <= date_to]
        if tier:
            leads = [l for l in leads if l.tier.lower() == tier.lower()]

        appts = [a for a in self.list_appointments() if a.status in {"Confirmed", "Rescheduled"}]
        if date_from:
            appts = [a for a in appts if a.created_at[:10] >= date_from]
        if date_to:
            appts = [a for a in appts if a.created_at[:10] <= date_to]

        if outcome == "booked":
            booked_lead_ids = {a.lead_id for a in appts if a.lead_id}
            leads = [l for l in leads if l.id in booked_lead_ids or l.stage == "Demo Scheduled"]
        elif outcome == "unbooked":
            booked_lead_ids = {a.lead_id for a in appts if a.lead_id}
            leads = [l for l in leads if l.id not in booked_lead_ids and l.stage != "Demo Scheduled"]

        with self._connect() as conn:
            sess_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            sum_rows = self.list_call_summaries(date_from=date_from, date_to=date_to, tier=tier, outcome=outcome)
            faq_rows = conn.execute("SELECT retrieved_chunks_json FROM messages WHERE role = 'user'").fetchall()

        total_calls = len(sum_rows) if (date_from or date_to or tier or outcome) else max(sess_count, len(sum_rows), len(leads))
        total_leads = len(leads)
        unique_keys = {(l.email or l.phone or l.id).lower() for l in leads}
        hot = sum(1 for l in leads if l.tier == "Hot")
        warm = sum(1 for l in leads if l.tier == "Warm")
        cold = sum(1 for l in leads if l.tier == "Cold")
        disq = sum(1 for l in leads if l.tier == "Disqualified")
        qualified = hot + warm
        booked = len(appts)

        conv = round((booked / total_leads) * 100, 1) if total_leads else 0.0
        booking_rate = round((booked / total_calls) * 100, 1) if total_calls else 0.0
        avg_score = round(sum(l.score for l in leads) / total_leads, 1) if total_leads else 0.0
        avg_sent = round(sum(l.last_sentiment for l in leads) / total_leads, 2) if total_leads else 0.0
        pipeline = sum(l.estimated_acv_usd for l in leads if l.tier in {"Hot", "Warm"} or l.stage == "Demo Scheduled")

        lang_counts = Counter(l.preferred_language for l in leads)
        pain_counts = Counter(p for l in leads for p in l.pain_points)
        obj_counts = Counter(o for l in leads for o in l.objections)
        stage_counts = Counter(l.stage for l in leads)

        faq_counts: Counter[str] = Counter()
        for r in faq_rows:
            for ch in json.loads(r["retrieved_chunks_json"]):
                title = ch.get("title")
                if title:
                    faq_counts[title] += 1

        avg_turns = round(sum(s.turn_count for s in sum_rows) / len(sum_rows), 1) if sum_rows else 0.0
        durations = [
            s.actual_duration_seconds if s.actual_duration_seconds > 0 else float(s.duration_estimate_sec)
            for s in sum_rows
        ]
        avg_dur = round(sum(durations) / len(durations), 1) if durations else 0.0
        avg_lat = (
            round(sum(s.average_response_latency_ms for s in sum_rows) / len(sum_rows), 1)
            if sum_rows
            else 0.0
        )
        total_interruptions = sum(s.interruption_count for s in sum_rows)
        return BusinessInsights(
            total_calls=total_calls,
            unique_leads=len(unique_keys),
            total_leads=total_leads,
            qualified_leads=qualified,
            hot_leads=hot,
            warm_leads=warm,
            cold_leads=cold,
            disqualified_leads=disq,
            booked_demos=booked,
            conversion_rate_pct=conv,
            booking_rate_pct=booking_rate,
            average_lead_score=avg_score,
            average_sentiment=avg_sent,
            average_call_duration_sec=avg_dur,
            total_pipeline_usd=pipeline,
            language_distribution=dict(lang_counts),
            top_pain_points=[{"label": k, "count": v} for k, v in pain_counts.most_common(5)],
            top_objections=[{"label": k, "count": v} for k, v in obj_counts.most_common(5)],
            frequently_asked_questions=[{"topic": k, "count": v} for k, v in faq_counts.most_common(5)],
            stage_funnel=dict(stage_counts),
            agent_performance={
                "avg_turns_per_call": avg_turns,
                "avg_call_duration_sec": avg_dur,
                "avg_response_latency_ms": avg_lat,
                "total_interruptions": total_interruptions,
                "qualification_rate_pct": round((qualified / total_leads) * 100, 1) if total_leads else 0.0,
                "summaries_generated": len(sum_rows),
            },
        )

    def cleanup_old_records(self, retention_days: Optional[int] = None) -> dict[str, int]:
        """Enforce configurable data retention policy by purging expired sessions & logs."""
        days = retention_days if retention_days is not None else settings.data_retention_days
        cutoff = (utc_now() - timedelta(days=days)).isoformat()
        with self._connect() as conn:
            m_cur = conn.execute("DELETE FROM messages WHERE created_at < ?", (cutoff,))
            t_cur = conn.execute("DELETE FROM tool_execution_logs WHERE executed_at < ?", (cutoff,))
            w_cur = conn.execute("DELETE FROM webhook_events WHERE created_at < ?", (cutoff,))
            conn.commit()
            return {
                "deleted_messages": m_cur.rowcount,
                "deleted_tool_logs": t_cur.rowcount,
                "deleted_webhooks": w_cur.rowcount,
            }

    def create_backup(self) -> Path:
        """Create an atomic online SQLite backup snapshot."""
        backup_dir = Path(settings.backup_dir)
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = utc_now().strftime("%Y%m%d_%H%M%S")
        dest_path = backup_dir / f"fortivex_backup_{stamp}.db"
        with self._connect() as src, sqlite3.connect(str(dest_path)) as dst:
            src.backup(dst)
        return dest_path
