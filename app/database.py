"""SQLite persistence layer for CRM Leads, Sessions, Appointments, Webhooks & Analytics."""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from app.config import settings
from app.models import (
    AppointmentCreate,
    AppointmentRecord,
    BANTProfile,
    BusinessInsights,
    CallSummary,
    ConversationMessage,
    LeadMemory,
    LeadQualification,
    LeadRecord,
    SentimentResult,
    SupportedLang,
)


class DatabaseStore:
    """Thread-safe SQLite store for CRM, calendar booking, memory, and call analytics."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path or settings.fortivex_db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()
        self._seed_demo_data_if_empty()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS leads (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    email TEXT,
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

                CREATE TABLE IF NOT EXISTS webhook_events (
                    id TEXT PRIMARY KEY,
                    event_type TEXT NOT NULL,
                    target TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    delivered INTEGER DEFAULT 1,
                    created_at TEXT NOT NULL
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

    def _seed_demo_data_if_empty(self) -> None:
        """Seed two realistic sample CRM leads & one upcoming demo so dashboard is rich on first boot."""
        with self._connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM leads").fetchone()[0]
            if count > 0:
                return
            now = datetime.utcnow()
            tomorrow_15 = (now + timedelta(days=1)).replace(hour=15, minute=0, second=0, microsecond=0).isoformat() + "Z"

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
                INSERT INTO leads (id, name, email, company, role, preferred_language, score, tier, stage,
                                   estimated_acv_usd, bant_json, pain_points_json, objections_json, last_sentiment, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead1_id, "Elena Rostova", "elena@novascale.io", "NovaScale Inc", "VP of Sales",
                    "es", 94, "Hot", "Demo Scheduled", 42000,
                    bant1.model_dump_json(),
                    json.dumps(["Manual CRM data entry overhead", "Multilingual buyer coverage gap"]),
                    json.dumps(["GDPR data residency concern"]),
                    0.72, (now - timedelta(hours=5)).isoformat(), (now - timedelta(hours=4)).isoformat(),
                ),
            )
            conn.execute(
                """
                INSERT INTO leads (id, name, email, company, role, preferred_language, score, tier, stage,
                                   estimated_acv_usd, bant_json, pain_points_json, objections_json, last_sentiment, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    lead2_id, "Marcus Weber", "m.weber@cloudwerk.de", "CloudWerk GmbH", "RevOps Manager",
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

            # Seed summary for lead 1 so insights view has historical context
            summary1 = CallSummary(
                session_id="sess-demo-001",
                lead_id=lead1_id,
                lead_name="Elena Rostova",
                company="NovaScale Inc",
                language="es",
                turn_count=6,
                duration_estimate_sec=210,
                executive_summary="Elena Rostova (VP of Sales at NovaScale Inc) evaluated Fortivex in Spanish for her 65-rep EMEA/LATAM team. Key drivers are eliminating manual CRM logging and supporting bilingual Spanish/English prospects under GDPR EU residency. Lead scored 94/100 (Hot) and booked an architecture demo.",
                key_pain_points=["Manual CRM data entry overhead", "Multilingual buyer coverage gap"],
                objections_raised=["GDPR data residency concern"],
                qualification=LeadQualification(
                    score=94, tier="Hot", stage="Demo Scheduled", bant=bant1,
                    missing_dimensions=[], estimated_acv_usd=42000,
                ),
                average_sentiment=0.68,
                sentiment_trend="Improving",
                next_steps=[
                    f"Host confirmed 30-min Enterprise Demo at {tomorrow_15}",
                    "Send Frankfurt EU data residency & SOC2 Type II pack to elena@novascale.io",
                ],
                generated_at=(now - timedelta(hours=4)).isoformat(),
            )
            conn.execute(
                "INSERT INTO call_summaries (session_id, summary_json, generated_at) VALUES (?, ?, ?)",
                ("sess-demo-001", summary1.model_dump_json(), (now - timedelta(hours=4)).isoformat()),
            )
            conn.commit()

    # ---------------- Session & Memory ----------------

    def get_or_create_session(self, session_id: str, language: SupportedLang = "en") -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM sessions WHERE session_id = ?", (session_id,)).fetchone()
            if row:
                return {
                    "session_id": row["session_id"],
                    "lead_id": row["lead_id"],
                    "language": row["language"],
                    "memory": LeadMemory.model_validate_json(row["memory_json"]),
                    "qualification": LeadQualification.model_validate_json(row["qualification_json"]),
                    "awaiting_slot_choice": bool(row["awaiting_slot_choice"]),
                    "proposed_slots": json.loads(row["proposed_slots_json"]),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }

            now = datetime.utcnow().isoformat()
            mem = LeadMemory(preferred_language=language)
            qual = LeadQualification()
            conn.execute(
                """
                INSERT INTO sessions (session_id, lead_id, language, memory_json, qualification_json,
                                      awaiting_slot_choice, proposed_slots_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (session_id, None, language, mem.model_dump_json(), qual.model_dump_json(), 0, "[]", now, now),
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
                "created_at": now,
                "updated_at": now,
            }

    def save_session_state(
        self,
        session_id: str,
        lead_id: Optional[str],
        language: SupportedLang,
        memory: LeadMemory,
        qualification: LeadQualification,
        awaiting_slot_choice: bool,
        proposed_slots: list[str],
    ) -> None:
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
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

    # ---------------- CRM Leads & Cross-Call Memory ----------------

    def find_returning_lead(self, email: Optional[str], name: Optional[str], company: Optional[str]) -> Optional[LeadRecord]:
        with self._connect() as conn:
            row = None
            if email:
                row = conn.execute("SELECT * FROM leads WHERE LOWER(email) = LOWER(?)", (email,)).fetchone()
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
    ) -> tuple[LeadRecord, list[dict[str, Any]]]:
        """Create or update a CRM Lead and record an outbound CRM webhook event."""
        now = datetime.utcnow().isoformat()
        existing = None
        if memory.lead_id:
            existing = self.get_lead(memory.lead_id)
        if not existing:
            existing = self.find_returning_lead(memory.email, memory.name, memory.company)

        lead_name = memory.name or (f"Prospect ({memory.company})" if memory.company else f"Voice Lead {session_id[-6:]}")

        with self._connect() as conn:
            if existing:
                lead_id = existing.id
                merged_pains = list(dict.fromkeys(existing.pain_points + memory.pain_points))
                merged_objs = list(dict.fromkeys(existing.objections + memory.objections))
                conn.execute(
                    """
                    UPDATE leads
                    SET name = ?, email = COALESCE(?, email), company = COALESCE(?, company),
                        role = COALESCE(?, role), preferred_language = ?, score = ?, tier = ?,
                        stage = ?, estimated_acv_usd = ?, bant_json = ?, pain_points_json = ?,
                        objections_json = ?, last_sentiment = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        lead_name if memory.name else existing.name,
                        memory.email,
                        memory.company,
                        memory.role,
                        memory.preferred_language,
                        max(existing.score, qualification.score) if qualification.score < 30 else qualification.score,
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
                    INSERT INTO leads (id, name, email, company, role, preferred_language, score, tier, stage,
                                       estimated_acv_usd, bant_json, pain_points_json, objections_json, last_sentiment,
                                       created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        lead_id, lead_name, memory.email, memory.company, memory.role,
                        memory.preferred_language, qualification.score, qualification.tier,
                        qualification.stage, qualification.estimated_acv_usd,
                        qualification.bant.model_dump_json(),
                        json.dumps(memory.pain_points),
                        json.dumps(memory.objections),
                        last_sentiment, now, now,
                    ),
                )
                event_type = "crm.lead.created"
            conn.commit()

        lead = self.get_lead(lead_id)
        assert lead is not None
        webhook = self.record_webhook(
            event_type=event_type,
            target="Salesforce / HubSpot Bidirectional Sync",
            payload={
                "event": event_type,
                "session_id": session_id,
                "lead_id": lead.id,
                "properties": {
                    "firstname_lastname": lead.name,
                    "email": lead.email,
                    "company": lead.company,
                    "jobtitle": lead.role,
                    "hs_lead_status": lead.tier,
                    "lifecyclestage": lead.stage,
                    "fortivex_bant_score": lead.score,
                    "amount": lead.estimated_acv_usd,
                    "preferred_language": lead.preferred_language,
                },
            },
        )
        return lead, [webhook]

    def _row_to_lead(self, row: sqlite3.Row) -> LeadRecord:
        return LeadRecord(
            id=row["id"],
            name=row["name"],
            email=row["email"],
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
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def get_lead(self, lead_id: str) -> Optional[LeadRecord]:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM leads WHERE id = ?", (lead_id,)).fetchone()
            return self._row_to_lead(row) if row else None

    def list_leads(self) -> list[LeadRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM leads ORDER BY updated_at DESC").fetchall()
            return [self._row_to_lead(r) for r in rows]

    # ---------------- Calendar & Appointment Booking ----------------

    def get_available_slots(self, count: int = 4) -> list[str]:
        """Return next conflict-free 30-minute business slots starting tomorrow."""
        with self._connect() as conn:
            booked = {
                r["slot_iso"]
                for r in conn.execute(
                    "SELECT slot_iso FROM appointments WHERE status = 'Confirmed'"
                ).fetchall()
            }

        now = datetime.utcnow()
        candidates: list[str] = []
        day_offset = 1
        while len(candidates) < count and day_offset <= 10:
            day = now + timedelta(days=day_offset)
            if day.weekday() < 5:  # Mon-Fri
                for hr in (10, 14, 16):
                    slot_dt = day.replace(hour=hr, minute=0, second=0, microsecond=0)
                    slot_iso = slot_dt.isoformat() + "Z"
                    if slot_iso not in booked:
                        candidates.append(slot_iso)
                        if len(candidates) >= count:
                            break
            day_offset += 1
        return candidates

    def book_appointment(self, req: AppointmentCreate) -> tuple[AppointmentRecord, dict[str, Any]]:
        """Book an appointment after checking for slot conflicts."""
        with self._connect() as conn:
            conflict = conn.execute(
                "SELECT id FROM appointments WHERE slot_iso = ? AND status = 'Confirmed'",
                (req.slot_iso,),
            ).fetchone()
            if conflict:
                # Auto-shift by 1 hour to guarantee conflict-free scheduling
                dt = datetime.fromisoformat(req.slot_iso.replace("Z", "")) + timedelta(hours=1)
                slot_iso = dt.isoformat() + "Z"
            else:
                slot_iso = req.slot_iso

            appt_id = f"appt-{uuid.uuid4().hex[:8]}"
            now = datetime.utcnow().isoformat()
            meeting_link = f"https://meet.fortivex.ai/demo/{appt_id}"
            conn.execute(
                """
                INSERT INTO appointments (id, lead_id, session_id, lead_name, lead_email, company, slot_iso,
                                          duration_minutes, topic, timezone, status, meeting_link, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    appt_id, req.lead_id, req.session_id, req.lead_name, req.lead_email,
                    req.company, slot_iso, req.duration_minutes, req.topic, req.timezone,
                    "Confirmed", meeting_link, now,
                ),
            )
            if req.lead_id:
                conn.execute(
                    "UPDATE leads SET stage = 'Demo Scheduled', updated_at = ? WHERE id = ?",
                    (now, req.lead_id),
                )
            conn.commit()

        record = AppointmentRecord(
            id=appt_id,
            lead_id=req.lead_id,
            session_id=req.session_id,
            lead_name=req.lead_name,
            lead_email=req.lead_email,
            company=req.company,
            slot_iso=slot_iso,
            duration_minutes=req.duration_minutes,
            topic=req.topic,
            timezone=req.timezone,
            status="Confirmed",
            meeting_link=meeting_link,
            created_at=now,
        )
        webhook = self.record_webhook(
            event_type="calendar.appointment.booked",
            target="Google Calendar / Outlook + CRM Opportunity",
            payload=record.model_dump(),
        )
        return record, webhook

    def get_session_appointment(self, session_id: str) -> Optional[AppointmentRecord]:
        with self._connect() as conn:
            r = conn.execute(
                "SELECT * FROM appointments WHERE session_id = ? AND status = 'Confirmed' ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
            if not r:
                return None
            return AppointmentRecord(**dict(r))

    def list_appointments(self) -> list[AppointmentRecord]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM appointments ORDER BY slot_iso ASC").fetchall()
            return [AppointmentRecord(**dict(r)) for r in rows]

    def cancel_appointment(self, appt_id: str) -> bool:
        with self._connect() as conn:
            cur = conn.execute(
                "UPDATE appointments SET status = 'Cancelled' WHERE id = ?", (appt_id,)
            )
            conn.commit()
            return cur.rowcount > 0

    # ---------------- Webhooks, Summaries, Custom KB & Insights ----------------

    def record_webhook(self, event_type: str, target: str, payload: dict[str, Any]) -> dict[str, Any]:
        evt = {
            "id": f"wh-{uuid.uuid4().hex[:8]}",
            "event_type": event_type,
            "target": target,
            "payload": payload,
            "delivered": True,
            "created_at": datetime.utcnow().isoformat(),
        }
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO webhook_events (id, event_type, target, payload_json, delivered, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (evt["id"], event_type, target, json.dumps(payload), 1, evt["created_at"]),
            )
            conn.commit()
        return evt

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

    def list_call_summaries(self) -> list[CallSummary]:
        with self._connect() as conn:
            rows = conn.execute("SELECT summary_json FROM call_summaries ORDER BY generated_at DESC").fetchall()
            return [CallSummary.model_validate_json(r["summary_json"]) for r in rows]

    def save_custom_doc(self, title: str, category: str, content: str) -> str:
        doc_id = f"doc-{uuid.uuid4().hex[:8]}"
        now = datetime.utcnow().isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO custom_docs (id, title, category, content, created_at) VALUES (?, ?, ?, ?, ?)",
                (doc_id, title, category, content, now),
            )
            conn.commit()
        return doc_id

    def list_custom_docs(self) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM custom_docs ORDER BY created_at ASC").fetchall()
            return [dict(r) for r in rows]

    def get_business_insights(self) -> BusinessInsights:
        leads = self.list_leads()
        appts = [a for a in self.list_appointments() if a.status == "Confirmed"]
        with self._connect() as conn:
            sess_count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
            sum_count = conn.execute("SELECT COUNT(*) FROM call_summaries").fetchone()[0]

        total_calls = max(sess_count, sum_count, len(leads))
        total_leads = len(leads)
        hot = sum(1 for l in leads if l.tier == "Hot")
        warm = sum(1 for l in leads if l.tier == "Warm")
        cold = sum(1 for l in leads if l.tier == "Cold")
        booked = len(appts)

        conv = round((booked / total_leads) * 100, 1) if total_leads else 0.0
        avg_score = round(sum(l.score for l in leads) / total_leads, 1) if total_leads else 0.0
        avg_sent = round(sum(l.last_sentiment for l in leads) / total_leads, 2) if total_leads else 0.0
        pipeline = sum(l.estimated_acv_usd for l in leads if l.tier in {"Hot", "Warm"} or l.stage == "Demo Scheduled")

        lang_counts = Counter(l.preferred_language for l in leads)
        pain_counts = Counter(p for l in leads for p in l.pain_points)
        obj_counts = Counter(o for l in leads for o in l.objections)
        stage_counts = Counter(l.stage for l in leads)

        return BusinessInsights(
            total_calls=total_calls,
            total_leads=total_leads,
            hot_leads=hot,
            warm_leads=warm,
            cold_leads=cold,
            booked_demos=booked,
            conversion_rate_pct=conv,
            average_lead_score=avg_score,
            average_sentiment=avg_sent,
            total_pipeline_usd=pipeline,
            language_distribution=dict(lang_counts),
            top_pain_points=[{"label": k, "count": v} for k, v in pain_counts.most_common(5)],
            top_objections=[{"label": k, "count": v} for k, v in obj_counts.most_common(5)],
            stage_funnel=dict(stage_counts),
        )
