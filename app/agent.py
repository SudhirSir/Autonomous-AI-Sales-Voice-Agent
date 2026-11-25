"""Autonomous Sales Voice Agent Orchestrator (Tools + 3-Tier Memory + Hybrid RAG + CRM)."""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta
from typing import Any, Optional

from app.config import get_runtime_gemini_key, settings
from app.crm import CRMIntegrationManager
from app.database import BookingConflictError, BookingValidationError, DatabaseStore
from app.models import (
    AppointmentRecord,
    CallSummary,
    ConversationMessage,
    LeadMemory,
    LeadQualification,
    RetrievedChunk,
    SentimentResult,
    SupportedLang,
    ThreeTierMemorySnapshot,
    ToolExecutionLog,
    TurnRequest,
    TurnResponse,
    utc_now,
    utc_now_iso,
)
from app.nlp import (
    LANGUAGE_NAMES,
    TTS_LOCALES,
    analyze_sentiment,
    compute_qualification,
    detect_language,
    extract_memory_and_bant,
    parse_requested_slot,
)
from app.rag import HybridRAGRetriever
from app.security import mask_pii
from app.tools import ToolRegistry
from app.voice import VoiceProviderRegistry

logger = logging.getLogger(__name__)


def _format_slot_human(slot_iso: str, lang: SupportedLang) -> str:
    dt = datetime.fromisoformat(slot_iso.replace("Z", "+00:00"))
    if lang == "es":
        return dt.strftime("%d/%m a las %H:%M UTC")
    if lang == "fr":
        return dt.strftime("%d/%m à %H:%M UTC")
    if lang == "de":
        return dt.strftime("%d.%m. um %H:%M UTC")
    if lang == "hi":
        return dt.strftime("%d %b को %I:%M %p UTC")
    if lang == "hinglish":
        return dt.strftime("%d %b ko %I:%M %p UTC")
    return dt.strftime("%A, %b %d at %I:%M %p UTC")


class SalesVoiceAgent:
    """Orchestrates voice conversations using validated tools, 3-tier memory, and hybrid RAG."""

    def __init__(
        self,
        db: DatabaseStore,
        rag: HybridRAGRetriever,
        crm: Optional[CRMIntegrationManager] = None,
        voice: Optional[VoiceProviderRegistry] = None,
    ) -> None:
        self.db = db
        self.rag = rag
        self.crm = crm or CRMIntegrationManager(db)
        self.voice = voice or VoiceProviderRegistry()
        self.tools = ToolRegistry(db=db, rag=rag, crm=self.crm)

        for doc in self.db.list_custom_docs():
            self.rag.add_document(
                doc["title"],
                doc["category"],
                doc["content"],
                parent_id=doc["id"],
            )

    def process_turn(self, req: TurnRequest) -> TurnResponse:
        t0 = time.perf_counter()
        sess = self.db.get_or_create_session(req.session_id)
        prev_lang: SupportedLang = sess["language"]
        lang: SupportedLang = req.language_override or detect_language(req.text, fallback=prev_lang)

        memory: LeadMemory = sess["memory"]
        memory.preferred_language = lang
        qualification: LeadQualification = sess["qualification"]
        awaiting_slot_choice: bool = sess["awaiting_slot_choice"]
        proposed_slots: list[str] = sess["proposed_slots"]
        tool_logs: list[ToolExecutionLog] = []

        # 1. Sentiment & Purchase Intent analysis
        sentiment = analyze_sentiment(req.text)

        # 2. Extract entity updates & BANT signals
        memory, updated_bant = extract_memory_and_bant(req.text, memory, qualification.bant)
        for sig in sentiment.signals:
            if sig not in memory.buying_signals:
                memory.buying_signals.append(sig)

        # 3. Cross-call long-term lookup before generating response
        if not memory.lead_id and (memory.email or memory.phone or (memory.name and memory.company)):
            returning = self.db.find_returning_lead(memory.email, memory.phone, memory.name, memory.company)
            if returning:
                memory.lead_id = returning.id
                memory.returning_lead = True
                memory.prior_calls_count += 1
                memory.name = memory.name or returning.name
                memory.email = memory.email or returning.email
                memory.phone = memory.phone or returning.phone
                memory.company = memory.company or returning.company
                memory.role = memory.role or returning.role
                memory.pain_points = list(dict.fromkeys(returning.pain_points + memory.pain_points))
                memory.objections = list(dict.fromkeys(returning.objections + memory.objections))
                if not updated_bant.budget and returning.bant.budget:
                    updated_bant.budget = returning.bant.budget
                    updated_bant.budget_score = returning.bant.budget_score
                if not updated_bant.authority and returning.bant.authority:
                    updated_bant.authority = returning.bant.authority
                    updated_bant.authority_score = returning.bant.authority_score
                if not updated_bant.need and returning.bant.need:
                    updated_bant.need = returning.bant.need
                    updated_bant.need_score = returning.bant.need_score
                if not updated_bant.timeline and returning.bant.timeline:
                    updated_bant.timeline = returning.bant.timeline
                    updated_bant.timeline_score = returning.bant.timeline_score

        # Tool 1: Retrieve 3-tier memory snapshot
        memory_snapshot, mem_log = self.tools.execute(
            req.session_id,
            "retrieve_customer_memory",
            {"session_id": req.session_id},
            extra_context={"memory": memory},
        )
        tool_logs.append(mem_log)

        # Tool 2: Hybrid RAG Retrieval
        t_rag = time.perf_counter()
        retrieved_chunks: list[RetrievedChunk]
        retrieved_chunks, rag_log = self.tools.execute(
            req.session_id,
            "retrieve_knowledge",
            {"query": req.text, "top_k": 3},
        )
        tool_logs.append(rag_log)
        rag_ms = int((time.perf_counter() - t_rag) * 1000)

        is_kb_question = self.rag.is_knowledge_query(req.text)
        kb_grounded = True
        if is_kb_question and (not retrieved_chunks or retrieved_chunks[0].score < 0.22):
            # Strict grounding: do not pretend low-relevance chunks answer an out-of-scope factual question
            if retrieved_chunks and retrieved_chunks[0].score < 0.22:
                retrieved_chunks = []
            kb_grounded = False

        # Tool 3: Autonomous Appointment Booking / Rescheduling / Cancellation
        existing_appt = self.db.get_session_appointment(req.session_id)
        booked_appt: Optional[AppointmentRecord] = None
        cancelled_appt = False
        new_webhooks: list[dict[str, Any]] = []
        offered_slots: list[str] = []
        booking_error: Optional[str] = None

        wants_cancel = "cancel_intent" in sentiment.signals and existing_appt is not None
        wants_reschedule = "reschedule_intent" in sentiment.signals and existing_appt is not None
        wants_booking = ("booking_intent" in sentiment.signals or awaiting_slot_choice) and not existing_appt

        if wants_cancel and existing_appt:
            try:
                _, cancel_log = self.tools.execute(
                    req.session_id, "cancel_appointment", {"appt_id": existing_appt.id}
                )
                tool_logs.append(cancel_log)
                cancelled_appt = True
                existing_appt = None
            except Exception as exc:
                booking_error = str(exc)

        elif wants_reschedule and existing_appt:
            available, avail_log = self.tools.execute(
                req.session_id, "check_availability", {"count": 4}
            )
            tool_logs.append(avail_log)
            chosen_slot, val_err = parse_requested_slot(req.text, available, tz_name=req.timezone)
            if val_err:
                booking_error = val_err
                offered_slots = available[:3]
            elif chosen_slot:
                try:
                    (booked_appt, appt_hooks), res_log = self.tools.execute(
                        req.session_id,
                        "reschedule_appointment",
                        {"appt_id": existing_appt.id, "slot_iso": chosen_slot, "timezone": req.timezone},
                    )
                    tool_logs.append(res_log)
                    new_webhooks.extend(appt_hooks)
                except (BookingConflictError, BookingValidationError) as exc:
                    booking_error = str(exc)
                    offered_slots = available[:3]
            else:
                offered_slots = available[:3]
                awaiting_slot_choice = True
                proposed_slots = offered_slots

        elif wants_booking:
            available, avail_log = self.tools.execute(
                req.session_id, "check_availability", {"count": 4}
            )
            tool_logs.append(avail_log)
            chosen_slot, val_err = parse_requested_slot(req.text, proposed_slots or available, tz_name=req.timezone)
            has_specific_time = any(
                k in req.text.lower()
                for k in [
                    "tomorrow", "today", "yesterday", "monday", "tuesday", "wednesday",
                    "thursday", "friday", "saturday", "sunday", "am", "pm", "morning",
                    "afternoon", "slot", "option", "first", "second", "mañana", "demain",
                    "morgen", "कल", "आज", "बजे", "kal", "subah", "shaam",
                ]
            ) or any(c.isdigit() for c in req.text)

            if val_err:
                booking_error = val_err
                offered_slots = available[:3]
                awaiting_slot_choice = True
                proposed_slots = offered_slots
            elif chosen_slot and (awaiting_slot_choice or has_specific_time):
                qualification = compute_qualification(
                    updated_bant, memory, has_booking=False, last_utterance=req.text
                )
                lead_rec, ev_type = self.db.upsert_lead_from_session(
                    req.session_id, memory, qualification, sentiment.polarity, sentiment.intent_score
                )
                memory.lead_id = lead_rec.id

                try:
                    (booked_appt, appt_hooks), book_log = self.tools.execute(
                        req.session_id,
                        "book_appointment",
                        {
                            "session_id": req.session_id,
                            "lead_id": lead_rec.id,
                            "lead_name": lead_rec.name,
                            "lead_email": memory.email,
                            "company": memory.company,
                            "slot_iso": chosen_slot,
                            "timezone": req.timezone,
                            "topic": f"Fortivex Demo — {memory.company or lead_rec.name}",
                        },
                    )
                    tool_logs.append(book_log)
                    new_webhooks.extend(appt_hooks)
                    awaiting_slot_choice = False
                    proposed_slots = []
                except (BookingConflictError, BookingValidationError) as exc:
                    booking_error = str(exc)
                    offered_slots = available[:3]
                    awaiting_slot_choice = True
                    proposed_slots = offered_slots
            else:
                offered_slots = available[:3]
                awaiting_slot_choice = True
                proposed_slots = offered_slots

        # 4. Deterministic BANT qualification + Tool 4: CRM Sync
        qualification = compute_qualification(
            updated_bant,
            memory,
            has_booking=bool(booked_appt or existing_appt),
            last_utterance=req.text,
        )
        lead_rec, event_type = self.db.upsert_lead_from_session(
            req.session_id, memory, qualification, sentiment.polarity, sentiment.intent_score
        )
        memory.lead_id = lead_rec.id

        crm_hooks, crm_log = self.tools.execute(
            req.session_id,
            "sync_crm",
            {"session_id": req.session_id, "lead_id": lead_rec.id, "event_type": event_type},
        )
        tool_logs.append(crm_log)
        new_webhooks.extend(crm_hooks)

        # Persist user message (with optional PII masking)
        self.db.add_message(
            ConversationMessage(
                session_id=req.session_id,
                role="user",
                text=mask_pii(req.text),
                language=lang,
                sentiment=sentiment,
                retrieved_chunks=retrieved_chunks,
            )
        )

        # 5. Generate Grounded Multilingual Response
        t_llm = time.perf_counter()
        history = self.db.get_messages(req.session_id)
        reply_text, engine_used = self._generate_reply(
            user_text=req.text,
            lang=lang,
            memory=memory,
            memory_snapshot=memory_snapshot,
            qualification=qualification,
            sentiment=sentiment,
            chunks=retrieved_chunks,
            kb_grounded=kb_grounded,
            booked_appt=booked_appt,
            cancelled_appt=cancelled_appt,
            booking_error=booking_error,
            offered_slots=offered_slots,
            history=history,
        )
        llm_ms = int((time.perf_counter() - t_llm) * 1000)

        self.db.add_message(
            ConversationMessage(
                session_id=req.session_id,
                role="assistant",
                text=reply_text,
                language=lang,
                retrieved_chunks=retrieved_chunks,
            )
        )
        self.db.save_session_state(
            session_id=req.session_id,
            lead_id=lead_rec.id,
            language=lang,
            memory=memory,
            qualification=qualification,
            awaiting_slot_choice=awaiting_slot_choice,
            proposed_slots=proposed_slots,
        )

        total_ms = int((time.perf_counter() - t0) * 1000)
        return TurnResponse(
            session_id=req.session_id,
            reply_text=reply_text,
            language=lang,
            tts_locale=TTS_LOCALES[lang],
            tts_provider=self.voice.active_tts_name(),
            sentiment=sentiment,
            qualification=qualification,
            memory=memory,
            memory_snapshot=memory_snapshot,
            retrieved_chunks=retrieved_chunks,
            kb_grounded=kb_grounded,
            booked_appointment=booked_appt or existing_appt,
            proposed_slots=offered_slots,
            booking_error=booking_error,
            crm_synced=True,
            crm_mode=self.crm.active_mode(),
            webhook_events=new_webhooks,
            tool_logs=tool_logs,
            ai_engine=engine_used,
            latency_ms=total_ms,
            latency_breakdown={
                "rag_ms": rag_ms,
                "llm_ms": llm_ms,
                "total_ms": total_ms,
            },
        )

    def _generate_reply(
        self,
        user_text: str,
        lang: SupportedLang,
        memory: LeadMemory,
        memory_snapshot: ThreeTierMemorySnapshot,
        qualification: LeadQualification,
        sentiment: SentimentResult,
        chunks: list[RetrievedChunk],
        kb_grounded: bool,
        booked_appt: Optional[AppointmentRecord],
        cancelled_appt: bool,
        booking_error: Optional[str],
        offered_slots: list[str],
        history: list[ConversationMessage],
    ) -> tuple[str, str]:
        api_key = get_runtime_gemini_key()
        if api_key:
            try:
                text = self._generate_with_gemini(
                    api_key=api_key,
                    user_text=user_text,
                    lang=lang,
                    memory=memory,
                    memory_snapshot=memory_snapshot,
                    qualification=qualification,
                    sentiment=sentiment,
                    chunks=chunks,
                    kb_grounded=kb_grounded,
                    booked_appt=booked_appt,
                    cancelled_appt=cancelled_appt,
                    booking_error=booking_error,
                    offered_slots=offered_slots,
                    history=history,
                )
                if text:
                    return text, "gemini"
            except Exception as exc:
                logger.warning("Gemini generation fallback to local-hybrid: %s", exc)

        return (
            self._generate_local_hybrid(
                lang=lang,
                memory=memory,
                qualification=qualification,
                sentiment=sentiment,
                chunks=chunks,
                kb_grounded=kb_grounded,
                booked_appt=booked_appt,
                cancelled_appt=cancelled_appt,
                booking_error=booking_error,
                offered_slots=offered_slots,
                turn_count=len(history),
            ),
            "local-hybrid",
        )

    def _generate_with_gemini(
        self,
        api_key: str,
        user_text: str,
        lang: SupportedLang,
        memory: LeadMemory,
        memory_snapshot: ThreeTierMemorySnapshot,
        qualification: LeadQualification,
        sentiment: SentimentResult,
        chunks: list[RetrievedChunk],
        kb_grounded: bool,
        booked_appt: Optional[AppointmentRecord],
        cancelled_appt: bool,
        booking_error: Optional[str],
        offered_slots: list[str],
        history: list[ConversationMessage],
    ) -> str:
        from google import genai

        client = genai.Client(api_key=api_key)
        rag_ctx = (
            "\n\n".join(f"[Source: {c.title} ({c.source_file})] {c.content}" for c in chunks)
            if chunks
            else "NO MATCHING KNOWLEDGE BASE DOCUMENT FOUND."
        )
        recent_hist = "\n".join(f"{m.role.upper()}: {m.text}" for m in history[-6:])
        prior_ctx = "\n".join(memory_snapshot.previous_interactions[:2]) or "None"

        booking_ctx = ""
        if cancelled_appt:
            booking_ctx = "APPOINTMENT CANCELLED in database. Confirm cancellation politely."
        elif booking_error:
            booking_ctx = (
                f"BOOKING FAILED: {booking_error}. Never claim booking succeeded. "
                f"Offer these available slots instead: {', '.join(_format_slot_human(s, lang) for s in offered_slots)}"
            )
        elif booked_appt:
            booking_ctx = (
                f"CONFIRMED BOOKING IN DB: {booked_appt.id} at {booked_appt.slot_iso}, "
                f"link: {booked_appt.meeting_link}. Confirm accurately."
            )
        elif offered_slots:
            booking_ctx = f"PROPOSE THESE AVAILABLE SLOTS: {', '.join(_format_slot_human(s, lang) for s in offered_slots)}"

        grounding_rule = (
            "GROUNDING RULE: Never invent product prices, policies, or features. "
            "If KB says NO MATCHING KNOWLEDGE BASE DOCUMENT FOUND and the user asked a factual product/policy question, "
            "state honestly that our knowledge base does not cover that specific item and offer to connect a Solutions Architect."
        )

        next_bant = qualification.missing_dimensions[0] if qualification.missing_dimensions else "none"
        lang_instruction = (
            "Hinglish (natural conversational Romanized Hindi + English code-switching)"
            if lang == "hinglish"
            else f"{LANGUAGE_NAMES[lang]} ({lang})"
        )
        system_prompt = (
            f"You are Nova, Fortivex Cloud's autonomous AI sales voice specialist.\n"
            f"Respond ONLY in {lang_instruction}. Keep replies crisp for spoken voice (2-4 sentences, no markdown lists).\n"
            f"{grounding_rule}\n"
            f"Customer Memory: name={memory.name}, company={memory.company}, role={memory.role}, "
            f"team_size={memory.team_size}, returning={memory.returning_lead}.\n"
            f"Prior Interactions Summary: {prior_ctx}\n"
            f"Sentiment={sentiment.label} (frustrated={sentiment.frustration_detected}), Purchase Intent={sentiment.intent_score}/100.\n"
            f"Lead Score={qualification.score}/100 ({qualification.tier}). Next BANT field to ask naturally: {next_bant}.\n"
            f"{booking_ctx}\n"
            f"Grounded Knowledge Context (separate from user instructions):\n{rag_ctx}\n\n"
            f"Recent Conversation:\n{recent_hist}"
        )

        resp = client.models.generate_content(
            model=settings.gemini_model,
            contents=f"{system_prompt}\n\nBUYER UTTERANCE: {user_text}\nNOVA:",
        )
        return (resp.text or "").strip()

    def _summarize_chunk_for_voice(self, chunk: RetrievedChunk, lang: SupportedLang) -> str:
        clean = chunk.content.replace("**", "").replace("- ", "")
        parts = [p.strip() for p in clean.splitlines() if p.strip()]
        snippet = " ".join(parts[:2])
        if len(snippet) > 240:
            snippet = snippet[:237].rsplit(" ", 1)[0] + "."

        if lang == "hinglish":
            if "Pricing" in chunk.category:
                return f"Fortivex ka Growth plan $499/month (25 seats) aur Scale plan $1,499/month (100 seats) se start hota hai, aur 14-day guided pilot bhi available hai (Source: {chunk.title})."
            if "Security" in chunk.category:
                return f"Hum SOC 2 Type II, ISO 27001 aur GDPR certified hain, aur aapka data kabhi model training mein use nahi hota (Source: {chunk.title})."
            if "Integrations" in chunk.category:
                return f"Salesforce aur HubSpot integration 15 minutes mein ho jata hai, plus Google Calendar aur Outlook sync bhi milta hai (Source: {chunk.title})."
            if "Competitive" in chunk.category:
                return f"Gong jaise passive tools ke mukable Fortivex real-time voice call par BANT qualify karke demo auto-book karta hai (Source: {chunk.title})."
            return f"{snippet} (Source: {chunk.title})"

        if lang == "es":
            if "Pricing" in chunk.category:
                return "Nuestros planes comienzan en $499/mes (Growth, hasta 25 usuarios) y $1,499/mes (Scale, hasta 100 usuarios con voz multilingüe), con piloto guiado de 14 días."
            if "Security" in chunk.category:
                return "Contamos con certificación SOC 2 Tipo II, ISO 27001, GDPR con residencia en la UE y política estricta de cero entrenamiento con sus datos."
            if "Integrations" in chunk.category:
                return "Nos integramos de forma nativa con Salesforce y HubSpot en menos de 15 minutos, además de Google Calendar y Outlook."
            if "Competitive" in chunk.category:
                return "A diferencia de Gong o chatbots de reglas, Fortivex ejecuta llamadas de voz autónomas en tiempo real, califica BANT y agenda citas en 5 días."
        elif lang == "fr":
            if "Pricing" in chunk.category:
                return "Nos offres démarrent à 499 $/mois (Growth, 25 utilisateurs) et 1 499 $/mois (Scale, 100 utilisateurs avec voix multilingue), avec un pilote de 14 jours."
            if "Security" in chunk.category:
                return "Fortivex est certifié SOC 2 Type II, ISO 27001, conforme RGPD avec hébergement UE et garantit zéro entraînement sur vos données."
            if "Integrations" in chunk.category:
                return "Nous proposons une intégration native Salesforce et HubSpot en moins de 15 minutes, ainsi que Google Calendar et Outlook."
            if "Competitive" in chunk.category:
                return "Contrairement aux outils passifs comme Gong, Fortivex mène l'appel vocal en temps réel, qualifie le BANT et réserve la démo automatiquement."
        elif lang == "de":
            if "Pricing" in chunk.category:
                return "Unsere Tarife starten bei 499 $/Monat (Growth, bis 25 Nutzer) und 1.499 $/Monat (Scale, bis 100 Nutzer mit mehrsprachiger KI-Stimme), inklusive 14-tägigem Pilotprojekt."
            if "Security" in chunk.category:
                return "Fortivex ist SOC 2 Type II und ISO 27001 zertifiziert, DSGVO-konform mit EU-Hosting und garantiert Zero-Data-Training."
            if "Integrations" in chunk.category:
                return "Wir bieten native Integrationen für Salesforce und HubSpot in unter 15 Minuten sowie Google Calendar und Outlook."
            if "Competitive" in chunk.category:
                return "Im Gegensatz zu passiven Tools wie Gong führt Fortivex aktive Echtzeit-Sprachgespräche, bewertet BANT live und bucht Termine automatisch."
        elif lang == "hi":
            if "Pricing" in chunk.category:
                return "हमारा Growth प्लान $499/माह (25 सीटें) और Scale प्लान $1,499/माह (100 सीटें और मल्टीलिंगुअल वॉयस) से शुरू होता है, साथ ही 14-दिन का गाइडेड पायलट उपलब्ध है।"
            if "Security" in chunk.category:
                return "Fortivex SOC 2 Type II, ISO 27001 और GDPR प्रमाणित है, और हम आपके डेटा का उपयोग कभी भी मॉडल ट्रेनिंग के लिए नहीं करते।"
            if "Integrations" in chunk.category:
                return "हम Salesforce और HubSpot के साथ 15 मिनट में नेटिव इंटीग्रेशन देते हैं, साथ ही Google Calendar और Outlook भी जुड़ते हैं।"
            if "Competitive" in chunk.category:
                return "Gong जैसे पैसिव टूल के विपरीत, Fortivex रियल-टाइम वॉयस कॉल पर लाइव BANT क्वालिफिकेशन और ऑटोमैटिक डेमो बुकिंग करता है।"

        return f"{snippet} (Source: {chunk.title})"

    def _generate_local_hybrid(
        self,
        lang: SupportedLang,
        memory: LeadMemory,
        qualification: LeadQualification,
        sentiment: SentimentResult,
        chunks: list[RetrievedChunk],
        kb_grounded: bool,
        booked_appt: Optional[AppointmentRecord],
        cancelled_appt: bool,
        booking_error: Optional[str],
        offered_slots: list[str],
        turn_count: int,
    ) -> str:
        first_name = memory.name.split()[0] if memory.name else None
        greeting_name = f" {first_name}" if first_name else ""

        # Handle Disqualified politely
        if qualification.tier == "Disqualified":
            disq = {
                "en": "Understood—I won't take up any more of your time today. I've updated your preference in our system, and feel free to reach out if priorities ever change.",
                "hinglish": "Samajh gaya—main aaj aapka aur samay nahi lunga. Maine aapka status update kar diya hai; jab bhi zaroorat ho humein ping karein.",
                "hi": "समझ गया—मैं आज आपका और समय नहीं लूँगा। मैंने सिस्टम में आपका स्टेटस अपडेट कर दिया है।",
                "es": "Entendido, no te quito más tiempo hoy. He actualizado tu preferencia en nuestro sistema.",
                "fr": "Bien compris, je ne vous retiens pas davantage aujourd'hui. J'ai mis à jour votre dossier.",
                "de": "Verstanden, ich möchte Ihre Zeit heute nicht weiter beanspruchen. Ich habe Ihren Status entsprechend aktualisiert.",
            }
            return disq[lang]

        if cancelled_appt:
            canc = {
                "en": f"Your demo appointment has been cancelled{greeting_name}. Let me know if you'd like to pick a new time later.",
                "hinglish": f"Aapka demo appointment cancel kar diya gaya hai{greeting_name}. Jab bhi naya slot chahiye ho bataiyega.",
                "hi": f"आपकी डेमो मीटिंग रद्द कर दी गई है{greeting_name}।",
                "es": f"Tu cita de demostración ha sido cancelada{greeting_name}.",
                "fr": f"Votre rendez-vous de démonstration a été annulé{greeting_name}.",
                "de": f"Ihr Demo-Termin wurde storniert{greeting_name}.",
            }
            return canc[lang]

        if booking_error:
            slot_opts = ", ".join(
                f"Option {i}: {_format_slot_human(s, lang)}" for i, s in enumerate(offered_slots, 1)
            )
            return (
                f"I couldn't book that exact time because {booking_error} "
                f"{'Here are open business-hour slots instead: ' + slot_opts + '. Which one works?' if slot_opts else ''}"
            )

        if booked_appt:
            slot_txt = _format_slot_human(booked_appt.slot_iso, lang)
            if lang == "hinglish":
                return (
                    f"Perfect{greeting_name}! Maine aapka Fortivex demo {slot_txt} ke liye confirm book kar diya hai "
                    f"(Confirmation ID: {booked_appt.id}) aur CRM sync ho gaya hai. Meeting link: {booked_appt.meeting_link}."
                )
            if lang == "es":
                return (
                    f"Perfecto{greeting_name}, he confirmado tu demostración para el {slot_txt} "
                    f"(ID: {booked_appt.id}) y actualicé el CRM. Enlace: {booked_appt.meeting_link}."
                )
            if lang == "fr":
                return (
                    f"Parfait{greeting_name}, votre démonstration est confirmée pour le {slot_txt} "
                    f"(ID : {booked_appt.id}). Voici le lien : {booked_appt.meeting_link}."
                )
            if lang == "de":
                return (
                    f"Perfekt{greeting_name}, Ihre Fortivex-Demo ist für {slot_txt} bestätigt "
                    f"(ID: {booked_appt.id}). Meeting-Link: {booked_appt.meeting_link}."
                )
            if lang == "hi":
                return (
                    f"बढ़िया{greeting_name}, मैंने आपका Fortivex डेमो {slot_txt} के लिए कन्फर्म कर दिया है "
                    f"(ID: {booked_appt.id}) और CRM अपडेट कर दिया है। मीटिंग लिंक {booked_appt.meeting_link} है।"
                )
            return (
                f"You're all set{greeting_name}! I've confirmed your Fortivex demo for {slot_txt} "
                f"(Confirmation ID: {booked_appt.id}) and updated our CRM. Your meeting link is {booked_appt.meeting_link}."
            )

        if offered_slots:
            formatted = ", ".join(
                f"Option {i}: {_format_slot_human(s, lang)}" for i, s in enumerate(offered_slots, 1)
            )
            if lang == "hinglish":
                return f"Bilkul{greeting_name}, mere paas yeh available slots hain: {formatted}. Kaunsa slot book karein?"
            if lang == "es":
                return f"Con mucho gusto{greeting_name}. Tengo disponibilidad en: {formatted}. ¿Cuál prefieres?"
            if lang == "fr":
                return f"Avec plaisir{greeting_name}. Voici nos créneaux disponibles : {formatted}. Lequel préférez-vous ?"
            if lang == "de":
                return f"Sehr gerne{greeting_name}. Folgende Demo-Termine sind frei: {formatted}. Welcher passt Ihnen?"
            if lang == "hi":
                return f"ज़रूर{greeting_name}, ये समय उपलब्ध हैं: {formatted}। आप कौन-सा स्लॉट बुक करना चाहेंगे?"
            return f"I'd be happy to schedule that{greeting_name}. I have openings at {formatted}. Which slot works best?"

        # Strict Unknown-Answer Guardrail
        if not kb_grounded:
            unknown = {
                "en": f"I checked our Fortivex knowledge base{greeting_name}, and we don't have documented details on that specific topic—I want to avoid guessing. I can connect you with a Solutions Architect or answer questions about Fortivex pricing, SOC2 security, and CRM integrations.",
                "hinglish": f"Maine Fortivex knowledge base check kiya{greeting_name}, par is specific topic par humare paas documented details nahi hain—main galat estimate nahi dunga. Kya main aapko pricing, SOC2 security, ya CRM integration ke baare mein bataun?",
                "hi": f"मैंने Fortivex नॉलेज बेस चेक किया{greeting_name}, लेकिन इस विषय पर हमारे पास दस्तावेज़ उपलब्ध नहीं हैं। मैं गलत जानकारी नहीं दूँगा—क्या मैं आपको प्राइसिंग, सुरक्षा या CRM इंटीग्रेशन के बारे में बताऊँ?",
                "es": f"Revisé nuestra base de conocimientos{greeting_name} y no tengo información documentada sobre ese punto específico. Prefiero no improvisar; puedo ayudarte con precios, seguridad SOC2 o integraciones CRM.",
                "fr": f"J'ai vérifié notre base de connaissances{greeting_name}, mais ce sujet spécifique n'y figure pas. Je préfère ne pas inventer de réponse ; je peux en revanche vous renseigner sur nos tarifs, la sécurité SOC2 ou nos intégrations CRM.",
                "de": f"Ich habe unsere Wissensdatenbank geprüft{greeting_name}, dazu liegen mir jedoch keine dokumentierten Angaben vor. Ich möchte nichts erfinden—gern helfe ich Ihnen bei Preisen, SOC2-Sicherheit oder CRM-Integrationen weiter.",
            }
            return unknown[lang]

        parts: list[str] = []
        if memory.returning_lead and turn_count <= 2:
            if lang == "hinglish":
                parts.append(f"Wapas swagat hai{greeting_name}{f' ({memory.company})' if memory.company else ''}—pichli baatcheet ke details mere paas saved hain.")
            elif lang == "es":
                parts.append(f"Qué gusto saludarte de nuevo{greeting_name}{f' de {memory.company}' if memory.company else ''}.")
            elif lang == "fr":
                parts.append(f"Ravi de vous retrouver{greeting_name}{f' chez {memory.company}' if memory.company else ''}.")
            elif lang == "de":
                parts.append(f"Schön, Sie wieder begrüßen zu dürfen{greeting_name}.")
            elif lang == "hi":
                parts.append(f"दोबारा जुड़ने के लिए धन्यवाद{greeting_name}।")
            else:
                parts.append(f"Welcome back{greeting_name}{f' from {memory.company}' if memory.company else ''}—I still have your context from our earlier call.")

        if sentiment.frustration_detected or "price_objection" in sentiment.signals:
            emp = {
                "en": "I hear your concern—most teams achieve full payback in under 3 months by saving 6.5 hours per rep weekly, and we offer a 14-day guided pilot.",
                "hinglish": "Main aapki concern samajhta hoon—zyadatar teams har rep ke 6.5 hours bachakar 3 months se kam mein ROI nikal leti hain, aur 14-day pilot bhi milta hai.",
                "es": "Entiendo perfectamente tu preocupación; nuestros clientes recuperan la inversión en menos de 3 meses ahorrando 6.5 horas por vendedor a la semana.",
                "fr": "Je comprends votre préoccupation ; nos clients rentabilisent la solution en moins de 3 mois grâce à un gain de 6,5 heures par commercial.",
                "de": "Ich verstehe Ihren Punkt gut—durch 6,5 Stunden Zeitersparnis pro Mitarbeiter amortisiert sich Fortivex meist in unter 3 Monaten.",
                "hi": "मैं आपकी चिंता समझता हूँ—हमारे ग्राहक हर हफ्ते प्रति प्रतिनिधि 6.5 घंटे बचाकर 3 महीने से कम में ROI प्राप्त करते हैं।",
            }
            parts.append(emp[lang])
        elif chunks and chunks[0].score >= 0.14:
            parts.append(self._summarize_chunk_for_voice(chunks[0], lang))
        else:
            intro = {
                "en": f"Thanks for reaching out{greeting_name}! Fortivex automates real-time multilingual sales calls, BANT qualification, and CRM sync.",
                "hinglish": f"Namaste{greeting_name}! Fortivex real-time multilingual AI voice calls, BANT qualification, aur CRM sync automate karta hai.",
                "es": f"¡Gracias por comunicarte{greeting_name}! Fortivex automatiza llamadas de ventas multilingües, calificación BANT y CRM.",
                "fr": f"Merci de nous contacter{greeting_name} ! Fortivex automatise vos appels commerciaux multilingues, le BANT et le CRM.",
                "de": f"Vielen Dank für Ihre Anfrage{greeting_name}! Fortivex automatisiert mehrsprachige Verkaufsgespräche und CRM-Sync.",
                "hi": f"संपर्क करने के लिए धन्यवाद{greeting_name}! Fortivex रियल-टाइम मल्टीलिंगुअल सेल्स कॉल, BANT क्वालिफिकेशन और CRM सिंक को ऑटोमेट करता है।",
            }
            parts.append(intro[lang])

        missing = qualification.missing_dimensions
        if not missing or qualification.score >= 70:
            cta = {
                "en": "Since you're a strong fit, would you like me to book a 30-minute live architecture demo tomorrow?",
                "hinglish": "Aapki team ke liye yeh great fit hai—kya main kal ke liye 30-minute ka live demo book kar doon?",
                "es": "Dado el gran ajuste para tu equipo, ¿te gustaría agendar una demo en vivo de 30 minutos mañana?",
                "fr": "Votre profil correspond parfaitement—souhaitez-vous réserver une démo personnalisée demain ?",
                "de": "Das passt hervorragend—soll ich Ihnen eine 30-minütige Live-Demo für morgen reservieren?",
                "hi": "आपकी टीम के लिए यह बिल्कुल उपयुक्त है—क्या मैं कल के लिए 30 मिनट का लाइव डेमो बुक कर दूँ?",
            }
            parts.append(cta[lang])
        else:
            next_dim = missing[0]
            prompts: dict[str, dict[SupportedLang, str]] = {
                "need": {
                    "en": "What is the biggest sales bottleneck your team is trying to solve right now?",
                    "hinglish": "Abhi aapki sales team sabse badi kis challenge ko solve karna chahti hai?",
                    "es": "¿Cuál es el principal reto de ventas que buscan resolver en su equipo?",
                    "fr": "Quel est le principal défi commercial que votre équipe souhaite résoudre ?",
                    "de": "Was ist aktuell die größte Herausforderung in Ihrem Vertriebsprozess?",
                    "hi": "अभी आपकी सेल्स टीम किस सबसे बड़ी चुनौती को हल करना चाहती है?",
                },
                "timeline": {
                    "en": "How soon are you aiming to roll this out, and roughly how many reps are on your team?",
                    "hinglish": "Aap yeh solution kab tak launch karna chahte hain aur team mein kitne reps hain?",
                    "es": "¿En qué plazo planean implementar una solución así y cuántos vendedores tienen?",
                    "fr": "Dans quel délai souhaitez-vous déployer cette solution et combien de commerciaux avez-vous ?",
                    "de": "Bis wann möchten Sie starten und wie groß ist Ihr Vertriebsteam?",
                    "hi": "आप इसे कब तक लागू करना चाहते हैं और आपकी टीम में कितने सेल्स प्रतिनिधि हैं?",
                },
                "authority": {
                    "en": "What is your role in the evaluation, and who else joins the final decision?",
                    "hinglish": "Is evaluation mein aapka role kya hai aur final decision kaun leta hai?",
                    "es": "¿Cuál es tu rol en la evaluación y quién más participa en la decisión?",
                    "fr": "Quel est votre rôle dans ce projet et qui participe à la décision ?",
                    "de": "Welche Rolle haben Sie bei der Evaluierung und wer entscheidet mit?",
                    "hi": "इस मूल्यांकन में आपकी क्या भूमिका है और अंतिम निर्णय में कौन-कौन शामिल होंगे?",
                },
                "budget": {
                    "en": "Do you already have a target budget range allocated for sales automation this quarter?",
                    "hinglish": "Kya aapne is quarter ke liye sales automation ka budget allocate kiya hai?",
                    "es": "¿Cuentan con un rango de presupuesto asignado para automatización comercial?",
                    "fr": "Avez-vous déjà une enveloppe budgétaire dédiée à ce trimestre ?",
                    "de": "Haben Sie bereits ein Budget für Sales-Automatisierung eingeplant?",
                    "hi": "क्या आपने इस तिमाही के लिए कोई बजट तय किया है?",
                },
            }
            parts.append(prompts[next_dim][lang])

        return " ".join(parts)

    def generate_call_summary(self, session_id: str) -> CallSummary:
        """Compile, persist, and CRM-sync a comprehensive post-call report."""
        sess = self.db.get_or_create_session(session_id)
        memory: LeadMemory = sess["memory"]
        qual: LeadQualification = sess["qualification"]
        lang: SupportedLang = sess["language"]
        messages = self.db.get_messages(session_id)
        appt = self.db.get_session_appointment(session_id)

        user_msgs = [m for m in messages if m.role == "user"]
        sentiments = [m.sentiment.polarity for m in user_msgs if m.sentiment is not None]
        avg_sent = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0.0

        if len(sentiments) >= 2:
            if sentiments[-1] - sentiments[0] >= 0.2:
                trend = "Improving"
            elif sentiments[-1] - sentiments[0] <= -0.2:
                trend = "Declining"
            else:
                trend = "Stable"
        else:
            trend = "Stable"

        kb_sources = list(
            dict.fromkeys(
                (c.get("title", "") if isinstance(c, dict) else c.title)
                for m in user_msgs
                for c in m.retrieved_chunks
                if (c.get("title") if isinstance(c, dict) else c.title)
            )
        )

        lead_name = memory.name or f"Prospect ({session_id[-6:]})"
        duration_sec = max(30, len(messages) * 24)
        follow_up_dt = (utc_now() + timedelta(days=2)).strftime("%Y-%m-%d")

        if appt:
            appt_outcome = f"{appt.status} for {appt.slot_iso} ({appt.id})"
            follow_up_dt = appt.slot_iso[:10]
        elif qual.tier == "Disqualified":
            appt_outcome = "Disqualified — no follow-up scheduled"
        else:
            appt_outcome = "No demo booked yet"

        next_steps: list[str] = []
        if appt:
            next_steps.append(f"Host {appt.status.lower()} demo at {appt.slot_iso} ({appt.meeting_link})")
        elif qual.tier == "Disqualified":
            next_steps.append("Mark lead Disqualified in CRM and respect opt-out preference")
        elif qual.score >= 60:
            next_steps.append("Send 14-day guided pilot & architecture demo booking link")
        else:
            next_steps.append("Enroll in multilingual nurture sequence and qualify remaining BANT criteria")

        if memory.objections:
            next_steps.append(f"Address objections in follow-up: {', '.join(memory.objections)}")
        if memory.email:
            next_steps.append(f"Send call recap & SOC2/integration brief to {memory.email}")

        exec_summary = (
            f"{lead_name}{f' ({memory.role})' if memory.role else ''}"
            f"{f' from {memory.company}' if memory.company else ''} engaged in a "
            f"{LANGUAGE_NAMES[lang]} conversation across {len(user_msgs)} buyer turns. "
            f"Qualification scored {qual.score}/100 ({qual.tier}, stage: {qual.stage}) with estimated ACV ${qual.estimated_acv_usd:,}. "
            f"{'Pain points: ' + ', '.join(memory.pain_points) + '. ' if memory.pain_points else ''}"
            f"{'Appointment outcome: ' + appt_outcome + '.'}"
        )

        api_key = get_runtime_gemini_key()
        if api_key and user_msgs:
            try:
                from google import genai

                client = genai.Client(api_key=api_key)
                transcript = "\n".join(f"{m.role}: {m.text}" for m in messages)
                resp = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=(
                        "Write a concise 3-sentence executive sales call summary covering customer requirements, "
                        f"BANT qualification ({qual.score}/100, {qual.tier}), objections, and next steps:\n\n{transcript}"
                    ),
                )
                if resp.text and resp.text.strip():
                    exec_summary = resp.text.strip()
            except Exception as exc:
                logger.warning("Gemini summary fallback: %s", exc)

        summary = CallSummary(
            session_id=session_id,
            lead_id=memory.lead_id,
            lead_name=lead_name,
            customer_email=memory.email,
            customer_phone=memory.phone,
            company=memory.company,
            language=lang,
            turn_count=len(messages),
            duration_estimate_sec=duration_sec,
            executive_summary=exec_summary,
            key_pain_points=memory.pain_points,
            buying_signals=memory.buying_signals,
            objections_raised=memory.objections,
            competitor_mentions=memory.competitors_mentioned,
            knowledge_sources_used=kb_sources,
            qualification=qual,
            average_sentiment=avg_sent,
            sentiment_trend=trend,  # type: ignore[arg-type]
            appointment_outcome=appt_outcome,
            next_steps=next_steps,
            follow_up_date=follow_up_dt,
            booked_appointment=appt,
            generated_at=utc_now_iso(),
        )
        self.db.save_call_summary(summary)
        self.crm.sync_call_summary(summary)
        return summary
