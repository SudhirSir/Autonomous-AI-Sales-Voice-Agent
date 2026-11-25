"""Autonomous AI Sales Voice Agent Orchestrator (RAG + Memory + BANT + Booking + LLM/Hybrid)."""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, Optional

from app.config import get_runtime_gemini_key, settings
from app.database import DatabaseStore
from app.models import (
    AppointmentCreate,
    AppointmentRecord,
    CallSummary,
    ConversationMessage,
    LeadMemory,
    LeadQualification,
    RetrievedChunk,
    SentimentResult,
    SupportedLang,
    TurnRequest,
    TurnResponse,
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

logger = logging.getLogger(__name__)


def _format_slot_human(slot_iso: str, lang: SupportedLang) -> str:
    dt = datetime.fromisoformat(slot_iso.replace("Z", ""))
    day_str = dt.strftime("%A, %b %d at %I:%M %p UTC")
    if lang == "es":
        return dt.strftime("%d/%m a las %H:%M UTC")
    if lang == "fr":
        return dt.strftime("%d/%m à %H:%M UTC")
    if lang == "de":
        return dt.strftime("%d.%m. um %H:%M UTC")
    if lang == "hi":
        return dt.strftime("%d %b को %I:%M %p UTC")
    return day_str


class SalesVoiceAgent:
    """Coordinates real-time multilingual sales conversations, RAG, CRM sync, and booking."""

    def __init__(self, db: DatabaseStore, rag: HybridRAGRetriever) -> None:
        self.db = db
        self.rag = rag
        # Reload any persisted custom knowledge docs into RAG index
        for doc in self.db.list_custom_docs():
            self.rag.add_document(doc["title"], doc["category"], doc["content"])

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

        # 1. Sentiment & Intent analysis
        sentiment = analyze_sentiment(req.text)

        # 2. Contextual Memory & BANT extraction
        memory, updated_bant = extract_memory_and_bant(req.text, memory, qualification.bant)

        # 3. Cross-call memory enrichment if returning lead matches email or name+company
        if not memory.lead_id and (memory.email or (memory.name and memory.company)):
            returning = self.db.find_returning_lead(memory.email, memory.name, memory.company)
            if returning:
                memory.lead_id = returning.id
                memory.returning_lead = True
                memory.prior_calls_count += 1
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

        # 4. RAG Knowledge Retrieval
        retrieved_chunks = self.rag.retrieve(req.text, top_k=3)

        # 5. Automated Appointment Booking & Slot Negotiation
        existing_appt = self.db.get_session_appointment(req.session_id)
        booked_appt: Optional[AppointmentRecord] = None
        new_webhooks: list[dict[str, Any]] = []
        offered_slots: list[str] = []

        wants_booking = "booking_intent" in sentiment.signals or awaiting_slot_choice
        if wants_booking and not existing_appt:
            available = proposed_slots or self.db.get_available_slots(count=3)
            chosen_slot = parse_requested_slot(req.text, available)
            has_specific_time = any(
                k in req.text.lower()
                for k in [
                    "tomorrow", "today", "monday", "tuesday", "wednesday", "thursday", "friday",
                    "am", "pm", "morning", "afternoon", "slot", "option", "first", "second",
                    "mañana", "demain", "morgen", "कल", "आज", "बजे",
                ]
            ) or any(c.isdigit() for c in req.text)

            if chosen_slot and (awaiting_slot_choice or has_specific_time):
                # Recompute qualification first so lead exists before booking
                qualification = compute_qualification(updated_bant, memory, has_booking=True)
                lead_rec, crm_hooks = self.db.upsert_lead_from_session(
                    req.session_id, memory, qualification, sentiment.polarity
                )
                memory.lead_id = lead_rec.id
                new_webhooks.extend(crm_hooks)

                booked_appt, appt_hook = self.db.book_appointment(
                    AppointmentCreate(
                        session_id=req.session_id,
                        lead_id=lead_rec.id,
                        lead_name=lead_rec.name,
                        lead_email=memory.email,
                        company=memory.company,
                        slot_iso=chosen_slot,
                        topic=f"Fortivex Demo — {memory.company or lead_rec.name}",
                    )
                )
                new_webhooks.append(appt_hook)
                awaiting_slot_choice = False
                proposed_slots = []
            else:
                offered_slots = available[:3]
                awaiting_slot_choice = True
                proposed_slots = offered_slots

        # 6. Compute final Lead Qualification & Sync CRM
        qualification = compute_qualification(
            updated_bant, memory, has_booking=bool(booked_appt or existing_appt)
        )
        lead_rec, crm_hooks = self.db.upsert_lead_from_session(
            req.session_id, memory, qualification, sentiment.polarity
        )
        memory.lead_id = lead_rec.id
        new_webhooks.extend(crm_hooks)

        # Store user message
        self.db.add_message(
            ConversationMessage(
                session_id=req.session_id,
                role="user",
                text=req.text,
                language=lang,
                sentiment=sentiment,
                retrieved_chunks=retrieved_chunks,
            )
        )

        # 7. Generate natural voice response (Gemini if key set, else local-hybrid)
        history = self.db.get_messages(req.session_id)
        reply_text, engine_used = self._generate_reply(
            user_text=req.text,
            lang=lang,
            memory=memory,
            qualification=qualification,
            sentiment=sentiment,
            chunks=retrieved_chunks,
            booked_appt=booked_appt,
            offered_slots=offered_slots,
            history=history,
        )

        # Store assistant message & persist session state
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

        latency_ms = int((time.perf_counter() - t0) * 1000)
        return TurnResponse(
            session_id=req.session_id,
            reply_text=reply_text,
            language=lang,
            tts_locale=TTS_LOCALES[lang],
            sentiment=sentiment,
            qualification=qualification,
            memory=memory,
            retrieved_chunks=retrieved_chunks,
            booked_appointment=booked_appt or existing_appt,
            proposed_slots=offered_slots,
            crm_synced=True,
            webhook_events=new_webhooks,
            ai_engine=engine_used,
            latency_ms=latency_ms,
        )

    def _generate_reply(
        self,
        user_text: str,
        lang: SupportedLang,
        memory: LeadMemory,
        qualification: LeadQualification,
        sentiment: SentimentResult,
        chunks: list[RetrievedChunk],
        booked_appt: Optional[AppointmentRecord],
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
                    qualification=qualification,
                    sentiment=sentiment,
                    chunks=chunks,
                    booked_appt=booked_appt,
                    offered_slots=offered_slots,
                    history=history,
                )
                if text:
                    return text, "gemini"
            except Exception as exc:
                logger.warning("Gemini fallback to local-hybrid engine: %s", exc)

        return (
            self._generate_local_hybrid(
                user_text=user_text,
                lang=lang,
                memory=memory,
                qualification=qualification,
                sentiment=sentiment,
                chunks=chunks,
                booked_appt=booked_appt,
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
        qualification: LeadQualification,
        sentiment: SentimentResult,
        chunks: list[RetrievedChunk],
        booked_appt: Optional[AppointmentRecord],
        offered_slots: list[str],
        history: list[ConversationMessage],
    ) -> str:
        from google import genai  # lazy import

        client = genai.Client(api_key=api_key)
        rag_ctx = "\n\n".join(f"[{c.title}] {c.content}" for c in chunks) or "No specific KB match; use general Fortivex capabilities."
        recent_hist = "\n".join(f"{m.role.upper()}: {m.text}" for m in history[-6:])

        booking_ctx = ""
        if booked_appt:
            booking_ctx = f"CONFIRMED BOOKING: {booked_appt.slot_iso}, meeting link: {booked_appt.meeting_link}. Confirm this warmly."
        elif offered_slots:
            booking_ctx = f"OFFER THESE AVAILABLE DEMO SLOTS: {', '.join(_format_slot_human(s, lang) for s in offered_slots)}"

        next_bant = qualification.missing_dimensions[0] if qualification.missing_dimensions else "none (ready to propose demo)"
        system_prompt = (
            f"You are Nova, an elite autonomous AI sales voice representative for Fortivex Cloud.\n"
            f"Respond ONLY in {LANGUAGE_NAMES[lang]} ({lang}). Keep your response natural for spoken voice (2 to 4 concise sentences, no markdown bullets).\n"
            f"Lead Memory: name={memory.name}, company={memory.company}, role={memory.role}, team_size={memory.team_size}, "
            f"returning={memory.returning_lead}, pain_points={memory.pain_points}.\n"
            f"Sentiment: {sentiment.label} (signals={sentiment.signals}). Lead Score: {qualification.score}/100 ({qualification.tier}).\n"
            f"Next BANT dimension to uncover naturally (if not booking): {next_bant}.\n"
            f"{booking_ctx}\n"
            f"Grounded Knowledge Base Context:\n{rag_ctx}\n\n"
            f"Recent Transcript:\n{recent_hist}"
        )

        resp = client.models.generate_content(
            model=settings.gemini_model,
            contents=f"{system_prompt}\n\nUSER SAYS: {user_text}\nNOVA:",
        )
        return (resp.text or "").strip()

    def _summarize_chunk_for_voice(self, chunk: RetrievedChunk, lang: SupportedLang) -> str:
        """Condense the top RAG chunk into clean spoken facts."""
        # Extract up to 2 crisp sentences/bullets without markdown syntax
        clean = chunk.content.replace("**", "").replace("- ", "")
        parts = [p.strip() for p in clean.splitlines() if p.strip()]
        snippet = " ".join(parts[:2])
        if len(snippet) > 240:
            snippet = snippet[:237].rsplit(" ", 1)[0] + "."

        if lang == "es":
            if "Pricing" in chunk.category:
                return "Nuestros planes comienzan en $499/mes (Growth, hasta 25 usuarios) y $1,499/mes (Scale, hasta 100 usuarios con voz multilingüe), además de piloto guiado de 14 días y 18% de descuento anual."
            if "Security" in chunk.category:
                return "Contamos con certificación SOC 2 Tipo II, ISO 27001, GDPR con residencia en la UE, HIPAA bajo demanda y política estricta de cero entrenamiento con sus datos."
            if "Integrations" in chunk.category:
                return "Nos integramos de forma nativa con Salesforce y HubSpot en menos de 15 minutos, además de Google Calendar, Outlook y Twilio."
            if "Competitive" in chunk.category:
                return "A diferencia de herramientas pasivas como Gong o chatbots de reglas, Fortivex ejecuta llamadas de voz autónomas en tiempo real, califica BANT y agenda citas en 5 días."
        elif lang == "fr":
            if "Pricing" in chunk.category:
                return "Nos offres démarrent à 499 $/mois (Growth, jusqu'à 25 utilisateurs) et 1 499 $/mois (Scale, jusqu'à 100 utilisateurs avec voix multilingue), avec un pilote de 14 jours."
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
                return "Fortivex ist SOC 2 Type II und ISO 27001 zertifiziert, DSGVO-konform mit EU-Hosting in Frankfurt und garantiert Zero-Data-Training."
            if "Integrations" in chunk.category:
                return "Wir bieten native Integrationen für Salesforce und HubSpot in unter 15 Minuten sowie Google Calendar und Outlook."
            if "Competitive" in chunk.category:
                return "Im Gegensatz zu passiven Tools wie Gong führt Fortivex aktive Echtzeit-Sprachgespräche, bewertet BANT live und bucht Termine automatisch."
        elif lang == "hi":
            if "Pricing" in chunk.category:
                return "हमारा Growth प्लान $499/माह (25 सीटें) और Scale प्लान $1,499/माह (100 सीटें और मल्टीलिंगुअल वॉयस) से शुरू होता है, साथ ही 14-दिन का गाइडेड पायलट भी उपलब्ध है।"
            if "Security" in chunk.category:
                return "Fortivex SOC 2 Type II, ISO 27001 और GDPR प्रमाणित है, और हम आपके डेटा का उपयोग कभी भी मॉडल ट्रेनिंग के लिए नहीं करते।"
            if "Integrations" in chunk.category:
                return "हम Salesforce और HubSpot के साथ 15 मिनट में नेटिव इंटीग्रेशन देते हैं, साथ ही Google Calendar और Outlook भी जुड़ते हैं।"
            if "Competitive" in chunk.category:
                return "Gong जैसे पैसिव टूल के विपरीत, Fortivex रियल-टाइम वॉयस कॉल पर लाइव BANT क्वालिफिकेशन और ऑटोमैटिक डेमो बुकिंग करता है।"

        return snippet

    def _generate_local_hybrid(
        self,
        user_text: str,
        lang: SupportedLang,
        memory: LeadMemory,
        qualification: LeadQualification,
        sentiment: SentimentResult,
        chunks: list[RetrievedChunk],
        booked_appt: Optional[AppointmentRecord],
        offered_slots: list[str],
        turn_count: int,
    ) -> str:
        first_name = memory.name.split()[0] if memory.name else None
        greeting_name = f" {first_name}" if first_name else ""

        # 1. Booked appointment confirmation takes top priority
        if booked_appt:
            slot_txt = _format_slot_human(booked_appt.slot_iso, lang)
            if lang == "es":
                return (
                    f"Perfecto{greeting_name}, he confirmado tu demostración de Fortivex para el {slot_txt} "
                    f"y actualicé tu ficha en el CRM. El enlace de la reunión es {booked_appt.meeting_link}. ¿Deseas revisar algún requisito técnico antes de la sesión?"
                )
            if lang == "fr":
                return (
                    f"Parfait{greeting_name}, votre démonstration Fortivex est confirmée pour le {slot_txt} "
                    f"et votre fiche CRM est synchronisée. Voici le lien : {booked_appt.meeting_link}."
                )
            if lang == "de":
                return (
                    f"Perfekt{greeting_name}, ich habe Ihre Fortivex-Demo für {slot_txt} bestätigt "
                    f"und direkt im CRM eingetragen. Ihr Meeting-Link lautet {booked_appt.meeting_link}."
                )
            if lang == "hi":
                return (
                    f"बढ़िया{greeting_name}, मैंने आपका Fortivex डेमो {slot_txt} के लिए कन्फर्म कर दिया है "
                    f"और CRM में अपडेट कर दिया है। मीटिंग लिंक {booked_appt.meeting_link} है।"
                )
            return (
                f"You're all set{greeting_name}! I've booked your Fortivex demo for {slot_txt} "
                f"and synced your profile to our CRM. Your meeting link is {booked_appt.meeting_link}. "
                f"Is there anything specific you'd like our Solutions Architect to prepare?"
            )

        # 2. Offering calendar slots
        if offered_slots:
            formatted = ", ".join(
                f"Option {i}: {_format_slot_human(s, lang)}" for i, s in enumerate(offered_slots, 1)
            )
            if lang == "es":
                return f"Con mucho gusto{greeting_name}. Tengo disponibilidad en estos horarios: {formatted}. ¿Cuál prefieres?"
            if lang == "fr":
                return f"Avec plaisir{greeting_name}. Voici nos prochains créneaux disponibles : {formatted}. Lequel vous convient le mieux ?"
            if lang == "de":
                return f"Sehr gerne{greeting_name}. Folgende Demo-Termine sind frei: {formatted}. Welcher passt Ihnen am besten?"
            if lang == "hi":
                return f"ज़रूर{greeting_name}, मेरे पास ये समय उपलब्ध हैं: {formatted}। आप इनमें से कौन-सा स्लॉट बुक करना चाहेंगे?"
            return f"I'd be happy to schedule that{greeting_name}. I have openings at {formatted}. Which slot works best for you?"

        # 3. Greeting on first turn if no specific question
        parts: list[str] = []
        if memory.returning_lead and turn_count <= 2:
            if lang == "es":
                parts.append(f"Qué gusto saludarte de nuevo{greeting_name}{f' de {memory.company}' if memory.company else ''}.")
            elif lang == "fr":
                parts.append(f"Ravi de vous retrouver{greeting_name}{f' chez {memory.company}' if memory.company else ''}.")
            elif lang == "de":
                parts.append(f"Schön, Sie wieder begrüßen zu dürfen{greeting_name}.")
            elif lang == "hi":
                parts.append(f"दोबारा जुड़ने के लिए धन्यवाद{greeting_name}।")
            else:
                parts.append(f"Welcome back{greeting_name}{f' from {memory.company}' if memory.company else ''}—great to speak with you again.")

        # 4. Empathetic handling for negative/skeptical/objection sentiment
        if "price_objection" in sentiment.signals:
            emp = {
                "en": "I completely understand watching software spend closely—most teams see full payback in under 3 months by saving 6.5 hours per rep every week, and we offer a 14-day guided pilot.",
                "es": "Entiendo perfectamente el cuidado del presupuesto; nuestros clientes recuperan la inversión en menos de 3 meses ahorrando 6.5 horas por vendedor a la semana.",
                "fr": "Je comprends votre attention au budget ; nos clients rentabilisent la solution en moins de 3 mois grâce à un gain de 6,5 heures par commercial par semaine.",
                "de": "Ich verstehe das Budgetthema sehr gut—durch 6,5 Stunden Zeitersparnis pro Vertriebsmitarbeiter amortisiert sich Fortivex meist in unter 3 Monaten.",
                "hi": "मैं आपके बजट की चिंता समझता हूँ—हमारे ग्राहक हर हफ्ते प्रति प्रतिनिधि 6.5 घंटे बचाकर 3 महीने से कम में पूरा ROI हासिल कर लेते हैं।",
            }
            parts.append(emp[lang])
        elif chunks and chunks[0].score >= 0.14:
            parts.append(self._summarize_chunk_for_voice(chunks[0], lang))
        else:
            intro = {
                "en": f"Thanks for reaching out{greeting_name}! Fortivex automates real-time multilingual sales calls, BANT qualification, and instant CRM sync.",
                "es": f"¡Gracias por comunicarte{greeting_name}! Fortivex automatiza llamadas de ventas multilingües en tiempo real, calificación BANT y sincronización CRM.",
                "fr": f"Merci de nous contacter{greeting_name} ! Fortivex automatise vos appels commerciaux multilingues, la qualification BANT et la synchronisation CRM.",
                "de": f"Vielen Dank für Ihre Anfrage{greeting_name}! Fortivex automatisiert mehrsprachige Verkaufsgespräche, Live-BANT-Qualifizierung und CRM-Sync.",
                "hi": f"संपर्क करने के लिए धन्यवाद{greeting_name}! Fortivex रियल-टाइम मल्टीलिंगुअल सेल्स कॉल, BANT क्वालिफिकेशन और CRM सिंक को ऑटोमेट करता है।",
            }
            parts.append(intro[lang])

        # 5. Adaptive next-best BANT question or booking CTA
        missing = qualification.missing_dimensions
        if not missing or qualification.score >= 70:
            cta = {
                "en": "Since you're a strong fit, would you like me to book a 30-minute live architecture demo tomorrow?",
                "es": "Dado el gran ajuste para tu equipo, ¿te gustaría que agendemos una demo en vivo de 30 minutos mañana?",
                "fr": "Votre profil correspond parfaitement—souhaitez-vous que je planifie une démo personnalisée de 30 minutes demain ?",
                "de": "Das passt hervorragend—soll ich Ihnen direkt eine 30-minütige Live-Demo für morgen reservieren?",
                "hi": "आपकी टीम के लिए यह बिल्कुल उपयुक्त है—क्या मैं कल के लिए 30 मिनट का लाइव डेमो बुक कर दूँ?",
            }
            parts.append(cta[lang])
        else:
            next_dim = missing[0]
            prompts: dict[str, dict[SupportedLang, str]] = {
                "need": {
                    "en": "What is the biggest sales bottleneck your team is trying to solve right now?",
                    "es": "¿Cuál es el principal reto de ventas o CRM que buscan resolver en su equipo?",
                    "fr": "Quel est le principal défi commercial ou CRM que votre équipe souhaite résoudre ?",
                    "de": "Was ist aktuell die größte Herausforderung in Ihrem Vertriebs- oder CRM-Prozess?",
                    "hi": "अभी आपकी सेल्स टीम किस सबसे बड़ी चुनौती को हल करना चाहती है?",
                },
                "timeline": {
                    "en": "How soon are you aiming to roll out a solution like this, and roughly how many reps are on your team?",
                    "es": "¿En qué plazo planean implementar una solución así y cuántos vendedores tiene su equipo?",
                    "fr": "Dans quel délai souhaitez-vous déployer cette solution et combien de commerciaux compte votre équipe ?",
                    "de": "Bis wann möchten Sie eine Lösung einführen und wie groß ist Ihr Vertriebsteam?",
                    "hi": "आप इस समाधान को कब तक लागू करना चाहते हैं और आपकी टीम में कितने सेल्स प्रतिनिधि हैं?",
                },
                "authority": {
                    "en": "What is your role in the evaluation, and who else joins the final decision?",
                    "es": "¿Cuál es tu rol en la evaluación y quién más participa en la decisión final?",
                    "fr": "Quel est votre rôle dans ce projet et qui d'autre participe à la décision ?",
                    "de": "Welche Rolle haben Sie bei der Evaluierung und wer entscheidet mit?",
                    "hi": "इस मूल्यांकन में आपकी क्या भूमिका है और अंतिम निर्णय में कौन-कौन शामिल होंगे?",
                },
                "budget": {
                    "en": "Do you already have a target budget range allocated for sales automation this quarter?",
                    "es": "¿Ya cuentan con un rango de presupuesto asignado para automatización comercial este trimestre?",
                    "fr": "Avez-vous déjà une enveloppe budgétaire dédiée à l'automatisation commerciale ce trimestre ?",
                    "de": "Haben Sie für dieses Quartal bereits ein Budget für Sales-Automatisierung eingeplant?",
                    "hi": "क्या आपने इस तिमाही के लिए सेल्स ऑटोमेशन हेतु कोई बजट तय किया है?",
                },
            }
            parts.append(prompts[next_dim][lang])

        return " ".join(parts)

    def generate_call_summary(self, session_id: str) -> CallSummary:
        """Compile and persist a structured post-call summary & business insights."""
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

        lead_name = memory.name or f"Prospect ({session_id[-6:]})"
        duration_sec = max(30, len(messages) * 24)

        next_steps: list[str] = []
        if appt:
            next_steps.append(f"Conduct scheduled Fortivex demo on {appt.slot_iso} ({appt.meeting_link})")
        elif qual.score >= 60:
            next_steps.append("Send tailored pricing & 14-day pilot invitation with calendar booking link")
        else:
            next_steps.append("Enroll lead in multilingual nurture sequence and follow up on uncovered BANT criteria")

        if memory.objections:
            next_steps.append(f"Address raised objections in follow-up brief: {', '.join(memory.objections)}")
        if memory.email:
            next_steps.append(f"Email call recap and SOC2/integration pack to {memory.email}")

        exec_summary = (
            f"{lead_name}{f' ({memory.role})' if memory.role else ''}"
            f"{f' from {memory.company}' if memory.company else ''} engaged in a "
            f"{LANGUAGE_NAMES[lang]}-language sales conversation across {len(user_msgs)} buyer turns. "
            f"Qualification reached {qual.score}/100 ({qual.tier}, stage: {qual.stage}) with estimated ACV of ${qual.estimated_acv_usd:,}. "
            f"{'Pain points identified: ' + ', '.join(memory.pain_points) + '. ' if memory.pain_points else ''}"
            f"{'Demo confirmed for ' + appt.slot_iso + '.' if appt else 'Pending demo slot confirmation.'}"
        )

        # Optionally upgrade executive_summary with Gemini if configured
        api_key = get_runtime_gemini_key()
        if api_key and user_msgs:
            try:
                from google import genai

                client = genai.Client(api_key=api_key)
                transcript = "\n".join(f"{m.role}: {m.text}" for m in messages)
                resp = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=(
                        "Write a concise 3-sentence executive sales call summary covering buyer intent, "
                        f"BANT qualification ({qual.score}/100, {qual.tier}), and next steps:\n\n{transcript}"
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
            company=memory.company,
            language=lang,
            turn_count=len(messages),
            duration_estimate_sec=duration_sec,
            executive_summary=exec_summary,
            key_pain_points=memory.pain_points,
            objections_raised=memory.objections,
            qualification=qual,
            average_sentiment=avg_sent,
            sentiment_trend=trend,
            next_steps=next_steps,
            booked_appointment=appt,
            generated_at=datetime.utcnow().isoformat(),
        )
        self.db.save_call_summary(summary)
        return summary
