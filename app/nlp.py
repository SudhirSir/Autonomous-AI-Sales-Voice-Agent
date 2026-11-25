"""Multilingual NLP (EN/HI/Hinglish/ES/FR/DE), Sentiment vs Intent, Memory & BANT Engine."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from app.config import settings
from app.models import (
    BANTProfile,
    LeadMemory,
    LeadQualification,
    SentimentResult,
    SupportedLang,
)


TTS_LOCALES: dict[SupportedLang, str] = {
    "en": "en-US",
    "es": "es-ES",
    "fr": "fr-FR",
    "de": "de-DE",
    "hi": "hi-IN",
    "hinglish": "en-IN",
}

LANGUAGE_NAMES: dict[SupportedLang, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
    "hinglish": "Hinglish",
}

HINGLISH_WORDS: set[str] = {
    "mujhe", "chahiye", "chaiye", "kya", "hai", "hain", "hum", "humari", "hamara",
    "hamari", "aap", "aapka", "aapki", "kitna", "kitne", "keemat", "batao", "bataiye",
    "karna", "chahte", "chahta", "kal", "aaj", "subah", "shaam", "baje", "theek",
    "nahi", "nhi", "haan", "hanji", "mera", "meri", "naam", "kaam", "mehnga",
    "sasta", "samjha", "bhai", "yaar", "kripya", "dhanyawad", "shukriya", "abhi",
    "baad", "pehle", "iske", "liye", "lekin", "aur", "par", "se", "ko", "mein",
}

LANG_STOPWORDS: dict[str, set[str]] = {
    "es": {
        "hola", "buenos", "gracias", "quiero", "necesito", "precio", "precios",
        "equipo", "empresa", "nosotros", "como", "para", "con", "una", "pero",
        "demo", "reunión", "mañana", "seguridad", "presupuesto", "soy", "estamos",
        "cuánto", "cuesta", "tienen", "podemos", "agendar", "cita", "integración",
    },
    "fr": {
        "bonjour", "merci", "voudrais", "besoin", "prix", "tarif", "équipe",
        "entreprise", "nous", "avec", "pour", "dans", "mais", "sécurité",
        "réunion", "demain", "budget", "sommes", "combien", "coûte", "pouvez",
        "planifier", "rendez-vous", "intégration", "je", "suis", "chez",
    },
    "de": {
        "hallo", "guten", "danke", "möchte", "brauchen", "preis", "preise",
        "unternehmen", "wir", "für", "mit", "aber", "sicherheit", "termin",
        "morgen", "budget", "sind", "wie", "viel", "kostet", "können",
        "buchen", "ich", "bin", "bei", "unsere", "vertrieb", "mitarbeiter",
    },
    "en": {
        "hello", "hi", "thanks", "need", "want", "price", "pricing", "team",
        "company", "we", "with", "for", "budget", "timeline", "demo", "book",
        "schedule", "security", "salesforce", "hubspot", "our", "are", "how",
        "what", "your", "this", "month", "sales", "reps",
    },
}


def detect_language(text: str, fallback: SupportedLang = "en") -> SupportedLang:
    """Detect language among en, hi, hinglish, es, fr, de using script and lexical cues."""
    if not text or not text.strip():
        return fallback

    # Devanagari Unicode block -> Hindi
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    lower = text.lower()
    tokens = set(re.findall(r"[a-zA-Záéíóúñüàâçéèêëîïôùûüÿäöß]+", lower))

    # Hinglish code-switching detection (Romanized Hindi words mixed with English/Roman tokens)
    hinglish_hits = len(tokens & HINGLISH_WORDS)
    if hinglish_hits >= 2:
        return "hinglish"

    scores: dict[str, int] = {"en": 0, "es": 0, "fr": 0, "de": 0}
    for lang in ("es", "fr", "de", "en"):
        scores[lang] = len(tokens & LANG_STOPWORDS[lang])

    if re.search(r"[ñ¿¡áíóú]", lower):
        scores["es"] += 2
    if re.search(r"[àâçéèêëîïôùûœ]", lower):
        scores["fr"] += 2
    if re.search(r"[äöüß]", lower):
        scores["de"] += 2

    best_lang = max(("es", "fr", "de", "en"), key=lambda l: scores[l])
    if scores[best_lang] == 0:
        return "hinglish" if hinglish_hits == 1 and scores["en"] == 0 else fallback
    if best_lang != "en" and scores[best_lang] >= scores["en"]:
        return best_lang  # type: ignore[return-value]
    if scores["en"] > 0:
        return "en"
    return fallback


POSITIVE_TERMS = {
    "great", "awesome", "love", "interested", "perfect", "helpful", "impressive",
    "excited", "ready", "sounds good", "yes", "definitely", "excellent", "fast",
    "genial", "perfecto", "excelente", "interesado", "bueno", "sí", "listo",
    "parfait", "génial", "intéressé", "super", "oui", "prêt",
    "super", "perfekt", "interessiert", "ausgezeichnet", "ja", "bereit",
    "अच्छा", "बढ़िया", "सही", "तैयार", "पसंद", "शानदार",
    "badhiya", "accha", "badiya", "theek", "haan",
}

FRUSTRATION_TERMS = {
    "frustrated", "annoying", "waste of time", "useless", "terrible", "ridiculous",
    "horrible", "stop calling", "fed up", "pathetic", "angry", "irritated",
    "बेकार", "परेशान", "बकवास", "frustré", "genervt", "molesto",
}

NEGATIVE_TERMS = {
    "expensive", "costly", "too much", "bad", "hate", "problem", "slow",
    "concerned", "worried", "skeptical", "not sure", "no budget", "cancel",
    "caro", "problema", "preocupado", "difícil", "lento",
    "cher", "problème", "inquiet", "difficile", "trop",
    "teuer", "problem", "besorgt", "schwierig", "skeptisch",
    "महंगा", "समस्या", "चिंता", "मुश्किल", "नहीं",
    "mehnga", "mushkil", "nahi", "nhi",
}

URGENT_TERMS = {
    "asap", "urgent", "immediately", "this week", "this month", "right away",
    "q1", "q2", "deadline", "urgente", "inmediato", "sofort", "dringend",
    "तुरंत", "जल्दी", "abhi", "jaldi",
}


def analyze_sentiment(text: str) -> SentimentResult:
    """Analyze emotional sentiment separately from commercial purchase intent."""
    lower = text.lower()
    pos_hits = sum(1 for t in POSITIVE_TERMS if t in lower)
    neg_hits = sum(1 for t in NEGATIVE_TERMS if t in lower)
    frust_hits = sum(1 for t in FRUSTRATION_TERMS if t in lower)
    urg_hits = sum(1 for t in URGENT_TERMS if t in lower)

    signals: list[str] = []
    has_booking = any(
        k in lower
        for k in [
            "demo", "schedule", "book", "calendar", "meeting", "slot",
            "agendar", "cita", "planifier", "termin", "buchen", "डेमो", "मीटिंग",
        ]
    )
    if has_booking:
        signals.append("booking_intent")

    if any(k in lower for k in ["reschedule", "move the", "change the time", "push back", "postpone", "reprogramar", "verschieben"]):
        signals.append("reschedule_intent")
    if any(k in lower for k in ["cancel appointment", "cancel the demo", "cancel my demo", "cancel meeting", "cancelar cita"]):
        signals.append("cancel_intent")

    has_price = any(
        k in lower
        for k in [
            "price", "pricing", "cost", "expensive", "discount", "budget",
            "precio", "caro", "prix", "cher", "preis", "teuer", "कीमत", "महंगा",
            "keemat", "paisa", "mehnga",
        ]
    )
    if has_price:
        signals.append("price_discussion")
        if neg_hits > 0 or frust_hits > 0:
            signals.append("price_objection")

    trust_concern = any(
        k in lower
        for k in [
            "soc2", "soc 2", "gdpr", "hipaa", "security", "privacy", "encrypt",
            "trust", "data leak", "seguridad", "sécurité", "sicherheit", "सुरक्षा",
        ]
    )
    if trust_concern:
        signals.append("security_concern")

    if any(k in lower for k in ["gong", "chorus", "drift", "intercom", "clari", "competitor", "in-house"]):
        signals.append("competitor_comparison")

    has_buying = any(
        k in lower
        for k in [
            "ready", "sign", "pilot", "trial", "start", "move forward", "go ahead",
            "roll out", "purchase", "buy", "listo", "prêt", "bereit", "तैयार", "chahiye",
        ]
    )
    if has_buying:
        signals.append("buying_signal")
    if urg_hits > 0:
        signals.append("urgency")

    # Emotional polarity (kept distinct from purchase intent)
    raw_polarity = (pos_hits * 0.35) - (neg_hits * 0.35) - (frust_hits * 0.55)
    polarity = max(-1.0, min(1.0, round(raw_polarity, 2)))
    frustration_detected = frust_hits > 0 or polarity <= -0.55

    if frustration_detected:
        label = "Frustrated"
    elif polarity >= 0.25:
        label = "Positive"
    elif polarity <= -0.25:
        label = "Negative"
    elif neg_hits > 0 and pos_hits == 0:
        label = "Skeptical"
    else:
        label = "Neutral"

    # Commercial Purchase Intent (0-100), separated from emotional polarity
    intent_score = 18
    if has_buying:
        intent_score += 32
    if has_booking:
        intent_score += 30
    if urg_hits > 0:
        intent_score += 12
    if has_price:
        intent_score += 10
    if any(k in lower for k in ["wrong number", "not interested", "stop calling", "do not call", "no budget"]):
        intent_score = 5
    intent_score = max(0, min(100, intent_score))

    if intent_score >= 70:
        readiness = "High"
    elif intent_score >= 40:
        readiness = "Medium"
    elif intent_score >= 15:
        readiness = "Low"
    else:
        readiness = "None"

    word_count = len(text.split())
    engagement = "High" if (word_count >= 14 or len(signals) >= 2) else ("Medium" if word_count >= 5 else "Low")
    confidence = min(0.97, round(0.68 + 0.06 * (pos_hits + neg_hits + frust_hits + len(signals)), 2))

    return SentimentResult(
        polarity=polarity,
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        intent_score=intent_score,
        purchase_readiness=readiness,  # type: ignore[arg-type]
        frustration_detected=frustration_detected,
        trust_concern=trust_concern,
        engagement_level=engagement,  # type: ignore[arg-type]
        signals=signals,
    )


def extract_memory_and_bant(
    text: str,
    memory: LeadMemory,
    bant: BANTProfile,
) -> tuple[LeadMemory, BANTProfile]:
    """Update contextual LeadMemory and BANTProfile from the latest user utterance."""
    mem = memory.model_copy(deep=True)
    b = bant.model_copy(deep=True)
    lower = text.lower()

    # 1. Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if email_match:
        mem.email = email_match.group(0).rstrip(".,!?;:").lower()

    # 2. Phone number (US 3-3-4, Indian 5-5 / +91, international formats; avoid prices/team sizes)
    phone_match = re.search(
        r"(?<!\$)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s*\d{3}[\s.-]?\d{4}|\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b|\b\d{5}[\s.-]?\d{5}\b)",
        text,
    )
    if phone_match:
        mem.phone = phone_match.group(0).strip()

    # 3. Name extraction across languages (including Hinglish)
    name_patterns = [
        r"\b(?i:my name is|i'm|i am|this is|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"\b(?i:me llamo|soy)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)",
        r"\b(?i:je m'appelle|je suis)\s+([A-ZÀÂÇÉÈÊËÎÏÔÙÛ][a-zàâçéèêëîïôùû]+(?:\s+[A-ZÀÂÇÉÈÊËÎÏÔÙÛ][a-zàâçéèêëîïôùû]+)?)",
        r"\b(?i:ich heiße|ich bin)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)",
        r"\b(?i:mera naam)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"मेरा नाम\s+([^\s,।]+(?:\s+[^\s,।]+)?)",
    ]
    non_names = {
        "interested", "looking", "calling", "ready", "from", "here", "sure",
        "happy", "concerned", "good", "fine", "frustrated", "worried", "not",
    }
    for pat in name_patterns:
        m = re.search(pat, text)
        if m:
            candidate = m.group(1).strip()
            if candidate.lower().split()[0] not in non_names:
                mem.name = candidate
                break

    # 4. Company extraction
    comp_patterns = [
        r"\b(?i:from|at|with|company is|work at|represent)\s+([A-Z][A-Za-z0-9&.-]+(?:\s+(?:Inc|Corp|Labs|AI|Tech|Software|Group|LLC|Solutions|Systems))?)",
        r"\b(?i:en|de la empresa)\s+([A-Z][A-Za-z0-9&.-]+)",
        r"\b(?i:chez|de)\s+([A-Z][A-Za-z0-9&.-]+)",
        r"\b(?i:bei|von)\s+([A-Z][A-Za-z0-9&.-]+)",
    ]
    for pat in comp_patterns:
        m = re.search(pat, text)
        if m:
            comp = m.group(1).strip(".,!?")
            if comp.lower() not in {"fortivex", "salesforce", "hubspot", "google", "monday", "tuesday", "wednesday", "thursday", "friday"}:
                mem.company = comp
                break

    # 5. Role / Authority
    authority_patterns = [
        (r"\b(ceo|cto|cfo|cro|coo|founder|co-founder|president|vp|vice president)\b", 25, "Executive Decision Maker"),
        (r"\b(head of|director|svp|evp|general manager|gm)\b", 22, "Department Director / Head"),
        (r"\b(manager|lead|revops|sales ops|team lead|architect)\b", 16, "Team Manager / Evaluator"),
        (r"\b(rep|sdr|bdr|analyst|developer|intern|student)\b", 10, "Individual Contributor / Researcher"),
    ]
    for pat, score, label in authority_patterns:
        m = re.search(pat, lower)
        if m:
            role_txt = m.group(0).upper() if len(m.group(0)) <= 4 else m.group(0).title()
            mem.role = role_txt
            if score > b.authority_score:
                b.authority = f"{role_txt} ({label})"
                b.authority_score = score
            break
    if any(k in lower for k in ["i decide", "decision maker", "sign off", "my team", "approve", "tomo la decisión", "je décide"]):
        b.authority_score = max(b.authority_score, 23)
        if not b.authority:
            b.authority = "Confirmed Decision Maker"

    # 6. Team size
    size_match = re.search(
        r"\b(\d{1,5})\s+(?:[a-z]+\s+)?(?:reps|seats|users|people|agents|sellers|employees|personas|utilisateurs|mitarbeiter|लोगों?|log)",
        lower,
    )
    if size_match:
        mem.team_size = int(size_match.group(1))

    # 7. Budget extraction
    has_budget_word = any(
        k in lower
        for k in ["budget", "spend", "pricing", "cost", "afford", "allocated", "presupuesto", "बजट", "$", "€"]
    )
    budget_extracted = False
    if has_budget_word:
        for mm in re.finditer(r"(\$|€|usd|eur)?\s*(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\s*(k|m|thousand|million)?\b", lower):
            num = float(mm.group(2).replace(",", ""))
            unit = (mm.group(3) or "").lower()
            if unit in {"k", "thousand"}:
                num *= 1000
            elif unit in {"m", "million"}:
                num *= 1_000_000
            if num >= 500:
                b.budget = f"~${int(num):,}"
                b.budget_score = 25 if num >= 15000 else (20 if num >= 5000 else 14)
                budget_extracted = True
                break
    if not budget_extracted:
        if any(k in lower for k in ["no budget", "zero budget", "free only", "sin presupuesto", "0 budget"]):
            b.budget = "No allocated budget"
            b.budget_score = 3
        elif any(k in lower for k in ["budget approved", "budget is allocated", "we have budget", "presupuesto aprobado", "budget approuvé"]):
            b.budget = b.budget or "Approved budget allocated"
            b.budget_score = max(b.budget_score, 22)
        elif any(k in lower for k in ["growth plan", "scale plan", "enterprise plan", "499", "1499", "1,499", "36000", "36,000"]):
            b.budget = b.budget or "Evaluating Fortivex paid tier"
            b.budget_score = max(b.budget_score, 18)

    # 8. Need & Pain points
    pain_map = {
        "manual crm": "Manual CRM data entry overhead",
        "data entry": "Manual CRM data entry overhead",
        "logging": "Rep time lost logging calls",
        "conversion": "Low lead-to-demo conversion rate",
        "missed leads": "Slow inbound lead response time",
        "follow up": "Inconsistent sales follow-up",
        "follow-up": "Inconsistent sales follow-up",
        "multilingual": "Multilingual buyer coverage gap",
        "spanish": "Multilingual buyer coverage gap",
        "hindi": "Multilingual buyer coverage gap",
        "hinglish": "Multilingual buyer coverage gap",
        "french": "Multilingual buyer coverage gap",
        "german": "Multilingual buyer coverage gap",
        "language": "Multilingual buyer coverage gap",
        "qualify": "Unqualified leads wasting AE time",
        "qualification": "Unqualified leads wasting AE time",
        "forecast": "Lack of pipeline visibility",
        "compliance": "Strict SOC2/GDPR compliance requirements",
        "security": "Enterprise data security & privacy",
        "automate": "Need to automate sales workflows",
        "automation": "Need to automate sales workflows",
    }
    matched_pains: list[str] = []
    for kw, desc in pain_map.items():
        if kw in lower and desc not in mem.pain_points:
            mem.pain_points.append(desc)
        if kw in lower:
            matched_pains.append(desc)
    if matched_pains or any(k in lower for k in ["need", "looking for", "challenge", "problem", "struggling", "want to", "necesito", "besoin", "brauchen", "चाहिए", "chahiye"]):
        if mem.pain_points:
            b.need = "; ".join(mem.pain_points[:3])
            b.need_score = min(25, 14 + 4 * len(mem.pain_points))
        else:
            b.need = b.need or "Evaluating AI voice sales automation"
            b.need_score = max(b.need_score, 14)

    # 9. Timeline extraction
    timeline_rules = [
        (r"\b(this week|immediately|asap|right away|today|tomorrow|urgente|sofort|तुरंत|abhi)\b", "Immediate (0–2 weeks)", 25),
        (r"\b(this month|next week|2 weeks|30 days|q1|q2|q3|q4|next month|este mes|ce mois)\b", "Near-term (1–3 months)", 22),
        (r"\b(quarter|60 days|90 days|this year|soon|pronto|bientôt|bald)\b", "Mid-term (1–2 quarters)", 17),
        (r"\b(next year|just exploring|just looking|no rush|solo mirando)\b", "Exploratory / Long-term", 8),
    ]
    for pat, label, score in timeline_rules:
        if re.search(pat, lower):
            b.timeline = label
            b.timeline_score = max(b.timeline_score, score)
            break

    # 10. Objections, Buying Signals & Competitors
    objection_map = {
        "expensive": "Pricing / budget concern",
        "too much": "Pricing / budget concern",
        "caro": "Pricing / budget concern",
        "cher": "Pricing / budget concern",
        "teuer": "Pricing / budget concern",
        "महंगा": "Pricing / budget concern",
        "mehnga": "Pricing / budget concern",
        "already use": "Incumbent tool lock-in",
        "already have": "Incumbent tool lock-in",
        "security": "Security & data governance review",
        "gdpr": "GDPR data residency concern",
        "hipaa": "HIPAA compliance review",
        "implementation": "Onboarding & migration effort",
        "time to deploy": "Deployment timeline concern",
    }
    for kw, obj in objection_map.items():
        if kw in lower and obj not in mem.objections:
            mem.objections.append(obj)

    for comp in ("Gong", "Chorus", "Drift", "Intercom", "Clari", "HubSpot", "Salesforce"):
        if comp.lower() in lower and comp not in mem.competitors_mentioned:
            mem.competitors_mentioned.append(comp)

    return mem, b


def compute_qualification(
    bant: BANTProfile,
    memory: LeadMemory,
    has_booking: bool = False,
    last_utterance: str = "",
) -> LeadQualification:
    """Deterministic, explainable BANT scoring (0-100) with Disqualified handling."""
    lower = last_utterance.lower()
    disqualify_phrases = [
        "wrong number", "not interested", "stop calling", "do not call",
        "remove me", "student project", "personal hobby", "free forever only",
    ]
    explicit_disqualify = any(p in lower for p in disqualify_phrases) or (
        bant.budget == "No allocated budget" and bant.timeline_score <= 8 and not has_booking
    )

    rationale: list[str] = [
        f"Budget: {bant.budget_score}/25 ({bant.budget or 'missing'})",
        f"Authority: {bant.authority_score}/25 ({bant.authority or 'missing'})",
        f"Need: {bant.need_score}/25 ({bant.need or 'missing'})",
        f"Timeline: {bant.timeline_score}/25 ({bant.timeline or 'missing'})",
    ]

    if explicit_disqualify:
        return LeadQualification(
            score=min(15, bant.budget_score + bant.authority_score + bant.need_score + bant.timeline_score),
            tier="Disqualified",
            stage="Disqualified",
            bant=bant,
            missing_dimensions=["budget", "need", "timeline"],
            estimated_acv_usd=0,
            score_rationale=rationale + ["Disqualified: explicit non-fit or zero-budget opt-out."],
        )

    total = bant.budget_score + bant.authority_score + bant.need_score + bant.timeline_score
    if has_booking:
        total = min(100, max(total, 75))
        rationale.append("Booking bonus applied (+demo committed)")
    elif memory.team_size and memory.team_size >= 25:
        total = min(100, total + 5)
        rationale.append(f"Team size >= 25 seats (+5 boost for {memory.team_size} seats)")

    if total >= 70:
        tier = "Hot"
    elif total >= 40:
        tier = "Warm"
    else:
        tier = "Cold"

    missing: list[str] = []
    if bant.need_score < 12:
        missing.append("need")
    if bant.timeline_score < 12:
        missing.append("timeline")
    if bant.authority_score < 12:
        missing.append("authority")
    if bant.budget_score < 12:
        missing.append("budget")

    if has_booking:
        stage = "Demo Scheduled"
    elif total >= 65:
        stage = "Proposal"
    elif total >= 35:
        stage = "Qualification"
    else:
        stage = "Discovery"

    if memory.team_size:
        if memory.team_size <= 25:
            acv = 499 * 12
        elif memory.team_size <= 100:
            acv = 1499 * 12
        else:
            acv = max(36000, memory.team_size * 360)
    elif bant.budget and "$" in bant.budget:
        digits = re.sub(r"[^\d]", "", bant.budget)
        acv = int(digits) if digits else 17988
    elif total >= 70:
        acv = 36000
    elif total >= 40:
        acv = 17988
    else:
        acv = 5988

    return LeadQualification(
        score=total,
        tier=tier,  # type: ignore[arg-type]
        stage=stage,  # type: ignore[arg-type]
        bant=bant,
        missing_dimensions=missing,
        estimated_acv_usd=acv,
        score_rationale=rationale,
    )


def validate_business_slot(slot_iso: str, tz_name: str = "UTC") -> tuple[Optional[str], Optional[str]]:
    """Validate that a requested ISO datetime is in the future and within Mon-Fri business hours."""
    try:
        clean = slot_iso.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            try:
                dt = dt.replace(tzinfo=ZoneInfo(tz_name))
            except Exception:
                dt = dt.replace(tzinfo=timezone.utc)
        dt_utc = dt.astimezone(timezone.utc)
    except Exception:
        return None, f"Invalid date/time format: '{slot_iso}'. Please provide a valid ISO date/time."

    now_utc = datetime.now(timezone.utc)
    if dt_utc <= now_utc + timedelta(minutes=15):
        return None, "Requested slot is in the past. Please choose an upcoming weekday slot."

    if dt_utc.weekday() >= 5:
        return None, "Demos are only available Monday through Friday during business hours (09:00–18:00 UTC)."

    if not (settings.business_hours_start <= dt_utc.hour < settings.business_hours_end):
        return (
            None,
            f"Requested hour ({dt_utc.strftime('%H:%M')} UTC) is outside business hours "
            f"({settings.business_hours_start:02d}:00–{settings.business_hours_end:02d}:00 UTC).",
        )

    normalized = dt_utc.replace(second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    return normalized, None


def parse_requested_slot(
    text: str, available_slots: list[str], tz_name: str = "UTC"
) -> tuple[Optional[str], Optional[str]]:
    """Parse natural-language time/slot preference and validate business hours.

    Returns (validated_slot_iso, validation_error).
    """
    lower = text.lower()
    if not available_slots and not any(k in lower for k in ["tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday"]):
        return None, None

    if available_slots:
        if any(k in lower for k in ["first", "option 1", "slot 1", "#1"]):
            return available_slots[0], None
        if len(available_slots) > 1 and any(k in lower for k in ["second", "option 2", "slot 2", "#2"]):
            return available_slots[1], None
        if len(available_slots) > 2 and any(k in lower for k in ["third", "option 3", "slot 3", "#3"]):
            return available_slots[2], None

    now = datetime.now(timezone.utc)
    target_date = None
    if any(k in lower for k in ["yesterday", "last week"]):
        return None, "Cannot schedule an appointment in the past. Please choose an upcoming weekday."

    if re.search(r"\b(?:tomorrow|mañana|demain|morgen|kal)\b", lower) or "कल" in lower:
        cand = now + timedelta(days=1)
        if cand.weekday() >= 5:
            # Move to Monday if tomorrow is weekend and user didn't explicitly ask for Saturday/Sunday
            cand = cand + timedelta(days=(7 - cand.weekday()))
        target_date = cand.date()
    elif "today" in lower or "आज" in lower:
        target_date = now.date()

    weekdays = {
        "monday": 0, "lunes": 0, "lundi": 0, "montag": 0,
        "tuesday": 1, "martes": 1, "mardi": 1, "dienstag": 1,
        "wednesday": 2, "miércoles": 2, "mercredi": 2, "mittwoch": 2,
        "thursday": 3, "jueves": 3, "jeudi": 3, "donnerstag": 3,
        "friday": 4, "viernes": 4, "vendredi": 4, "freitag": 4,
        "saturday": 5, "sunday": 6,
    }
    for name, wd in weekdays.items():
        if name in lower:
            days_ahead = (wd - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = (now + timedelta(days=days_ahead)).date()
            break

    target_hour: Optional[int] = None
    hm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", lower)
    if hm:
        hr = int(hm.group(1))
        ampm = hm.group(3)
        if ampm == "pm" and hr < 12:
            hr += 12
        elif ampm == "am" and hr == 12:
            hr = 0
        elif not ampm and 1 <= hr <= 7:
            hr += 12
        target_hour = hr
    elif re.search(r"\b(?:morning|mañana|matin|subah)\b", lower) or "सुबह" in lower:
        target_hour = 10
    elif re.search(r"\b(?:afternoon|tarde|après-midi|shaam)\b", lower) or "दोपहर" in lower:
        target_hour = 14

    if target_date is not None or target_hour is not None:
        # First check explicit weekday/hour validation
        chosen_date = target_date or (now + timedelta(days=1)).date()
        if chosen_date.weekday() >= 5 and target_date is not None:
            return None, "Demos are only available Monday through Friday (09:00–18:00 UTC)."
        if target_hour is not None and not (settings.business_hours_start <= target_hour < settings.business_hours_end):
            return (
                None,
                f"{target_hour:02d}:00 is outside business hours ({settings.business_hours_start:02d}:00–{settings.business_hours_end:02d}:00 UTC).",
            )

        for slot in available_slots:
            dt = datetime.fromisoformat(slot.replace("Z", "+00:00"))
            date_ok = (target_date is None) or (dt.date() == target_date)
            hour_ok = (target_hour is None) or (dt.hour == target_hour)
            if date_ok and hour_ok:
                return slot, None

        chosen_hour = target_hour or 14
        candidate_iso = datetime(
            chosen_date.year, chosen_date.month, chosen_date.day, chosen_hour, 0, tzinfo=timezone.utc
        ).isoformat().replace("+00:00", "Z")
        return validate_business_slot(candidate_iso, tz_name=tz_name)

    if available_slots and any(
        k in lower for k in ["yes", "sure", "works", "book it", "confirm", "schedule it", "agendar", "sí", "oui", "ja", "हाँ", "बुक", "theek"]
    ):
        return available_slots[0], None

    return None, None
