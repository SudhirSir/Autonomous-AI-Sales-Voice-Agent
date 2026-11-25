"""Multilingual NLP, Sentiment Analysis, Contextual Memory & BANT Qualification Engine."""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

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
}

LANGUAGE_NAMES: dict[SupportedLang, str] = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "hi": "Hindi",
}

LANG_STOPWORDS: dict[SupportedLang, set[str]] = {
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
        "buchen", "ich", "bin", "bei", "unsere", "vertrieb",
    },
    "en": {
        "hello", "hi", "thanks", "need", "want", "price", "pricing", "team",
        "company", "we", "with", "for", "budget", "timeline", "demo", "book",
        "schedule", "security", "salesforce", "hubspot", "our", "are", "how",
    },
    "hi": {
        "नमस्ते", "धन्यवाद", "चाहिए", "कीमत", "बजट", "सुरक्षा", "टीम", "कंपनी",
        "डेमो", "मीटिंग", "हम", "मेरा", "नाम", "क्या", "कैसे", "कल", "आज",
    },
}


def detect_language(text: str, fallback: SupportedLang = "en") -> SupportedLang:
    """Detect language among en, es, fr, de, hi using script and lexical cues."""
    if not text or not text.strip():
        return fallback

    # Devanagari Unicode block -> Hindi
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    lower = text.lower()
    tokens = set(re.findall(r"[a-zA-Záéíóúñüàâçéèêëîïôùûüÿäöß]+", lower))

    scores: dict[SupportedLang, int] = {"en": 0, "es": 0, "fr": 0, "de": 0, "hi": 0}
    for lang in ("es", "fr", "de", "en"):
        scores[lang] = len(tokens & LANG_STOPWORDS[lang])

    # Character diacritic boosts
    if re.search(r"[ñ¿¡áíóú]", lower):
        scores["es"] += 2
    if re.search(r"[àâçéèêëîïôùûœ]", lower):
        scores["fr"] += 2
    if re.search(r"[äöüß]", lower):
        scores["de"] += 2

    best_lang = max(("es", "fr", "de", "en"), key=lambda l: scores[l])
    if scores[best_lang] == 0:
        return fallback
    # Require non-English to strictly beat English if English also has hits
    if best_lang != "en" and scores[best_lang] >= scores["en"]:
        return best_lang
    return "en" if scores["en"] > 0 else fallback


POSITIVE_TERMS = {
    "great", "awesome", "love", "interested", "perfect", "helpful", "impressive",
    "excited", "ready", "sounds good", "yes", "definitely", "excellent", "fast",
    "genial", "perfecto", "excelente", "interesado", "bueno", "sí", "listo",
    "parfait", "génial", "intéressé", "super", "oui", "prêt",
    "super", "perfekt", "interessiert", "ausgezeichnet", "ja", "bereit",
    "अच्छा", "बढ़िया", "सही", "तैयार", "पसंद", "शानदार",
}

NEGATIVE_TERMS = {
    "expensive", "costly", "too much", "bad", "frustrated", "hate", "problem",
    "slow", "concerned", "worried", "skeptical", "not sure", "no budget", "cancel",
    "caro", "problema", "preocupado", "difícil", "lento",
    "cher", "problème", "inquiet", "difficile", "trop",
    "teuer", "problem", "besorgt", "schwierig", "skeptisch",
    "महंगा", "समस्या", "चिंता", "मुश्किल", "नहीं",
}

URGENT_TERMS = {
    "asap", "urgent", "immediately", "this week", "this month", "right away",
    "q1", "q2", "deadline", "urgente", "inmediato", "sofort", "dringend", "तुरंत", "जल्दी",
}


def analyze_sentiment(text: str) -> SentimentResult:
    """Compute per-turn sentiment polarity, label, and commercial intent signals."""
    lower = text.lower()
    pos_hits = sum(1 for t in POSITIVE_TERMS if t in lower)
    neg_hits = sum(1 for t in NEGATIVE_TERMS if t in lower)
    urg_hits = sum(1 for t in URGENT_TERMS if t in lower)

    signals: list[str] = []
    if any(k in lower for k in ["demo", "schedule", "book", "calendar", "meeting", "slot", "agendar", "cita", "planifier", "termin", "buchen", "डेमो", "मीटिंग"]):
        signals.append("booking_intent")
    if any(k in lower for k in ["price", "pricing", "cost", "expensive", "discount", "budget", "precio", "caro", "prix", "cher", "preis", "teuer", "कीमत", "महंगा"]):
        signals.append("price_discussion")
        if neg_hits > 0:
            signals.append("price_objection")
    if any(k in lower for k in ["soc2", "soc 2", "gdpr", "hipaa", "security", "privacy", "encrypt", "seguridad", "sécurité", "sicherheit", "सुरक्षा"]):
        signals.append("security_concern")
    if any(k in lower for k in ["gong", "chorus", "drift", "intercom", "clari", "competitor", "in-house"]):
        signals.append("competitor_comparison")
    if any(k in lower for k in ["ready", "sign", "pilot", "trial", "start", "move forward", "go ahead", "listo", "prêt", "bereit", "तैयार"]):
        signals.append("buying_signal")
    if urg_hits > 0:
        signals.append("urgency")

    raw = (pos_hits * 0.35) - (neg_hits * 0.4) + (0.2 if "buying_signal" in signals else 0.0) + (0.15 if "booking_intent" in signals else 0.0)
    polarity = max(-1.0, min(1.0, round(raw, 2)))

    if urg_hits > 0 and polarity >= 0:
        label = "Urgent"
    elif polarity >= 0.25:
        label = "Positive"
    elif polarity <= -0.3:
        label = "Negative"
    elif neg_hits > 0 and pos_hits == 0:
        label = "Skeptical"
    else:
        label = "Neutral"

    confidence = min(0.96, round(0.68 + 0.07 * (pos_hits + neg_hits + len(signals)), 2))
    return SentimentResult(polarity=polarity, label=label, confidence=confidence, signals=signals)


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

    # 2. Name extraction across languages
    name_patterns = [
        r"\b(?i:my name is|i'm|i am|this is|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        r"\b(?i:me llamo|soy)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)?)",
        r"\b(?i:je m'appelle|je suis)\s+([A-ZÀÂÇÉÈÊËÎÏÔÙÛ][a-zàâçéèêëîïôùû]+(?:\s+[A-ZÀÂÇÉÈÊËÎÏÔÙÛ][a-zàâçéèêëîïôùû]+)?)",
        r"\b(?i:ich heiße|ich bin)\s+([A-ZÄÖÜ][a-zäöüß]+(?:\s+[A-ZÄÖÜ][a-zäöüß]+)?)",
        r"मेरा नाम\s+([^\s,।]+(?:\s+[^\s,।]+)?)",
    ]
    non_names = {"interested", "looking", "calling", "ready", "from", "here", "sure", "happy", "concerned", "good", "fine"}
    for pat in name_patterns:
        m = re.search(pat, text)
        if m:
            candidate = m.group(1).strip()
            if candidate.lower().split()[0] not in non_names:
                mem.name = candidate
                break

    # 3. Company extraction
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

    # 4. Role / Authority
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

    # 5. Team size
    size_match = re.search(
        r"\b(\d{1,5})\s+(?:[a-z]+\s+)?(?:reps|seats|users|people|agents|sellers|employees|personas|utilisateurs|mitarbeiter|लोगों?)",
        lower,
    )
    if size_match:
        mem.team_size = int(size_match.group(1))

    # 6. Budget extraction
    has_budget_word = any(k in lower for k in ["budget", "spend", "pricing", "cost", "afford", "allocated", "presupuesto", "बजट", "$", "€"])
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
        if any(k in lower for k in ["no budget", "zero budget", "free only", "sin presupuesto"]):
            b.budget = "No allocated budget"
            b.budget_score = 5
        elif any(k in lower for k in ["budget approved", "budget is allocated", "we have budget", "presupuesto aprobado", "budget approuvé"]):
            b.budget = b.budget or "Approved budget allocated"
            b.budget_score = max(b.budget_score, 22)
        elif any(k in lower for k in ["growth plan", "scale plan", "enterprise plan", "499", "1499", "1,499", "36000", "36,000"]):
            b.budget = b.budget or "Evaluating Fortivex paid tier"
            b.budget_score = max(b.budget_score, 18)

    # 7. Need & Pain points
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
        "french": "Multilingual buyer coverage gap",
        "german": "Multilingual buyer coverage gap",
        "language": "Multilingual buyer coverage gap",
        "qualify": "Unqualified leads wasting AE time",
        "qualification": "Unqualified leads wasting AE time",
        "forecast": "Lack of pipeline visibility",
        "compliance": "Strict SOC2/GDPR compliance requirements",
        "security": "Enterprise data security & privacy",
        "automate": "Need to automate outbound/inbound sales",
        "automation": "Need to automate sales workflows",
    }
    matched_pains: list[str] = []
    for kw, desc in pain_map.items():
        if kw in lower and desc not in mem.pain_points:
            mem.pain_points.append(desc)
        if kw in lower:
            matched_pains.append(desc)
    if matched_pains or any(k in lower for k in ["need", "looking for", "challenge", "problem", "struggling", "want to", "necesito", "besoin", "brauchen", "चाहिए"]):
        if mem.pain_points:
            b.need = "; ".join(mem.pain_points[:3])
            b.need_score = min(25, 14 + 4 * len(mem.pain_points))
        else:
            b.need = b.need or "Evaluating AI voice sales automation"
            b.need_score = max(b.need_score, 14)

    # 8. Timeline extraction
    timeline_rules = [
        (r"\b(this week|immediately|asap|right away|today|tomorrow|urgente|sofort|तुरंत)\b", "Immediate (0–2 weeks)", 25),
        (r"\b(this month|next week|2 weeks|30 days|q1|q2|q3|q4|next month|este mes|ce mois)\b", "Near-term (1–3 months)", 22),
        (r"\b(quarter|60 days|90 days|this year|soon|pronto|bientôt|bald)\b", "Mid-term (1–2 quarters)", 17),
        (r"\b(next year|just exploring|just looking|no rush|solo mirando)\b", "Exploratory / Long-term", 8),
    ]
    for pat, label, score in timeline_rules:
        if re.search(pat, lower):
            b.timeline = label
            b.timeline_score = max(b.timeline_score, score)
            break

    # 9. Objections & Competitors
    objection_map = {
        "expensive": "Pricing / budget concern",
        "too much": "Pricing / budget concern",
        "caro": "Pricing / budget concern",
        "cher": "Pricing / budget concern",
        "teuer": "Pricing / budget concern",
        "महंगा": "Pricing / budget concern",
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
) -> LeadQualification:
    """Calculate total lead score (0-100), tier, stage, and ACV estimate."""
    total = bant.budget_score + bant.authority_score + bant.need_score + bant.timeline_score
    if has_booking:
        total = min(100, max(total, 75))
    elif memory.team_size and memory.team_size >= 25:
        total = min(100, total + 5)

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

    # Estimate ACV in USD
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
        tier=tier,
        stage=stage,
        bant=bant,
        missing_dimensions=missing,
        estimated_acv_usd=acv,
    )


def parse_requested_slot(text: str, available_slots: list[str]) -> Optional[str]:
    """Match a natural-language time/slot preference against available ISO slots."""
    lower = text.lower()
    if not available_slots:
        return None

    # Explicit slot index ("slot 1", "option 2", "first one")
    if any(k in lower for k in ["first", "option 1", "slot 1", "#1"]):
        return available_slots[0]
    if len(available_slots) > 1 and any(k in lower for k in ["second", "option 2", "slot 2", "#2"]):
        return available_slots[1]
    if len(available_slots) > 2 and any(k in lower for k in ["third", "option 3", "slot 3", "#3"]):
        return available_slots[2]

    # Match hour or day keywords
    now = datetime.utcnow()
    target_date = None
    if any(k in lower for k in ["tomorrow", "mañana", "demain", "morgen", "कल"]):
        target_date = (now + timedelta(days=1)).date()
    elif "today" in lower or "आज" in lower:
        target_date = now.date()

    # Check weekday names
    weekdays = {
        "monday": 0, "lunes": 0, "lundi": 0, "montag": 0,
        "tuesday": 1, "martes": 1, "mardi": 1, "dienstag": 1,
        "wednesday": 2, "miércoles": 2, "mercredi": 2, "mittwoch": 2,
        "thursday": 3, "jueves": 3, "jeudi": 3, "donnerstag": 3,
        "friday": 4, "viernes": 4, "vendredi": 4, "freitag": 4,
    }
    for name, wd in weekdays.items():
        if name in lower:
            days_ahead = (wd - now.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = (now + timedelta(days=days_ahead)).date()
            break

    # Extract explicit hour
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
            hr += 12  # Assume business hours afternoon (1pm-7pm)
        if 8 <= hr <= 19:
            target_hour = hr
    elif "morning" in lower or "mañana" in lower or "matin" in lower or "सुबह" in lower:
        target_hour = 10
    elif "afternoon" in lower or "tarde" in lower or "après-midi" in lower or "दोपहर" in lower:
        target_hour = 14

    if target_date is not None or target_hour is not None:
        best_slot = None
        for slot in available_slots:
            dt = datetime.fromisoformat(slot.replace("Z", ""))
            date_ok = (target_date is None) or (dt.date() == target_date)
            hour_ok = (target_hour is None) or (abs(dt.hour - target_hour) <= 1)
            if date_ok and hour_ok:
                return slot
            if date_ok and best_slot is None:
                best_slot = slot
        if best_slot:
            return best_slot
        # Construct custom conflict-free slot if user asked for specific date/hour
        chosen_date = target_date or (now + timedelta(days=1)).date()
        chosen_hour = target_hour or 14
        return datetime(chosen_date.year, chosen_date.month, chosen_date.day, chosen_hour, 0).isoformat() + "Z"

    # Generic confirmation ("yes book it", "that works", "book the demo")
    if any(k in lower for k in ["yes", "sure", "works", "book it", "confirm", "schedule it", "agendar", "sí", "oui", "ja", "हाँ", "बुक"]):
        return available_slots[0]

    return None
