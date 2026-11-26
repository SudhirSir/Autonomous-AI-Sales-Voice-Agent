"""Multilingual NLP (EN/HI/Hinglish/ES/FR/DE), Contextual Intent, Sentiment vs Intent, BANT & Timezone Engine."""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone, tzinfo
from typing import Optional
from zoneinfo import ZoneInfo

from app.config import settings
from app.models import (
    BANTProfile,
    IntentClassification,
    IntentType,
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
    "karein", "karo", "mil", "sakta", "sakti",
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

TIMEZONE_ALIASES: dict[str, str] = {
    "utc": "UTC",
    "gmt": "UTC",
    "ist": "Asia/Kolkata",
    "india": "Asia/Kolkata",
    "est": "America/New_York",
    "edt": "America/New_York",
    "et": "America/New_York",
    "pst": "America/Los_Angeles",
    "pdt": "America/Los_Angeles",
    "pt": "America/Los_Angeles",
    "mst": "America/Denver",
    "mdt": "America/Denver",
    "cet": "Europe/Berlin",
    "cest": "Europe/Berlin",
    "bst": "Europe/London",
    "jst": "Asia/Tokyo",
    "sgt": "Asia/Singapore",
    "aest": "Australia/Sydney",
}

AMBIGUOUS_TZ_ABBREVS = {"cst", "cdt", "ist_ambig"}  # CST can be Central Standard (US) vs China Standard


def resolve_zoneinfo(tz_name: Optional[str]) -> tzinfo:
    """Resolve IANA name or alias safely to ZoneInfo/timezone."""
    raw = (tz_name or "UTC").strip()
    if raw.lower() in TIMEZONE_ALIASES:
        raw = TIMEZONE_ALIASES[raw.lower()]
    try:
        return ZoneInfo(raw)
    except Exception:
        return timezone.utc


def extract_timezone_from_text(text: str, default_tz: str = "UTC") -> tuple[str, bool]:
    """Detect natural-language timezone mentions. Returns (tz_name, is_ambiguous)."""
    lower = text.lower()
    if re.search(r"\bcst\b", lower):
        return default_tz, True

    m_offset = re.search(r"(?:utc|gmt)?\s*([+-])(\d{1,2}):?(\d{2})?", lower)
    if m_offset and ("+05:30" in lower or "+5:30" in lower):
        return "Asia/Kolkata", False

    for alias, iana in TIMEZONE_ALIASES.items():
        if re.search(rf"\b{re.escape(alias)}\b", lower):
            return iana, False

    for iana in ("Asia/Kolkata", "America/New_York", "America/Los_Angeles", "Europe/London", "Europe/Berlin", "UTC"):
        if iana.lower() in lower:
            return iana, False

    if default_tz == "ambiguous":
        return "UTC", True
    return default_tz, False


def detect_language(text: str, fallback: SupportedLang = "en") -> SupportedLang:
    """Detect language among en, hi, hinglish, es, fr, de using script and lexical cues."""
    if not text or not text.strip():
        return fallback

    if re.search(r"[\u0900-\u097F]", text):
        return "hi"

    lower = text.lower()
    tokens = set(re.findall(r"[a-zA-Záéíóúñüàâçéèêëîïôùûüÿäöß]+", lower))

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


# ---------------- Contextual Intent Classification (Phase 6) ----------------

def classify_intent(
    text: str,
    awaiting_slot_choice: bool = False,
    has_existing_appt: bool = False,
) -> IntentClassification:
    """Classify buyer utterance into one of 12 contextual intents without false booking triggers."""
    lower = text.lower().strip()
    if not lower:
        return IntentClassification(intent="general_conversation", confidence=0.5, reasoning="Empty input")

    # 1. Opt-out / Disqualification
    if any(
        p in lower
        for p in [
            "wrong number", "not interested", "stop calling", "do not call",
            "remove me", "unsubscribe", "opt out", "mat call karo", "band karo",
        ]
    ):
        return IntentClassification(
            intent="opt_out",
            confidence=0.95,
            requires_confirmation=False,
            reasoning="Explicit opt-out or disinterest phrase detected.",
        )

    # 2. Explicit rejection of booking (must NOT classify as booking_request)
    if re.search(
        r"\b(not ready to book|don't want to book|do not want to book|no demo|not looking to schedule|"
        r"not booking|maybe later|skip the demo|abhi book nahi|nahi book karna)\b",
        lower,
    ):
        return IntentClassification(
            intent="general_conversation",
            confidence=0.9,
            requires_confirmation=False,
            reasoning="Buyer explicitly declined or deferred booking.",
        )

    # 3. Cancellation request
    if any(
        k in lower
        for k in [
            "cancel appointment", "cancel the demo", "cancel my demo", "cancel meeting",
            "cancel the meeting", "cancel booking", "cancelar cita", "annuler",
            "stornieren", "demo cancel", "रद्द",
        ]
    ):
        return IntentClassification(
            intent="cancellation_request",
            confidence=0.92,
            requires_confirmation=not has_existing_appt,
            reasoning="Explicit appointment cancellation language.",
        )

    # 4. Reschedule request
    if any(
        k in lower
        for k in [
            "reschedule", "move my appointment", "move the appointment", "move the demo",
            "change the time", "push back", "postpone", "reprogramar", "verschieben",
            "shift the meeting", "time change",
        ]
    ):
        return IntentClassification(
            intent="reschedule_request",
            confidence=0.92,
            requires_confirmation=False,
            reasoning="Explicit appointment rescheduling language.",
        )

    # 5. Explanatory question ABOUT demo/product (must NOT trigger booking!)
    if re.search(
        r"\b(how does (?:your |the |a )?demo work|how do demos work|what is (?:in |included in )?(?:a |the |your )?demo|"
        r"what happens (?:in|during) (?:a |the )?demo|tell me about (?:your |the )?demo|explain (?:your |the )?demo|"
        r"how long is (?:the |a )?demo|what does fortivex do|how does fortivex work)\b",
        lower,
    ):
        return IntentClassification(
            intent="product_question",
            confidence=0.93,
            requires_confirmation=False,
            reasoning="Informational question about how the product or demo works (not a booking request).",
        )

    # 6. Availability request (asking about open slots without committing yet)
    if re.search(
        r"\b(do you have (?:any )?(?:slots|availability|openings|times)|what slots|when are you (?:free|available)|"
        r"available slots|open slots|check availability|slots on|times on|kab free|kaunse slots)\b",
        lower,
    ):
        return IntentClassification(
            intent="availability_request",
            confidence=0.9,
            requires_confirmation=False,
            reasoning="Prospect asked about calendar availability without committing to a slot.",
        )

    # 7. Booking confirmation when awaiting slot selection
    slot_selection_cues = [
        "option 1", "option 2", "option 3", "slot 1", "slot 2", "slot 3", "#1", "#2", "#3",
        "first one", "second one", "third one", "book it", "confirm", "schedule it", "works great",
        "lock it in", "haan book", "theek hai book",
    ]
    if awaiting_slot_choice and (
        any(c in lower for c in slot_selection_cues)
        or re.search(r"\b\d{1,2}\s*(?:am|pm)\b", lower)
        or any(d in lower for d in ["tomorrow", "monday", "tuesday", "wednesday", "thursday", "friday", "kal", "mañana", "demain", "morgen"])
    ):
        return IntentClassification(
            intent="booking_confirmation",
            confidence=0.91,
            requires_confirmation=False,
            reasoning="Slot confirmation or selection during active booking prompt.",
        )

    # 8. Genuine booking request
    booking_patterns = [
        r"\b(?:book|schedule|set up|arrange|reserve|agendar|planifier|réserver|buchen)\b.{0,28}\b(?:demo|call|meeting|session|appointment|cita|termin|démo|डेमो)\b",
        r"\b(?:can i|could i|i want to|i'd like to|let's|lets|please|want to|chahte hain|chahta hoon|karna hai)\b.{0,28}\b(?:see a demo|book a demo|schedule a demo|demo book|डेमो बुक)\b",
        r"\b(?:demo)\b.{0,20}\b(?:tomorrow|today|monday|tuesday|wednesday|thursday|friday|this week|next week|at \d|kal|सुबह|बजे|mañana|demain|morgen)\b",
        r"(?:डेमो|मीटिंग)\s*(?:बुक|शेड्यूल|तय)",
    ]
    if any(re.search(p, lower) for p in booking_patterns):
        return IntentClassification(
            intent="booking_request",
            confidence=0.91,
            requires_confirmation=False,
            reasoning="Actionable request to schedule a demo or meeting.",
        )

    # 9. Follow-up request
    if re.search(r"\b(send me|email me|follow up|follow-up|call me back|reach out later|brochure|deck)\b", lower):
        return IntentClassification(
            intent="follow_up_request",
            confidence=0.84,
            reasoning="Prospect requested follow-up material or callback.",
        )

    # 10. Objections
    if re.search(r"\b(too expensive|overpriced|already use|already have|worried about|concerned about|not sure if)\b", lower):
        return IntentClassification(
            intent="objection",
            confidence=0.85,
            reasoning="Commercial, incumbent, or implementation objection detected.",
        )

    # 11. Pricing questions
    if any(
        k in lower
        for k in ["price", "pricing", "cost", "how much", "plan", "discount", "tier", "precio", "precios", "prix", "tarif", "preis", "preise", "कीमत", "दाम", "keemat", "kitna"]
    ):
        return IntentClassification(
            intent="pricing_question",
            confidence=0.88,
            reasoning="Question regarding pricing, plans, or discounts.",
        )

    # 12. Feature / Security / Integration questions
    if any(
        k in lower
        for k in [
            "soc2", "soc 2", "gdpr", "hipaa", "iso", "security", "privacy", "encrypt",
            "integrat", "salesforce", "hubspot", "gong", "chorus", "drift", "sla",
            "uptime", "feature", "language", "voice", "deploy", "onboard", "seguridad",
            "sécurité", "sicherheit", "सुरक्षा", "suraksha",
        ]
    ):
        return IntentClassification(
            intent="feature_question",
            confidence=0.86,
            reasoning="Question regarding platform capabilities, security compliance, or integrations.",
        )

    if "?" in text or any(lower.startswith(w) for w in ("what ", "how ", "why ", "when ", "where ", "do you ", "does ", "can you ", "is ")):
        return IntentClassification(
            intent="product_question",
            confidence=0.78,
            reasoning="General product or capability question.",
        )

    return IntentClassification(
        intent="general_conversation",
        confidence=0.7,
        reasoning="Discovery or conversational turn.",
    )


# ---------------- Calibrated Multilingual Sentiment vs Intent (Phase 8) ----------------

POSITIVE_TERMS = {
    "great", "awesome", "love", "interested", "perfect", "helpful", "impressive",
    "excited", "ready", "sounds good", "definitely", "excellent", "fast", "good",
    "genial", "perfecto", "excelente", "interesado", "bueno", "listo",
    "parfait", "génial", "intéressé", "super", "prêt",
    "perfekt", "interessiert", "ausgezeichnet", "bereit",
    "अच्छा", "बढ़िया", "सही", "तैयार", "पसंद", "शानदार",
    "badhiya", "accha", "badiya", "theek",
}

FRUSTRATION_TERMS = {
    "frustrated", "annoying", "waste of time", "useless", "terrible", "ridiculous",
    "horrible", "stop calling", "fed up", "pathetic", "angry", "irritated",
    "बेकार", "परेशान", "बकवास", "frustré", "genervt", "molesto",
}

NEGATIVE_TERMS = {
    "expensive", "costly", "too much", "bad", "hate", "problem", "slow",
    "concerned", "worried", "skeptical", "no budget", "cancel", "overpriced",
    "caro", "problema", "preocupado", "difícil", "lento",
    "cher", "problème", "inquiet", "difficile",
    "teuer", "besorgt", "schwierig", "skeptisch",
    "महंगा", "समस्या", "चिंता", "मुश्किल",
    "mehnga", "mushkil",
}

NEGATION_WORDS = {"not", "no", "never", "neither", "don't", "dont", "doesn't", "didn't", "isn't", "aren't", "nahi", "nhi", "नहीं", "nunca", "pas", "nicht"}
SARCASM_PATTERNS = [
    r"\boh great\b.*\b(another|again|wait|slow|spam|broken|useless)\b",
    r"\byeah right\b",
    r"\bjust what i needed\b",
    r"\blove waiting\b",
    r"\bwonderful,?\s+(?:another|more)\b",
]
URGENT_TERMS = {
    "asap", "urgent", "immediately", "this week", "this month", "right away",
    "q1", "q2", "deadline", "urgente", "inmediato", "sofort", "dringend",
    "तुरंत", "जल्दी", "abhi", "jaldi",
}


def _count_with_negation(lower: str, terms: set[str]) -> tuple[int, int, bool]:
    """Count unnegated vs negated occurrences of sentiment terms."""
    tokens = re.findall(r"[\w'\u0900-\u097F]+", lower)
    unnegated = 0
    negated = 0
    neg_adjusted = False
    for term in sorted(terms, key=len, reverse=True):
        if " " in term:
            if term in lower:
                idx = lower.find(term)
                prefix = lower[max(0, idx - 22) : idx].split()
                if any(w in NEGATION_WORDS for w in prefix[-3:]):
                    negated += 1
                    neg_adjusted = True
                else:
                    unnegated += 1
        else:
            for i, tok in enumerate(tokens):
                if tok == term:
                    window = tokens[max(0, i - 3) : i]
                    if any(w in NEGATION_WORDS for w in window):
                        negated += 1
                        neg_adjusted = True
                    else:
                        unnegated += 1
    return unnegated, negated, neg_adjusted


def analyze_sentiment(text: str, intent_cls: Optional[IntentClassification] = None) -> SentimentResult:
    """Analyze emotional sentiment separately from commercial purchase intent with negation & sarcasm handling."""
    lower = text.lower()
    intent_obj = intent_cls or classify_intent(text)

    pos_hits, pos_negated, neg_adj1 = _count_with_negation(lower, POSITIVE_TERMS)
    neg_hits, neg_negated, neg_adj2 = _count_with_negation(lower, NEGATIVE_TERMS)
    frust_hits, frust_negated, neg_adj3 = _count_with_negation(lower, FRUSTRATION_TERMS)
    negation_adjusted = neg_adj1 or neg_adj2 or neg_adj3

    # A negated frustration ("I'm not frustrated") or negated negative ("not expensive") softens negativity
    effective_pos = pos_hits + (neg_negated * 0.5)
    effective_neg = neg_hits + (pos_negated * 0.8)
    sarcasm_detected = any(re.search(pat, lower) for pat in SARCASM_PATTERNS)
    if sarcasm_detected:
        effective_pos = max(0.0, effective_pos - 1.5)
        effective_neg += 1.5

    urg_hits = sum(1 for t in URGENT_TERMS if t in lower)

    signals: list[str] = []
    if intent_obj.intent in {"booking_request", "booking_confirmation"}:
        signals.append("booking_intent")
    elif intent_obj.intent == "availability_request":
        signals.append("availability_inquiry")
    elif intent_obj.intent == "reschedule_request":
        signals.append("reschedule_intent")
    elif intent_obj.intent == "cancellation_request":
        signals.append("cancel_intent")

    has_price = any(
        k in lower
        for k in [
            "price", "pricing", "cost", "expensive", "discount", "budget",
            "precio", "caro", "prix", "cher", "preis", "teuer", "कीमत", "महंगा",
            "keemat", "paisa", "mehnga",
        ]
    )
    price_objection = bool(has_price and (neg_hits > 0 or frust_hits > 0 or "too much" in lower or "overpriced" in lower))
    if has_price:
        signals.append("price_discussion")
        if price_objection:
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

    competitor_objection = any(k in lower for k in ["gong", "chorus", "drift", "intercom", "clari", "competitor", "in-house"])
    if competitor_objection:
        signals.append("competitor_comparison")

    has_buying = (
        any(
            k in lower
            for k in [
                "ready", "sign", "pilot", "trial", "start", "move forward", "go ahead",
                "roll out", "purchase", "buy", "listo", "prêt", "bereit", "तैयार", "chahiye",
            ]
        )
        and intent_obj.intent != "opt_out"
        and "not ready" not in lower
    )
    if has_buying:
        signals.append("buying_signal")
    if urg_hits > 0:
        signals.append("urgency")

    raw_polarity = (effective_pos * 0.35) - (effective_neg * 0.35) - (frust_hits * 0.55)
    polarity = max(-1.0, min(1.0, round(raw_polarity, 2)))
    frustration_detected = frust_hits > 0 or polarity <= -0.55
    mixed_sentiment = (effective_pos > 0 and (effective_neg > 0 or frust_hits > 0))

    if frustration_detected:
        label = "Frustrated"
    elif sarcasm_detected or (effective_neg > 0 and effective_pos == 0):
        label = "Skeptical" if polarity > -0.4 else "Negative"
    elif mixed_sentiment and abs(polarity) < 0.25:
        label = "Mixed"
    elif polarity >= 0.22:
        label = "Positive"
    elif polarity <= -0.22:
        label = "Negative"
    else:
        label = "Neutral"

    # Commercial Purchase Intent (0-100), separated from emotional polarity
    intent_score = 18
    if has_buying:
        intent_score += 32
    if intent_obj.intent in {"booking_request", "booking_confirmation"}:
        intent_score += 30
    elif intent_obj.intent == "availability_request":
        intent_score += 18
    if urg_hits > 0:
        intent_score += 12
    if has_price and not price_objection:
        intent_score += 10
    elif price_objection and has_buying:
        intent_score += 10
    if intent_obj.intent == "opt_out" or "no budget" in lower or "not interested" in lower:
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
    confidence = min(0.97, round(0.68 + 0.06 * (effective_pos + effective_neg + frust_hits + len(signals)), 2))

    return SentimentResult(
        polarity=polarity,
        label=label,  # type: ignore[arg-type]
        confidence=confidence,
        intent_score=intent_score,
        purchase_readiness=readiness,  # type: ignore[arg-type]
        frustration_detected=frustration_detected,
        trust_concern=trust_concern,
        price_objection=price_objection,
        competitor_objection=competitor_objection,
        sarcasm_detected=sarcasm_detected,
        mixed_sentiment=mixed_sentiment,
        negation_adjusted=negation_adjusted,
        engagement_level=engagement,  # type: ignore[arg-type]
        signals=signals,
    )


# ---------------- BANT Memory Extraction & Calibration (Phase 8) ----------------

def extract_memory_and_bant(
    text: str,
    memory: LeadMemory,
    bant: BANTProfile,
) -> tuple[LeadMemory, BANTProfile]:
    """Update contextual LeadMemory and BANTProfile with evidence, confidence, and self-correction."""
    mem = memory.model_copy(deep=True)
    b = bant.model_copy(deep=True)
    lower = text.lower()
    is_correction = any(k in lower for k in ["actually", "correction", "instead", "sorry,", "i meant", "not $", "rather than"])

    # 1. Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text)
    if email_match:
        mem.email = email_match.group(0).rstrip(".,!?;:").lower()

    # 2. Phone number
    phone_match = re.search(
        r"(?<!\$)(?:\+\d{1,3}[\s.-]?)?(?:\(\d{3}\)\s*\d{3}[\s.-]?\d{4}|\b\d{3}[\s.-]?\d{3}[\s.-]?\d{4}\b|\b\d{5}[\s.-]?\d{5}\b)",
        text,
    )
    if phone_match:
        mem.phone = phone_match.group(0).strip()

    # 3. Timezone hint
    tz_detected, tz_ambig = extract_timezone_from_text(text, default_tz=mem.timezone or "UTC")
    if not tz_ambig and tz_detected:
        mem.timezone = tz_detected

    # 4. Name extraction
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

    # 5. Company extraction
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

    # 6. Role / Authority (allows explicit self-correction)
    authority_patterns = [
        (r"\b(ceo|cto|cfo|cro|coo|founder|co-founder|president|vp|vice president)\b", 25, "Executive Decision Maker", 0.92),
        (r"\b(head of|director|svp|evp|general manager|gm)\b", 22, "Department Director / Head", 0.88),
        (r"\b(manager|lead|revops|sales ops|team lead|architect)\b", 16, "Team Manager / Evaluator", 0.82),
        (r"\b(rep|sdr|bdr|analyst|developer|intern|student)\b", 10, "Individual Contributor / Researcher", 0.78),
    ]
    for pat, score, label, conf in authority_patterns:
        m = re.search(pat, lower)
        if m:
            role_txt = m.group(0).upper() if len(m.group(0)) <= 4 else m.group(0).title()
            mem.role = role_txt
            if is_correction or score > b.authority_score:
                b.authority = f"{role_txt} ({label})"
                b.authority_score = score
                b.authority_evidence = m.group(0)
                b.authority_confidence = conf
            break
    if any(k in lower for k in ["i decide", "decision maker", "sign off", "my team", "approve", "tomo la decisión", "je décide"]):
        b.authority_score = max(b.authority_score, 23)
        b.authority_evidence = b.authority_evidence or "Confirmed decision authority"
        b.authority_confidence = max(b.authority_confidence, 0.88)
        if not b.authority:
            b.authority = "Confirmed Decision Maker"

    # 7. Team size
    size_match = re.search(
        r"\b(\d{1,5})\s+(?:[a-z]+\s+)?(?:reps|seats|users|people|agents|sellers|employees|personas|utilisateurs|mitarbeiter|लोगों?|log)",
        lower,
    )
    if size_match:
        mem.team_size = int(size_match.group(1))

    # 8. Budget extraction (with self-correction)
    has_budget_word = any(
        k in lower
        for k in ["budget", "spend", "pricing", "cost", "afford", "allocated", "presupuesto", "बजट", "$", "€"]
    )
    budget_extracted = False
    if has_budget_word:
        for mm in re.finditer(r"(\$|€|usd|eur)?\s*(\d{1,3}(?:,\d{3})*|\d+(?:\.\d+)?)\s*(k|m|thousand|million)?\b", lower):
            # Ignore negated previous number in correction like "not $40k"
            prefix = lower[max(0, mm.start() - 10) : mm.start()]
            if is_correction and re.search(r"\bnot\s*$", prefix):
                continue
            num = float(mm.group(2).replace(",", ""))
            unit = (mm.group(3) or "").lower()
            if unit in {"k", "thousand"}:
                num *= 1000
            elif unit in {"m", "million"}:
                num *= 1_000_000
            if num >= 500:
                b.budget = f"~${int(num):,}"
                b.budget_score = 25 if num >= 15000 else (20 if num >= 5000 else 14)
                b.budget_evidence = mm.group(0).strip()
                b.budget_confidence = 0.9
                budget_extracted = True
                break
    if not budget_extracted:
        if any(k in lower for k in ["no budget", "zero budget", "free only", "sin presupuesto", "0 budget"]):
            b.budget = "No allocated budget"
            b.budget_score = 3
            b.budget_evidence = "Explicit zero/no budget statement"
            b.budget_confidence = 0.92
        elif any(k in lower for k in ["budget approved", "budget is allocated", "we have budget", "presupuesto aprobado", "budget approuvé"]):
            b.budget = b.budget or "Approved budget allocated"
            b.budget_score = max(b.budget_score, 22)
            b.budget_evidence = "Approved budget confirmed"
            b.budget_confidence = 0.85
        elif any(k in lower for k in ["growth plan", "scale plan", "enterprise plan"]):
            b.budget = b.budget or "Evaluating paid tier"
            b.budget_score = max(b.budget_score, 18)
            b.budget_evidence = "Evaluated tier alignment"
            b.budget_confidence = 0.72

    # 9. Need & Pain points
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
            b.need_evidence = ", ".join(mem.pain_points[:2])
            b.need_confidence = 0.88
        else:
            b.need = b.need or "Evaluating AI voice sales automation"
            b.need_score = max(b.need_score, 14)
            b.need_evidence = "General need stated"
            b.need_confidence = 0.65

    # 10. Timeline extraction (with self-correction)
    timeline_rules = [
        (r"\b(this week|immediately|asap|right away|today|tomorrow|urgente|sofort|तुरंत|abhi)\b", "Immediate (0–2 weeks)", 25, 0.92),
        (r"\b(this month|next week|2 weeks|30 days|q1|q2|q3|q4|next month|este mes|ce mois|is mahine)\b", "Near-term (1–3 months)", 22, 0.88),
        (r"\b(quarter|60 days|90 days|this year|soon|pronto|bientôt|bald)\b", "Mid-term (1–2 quarters)", 17, 0.78),
        (r"\b(next year|just exploring|just looking|no rush|solo mirando)\b", "Exploratory / Long-term", 8, 0.85),
    ]
    for pat, label, score, conf in timeline_rules:
        m_tl = re.search(pat, lower)
        if m_tl:
            prefix = lower[max(0, m_tl.start() - 8) : m_tl.start()]
            if is_correction and re.search(r"\bnot\s*$", prefix):
                continue
            if is_correction or score > b.timeline_score:
                b.timeline = label
                b.timeline_score = score
                b.timeline_evidence = m_tl.group(0)
                b.timeline_confidence = conf
            break

    # 11. Objections & Competitors (respecting negation)
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
        if kw in lower and f"not {kw}" not in lower and obj not in mem.objections:
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
    """Deterministic, explainable BANT scoring (0-100).

    Note: Booking a demo updates stage & lead_priority, NEVER artificially inflating BANT score.
    Missing BANT fields reduce completeness rather than causing automatic disqualification.
    """
    lower = last_utterance.lower()
    disqualify_phrases = [
        "wrong number", "not interested", "stop calling", "do not call",
        "remove me", "student project", "personal hobby", "free forever only",
    ]
    explicit_disqualify = any(p in lower for p in disqualify_phrases) or (
        bant.budget == "No allocated budget" and bant.timeline_score <= 8 and not has_booking and bool(bant.budget_evidence)
    )

    rationale: list[str] = [
        f"Budget: {bant.budget_score}/25 ({bant.budget or 'missing'})",
        f"Authority: {bant.authority_score}/25 ({bant.authority or 'missing'})",
        f"Need: {bant.need_score}/25 ({bant.need or 'missing'})",
        f"Timeline: {bant.timeline_score}/25 ({bant.timeline or 'missing'})",
    ]
    evidence_summary = {
        k: v
        for k, v in {
            "budget": bant.budget_evidence or "",
            "authority": bant.authority_evidence or "",
            "need": bant.need_evidence or "",
            "timeline": bant.timeline_evidence or "",
        }.items()
        if v
    }
    confs = [c for c in (bant.budget_confidence, bant.authority_confidence, bant.need_confidence, bant.timeline_confidence) if c > 0]
    avg_conf = round(sum(confs) / len(confs), 2) if confs else 0.4

    if explicit_disqualify:
        return LeadQualification(
            score=min(15, bant.budget_score + bant.authority_score + bant.need_score + bant.timeline_score),
            tier="Disqualified",
            stage="Disqualified",
            lead_priority="Disqualified",
            confidence=max(0.85, avg_conf),
            bant=bant,
            missing_dimensions=["budget", "need", "timeline"],
            estimated_acv_usd=0,
            score_rationale=rationale + ["Disqualified: explicit non-fit or zero-budget opt-out."],
            evidence_summary=evidence_summary,
        )

    # Purely calibrated BANT score (no automatic booking inflation)
    total = min(100, bant.budget_score + bant.authority_score + bant.need_score + bant.timeline_score)

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

    if total >= 75 or (has_booking and total >= 55):
        priority = "P1-Urgent"
    elif total >= 55 or has_booking:
        priority = "P2-High"
    elif total >= 30:
        priority = "P3-Standard"
    else:
        priority = "P4-Low"

    if bant.budget and "$" in bant.budget:
        digits = re.sub(r"[^\d]", "", bant.budget)
        acv = int(digits) if digits else 0
    elif memory.team_size:
        acv = memory.team_size * 300
    else:
        acv = 0

    return LeadQualification(
        score=total,
        tier=tier,  # type: ignore[arg-type]
        stage=stage,  # type: ignore[arg-type]
        lead_priority=priority,  # type: ignore[arg-type]
        confidence=avg_conf,
        bant=bant,
        missing_dimensions=missing,
        estimated_acv_usd=acv,
        score_rationale=rationale,
        evidence_summary=evidence_summary,
    )


# ---------------- Timezone-Aware & DST-Safe Slot Validation (Phase 9) ----------------

def convert_slot_to_local(slot_iso: str, tz_name: str = "UTC") -> str:
    """Convert a UTC ISO timestamp to formatted local time string."""
    dt_utc = datetime.fromisoformat(slot_iso.replace("Z", "+00:00"))
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=timezone.utc)
    zone = resolve_zoneinfo(tz_name)
    dt_local = dt_utc.astimezone(zone)
    tz_label = dt_local.tzname() or tz_name
    return dt_local.strftime(f"%a %b %d, %I:%M %p {tz_label}")


def validate_business_slot(slot_iso: str, tz_name: str = "UTC") -> tuple[Optional[str], Optional[str]]:
    """Validate requested ISO datetime in business timezone (DST-safe) and normalize to UTC ISO."""
    try:
        clean = slot_iso.strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=resolve_zoneinfo(tz_name))
        dt_utc = dt.astimezone(timezone.utc)
    except Exception:
        return None, f"Invalid date/time format: '{slot_iso}'. Please provide a valid ISO date/time."

    now_utc = datetime.now(timezone.utc)
    if dt_utc <= now_utc + timedelta(minutes=15):
        return None, "Requested slot is in the past. Please choose an upcoming weekday slot."

    biz_tz = resolve_zoneinfo(settings.business_timezone)
    dt_biz = dt_utc.astimezone(biz_tz)
    biz_label = dt_biz.tzname() or settings.business_timezone

    if dt_biz.weekday() >= 5:
        return (
            None,
            f"Demos are only available Monday through Friday during business hours "
            f"({settings.business_hours_start:02d}:00–{settings.business_hours_end:02d}:00 {biz_label}).",
        )

    if not (settings.business_hours_start <= dt_biz.hour < settings.business_hours_end):
        return (
            None,
            f"Requested hour ({dt_biz.strftime('%H:%M')} {biz_label}) is outside business hours "
            f"({settings.business_hours_start:02d}:00–{settings.business_hours_end:02d}:00 {biz_label}).",
        )

    normalized = dt_utc.replace(second=0, microsecond=0).isoformat().replace("+00:00", "Z")
    return normalized, None


def parse_requested_slot(
    text: str, available_slots: list[str], tz_name: str = "UTC"
) -> tuple[Optional[str], Optional[str]]:
    """Parse natural-language time/slot preference with timezone conversion and business-hour checks."""
    lower = text.lower()
    detected_tz, is_ambiguous = extract_timezone_from_text(text, default_tz=tz_name or "UTC")
    customer_zone = resolve_zoneinfo(detected_tz)

    if available_slots:
        if any(k in lower for k in ["first", "option 1", "slot 1", "#1"]):
            return available_slots[0], None
        if len(available_slots) > 1 and any(k in lower for k in ["second", "option 2", "slot 2", "#2"]):
            return available_slots[1], None
        if len(available_slots) > 2 and any(k in lower for k in ["third", "option 3", "slot 3", "#3"]):
            return available_slots[2], None

    if any(k in lower for k in ["yesterday", "last week"]):
        return None, "Cannot schedule an appointment in the past. Please choose an upcoming weekday."

    now_local = datetime.now(customer_zone)
    target_date = None

    if re.search(r"\b(?:tomorrow|mañana|demain|morgen|kal)\b", lower) or "कल" in lower:
        cand = now_local + timedelta(days=1)
        if cand.weekday() >= 5:
            cand = cand + timedelta(days=(7 - cand.weekday()))
        target_date = cand.date()
    elif "today" in lower or "आज" in lower:
        target_date = now_local.date()

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
            days_ahead = (wd - now_local.weekday()) % 7
            if days_ahead == 0:
                days_ahead = 7
            target_date = (now_local + timedelta(days=days_ahead)).date()
            break

    target_hour: Optional[int] = None
    target_minute: int = 0
    hm = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", lower)
    if hm:
        hr = int(hm.group(1))
        mn = int(hm.group(2) or 0)
        ampm = hm.group(3)
        if ampm == "pm" and hr < 12:
            hr += 12
        elif ampm == "am" and hr == 12:
            hr = 0
        elif not ampm and 1 <= hr <= 7:
            hr += 12
        target_hour = hr
        target_minute = mn
    elif re.search(r"\b(?:morning|mañana|matin|subah)\b", lower) or "सुबह" in lower:
        target_hour = 10
    elif re.search(r"\b(?:afternoon|tarde|après-midi|shaam)\b", lower) or "दोपहर" in lower:
        target_hour = 14

    if target_date is not None or target_hour is not None:
        if is_ambiguous and target_hour is not None:
            return None, "Please confirm your timezone (e.g., UTC, IST, EST, PST) before I lock in this time slot."

        chosen_date = target_date or (now_local + timedelta(days=1)).date()
        chosen_hour = target_hour if target_hour is not None else 14
        if chosen_hour >= 24 or target_minute >= 60:
            return None, "Invalid hour or minute provided. Please choose a valid time."

        local_dt = datetime(
            chosen_date.year, chosen_date.month, chosen_date.day, chosen_hour, target_minute, tzinfo=customer_zone
        )
        candidate_utc_iso = local_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

        # Match against precomputed available slots if same UTC hour/date
        for slot in available_slots:
            if slot == candidate_utc_iso:
                return slot, None
            if target_hour is None:
                dt_s = datetime.fromisoformat(slot.replace("Z", "+00:00")).astimezone(customer_zone)
                if dt_s.date() == chosen_date:
                    return slot, None

        return validate_business_slot(candidate_utc_iso, tz_name=detected_tz)

    if available_slots and any(
        k in lower for k in ["yes", "sure", "works", "book it", "confirm", "schedule it", "agendar", "sí", "oui", "ja", "हाँ", "बुक", "theek"]
    ):
        return available_slots[0], None

    return None, None
