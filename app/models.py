"""Pydantic data models for Fortivex AI Sales Voice Agent."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


SupportedLang = Literal["en", "es", "fr", "de", "hi"]


class BANTProfile(BaseModel):
    """Structured BANT lead qualification dimensions."""

    budget: Optional[str] = None
    budget_score: int = Field(default=0, ge=0, le=25)
    authority: Optional[str] = None
    authority_score: int = Field(default=0, ge=0, le=25)
    need: Optional[str] = None
    need_score: int = Field(default=0, ge=0, le=25)
    timeline: Optional[str] = None
    timeline_score: int = Field(default=0, ge=0, le=25)


class LeadQualification(BaseModel):
    """Computed lead score, tier, and next qualification focus."""

    score: int = Field(default=0, ge=0, le=100)
    tier: Literal["Hot", "Warm", "Cold", "Disqualified"] = "Cold"
    stage: Literal["Discovery", "Qualification", "Proposal", "Demo Scheduled", "Nurture", "Closed-Lost"] = "Discovery"
    bant: BANTProfile = Field(default_factory=BANTProfile)
    missing_dimensions: list[str] = Field(default_factory=lambda: ["budget", "authority", "need", "timeline"])
    estimated_acv_usd: int = 0


class SentimentResult(BaseModel):
    """Per-turn sentiment and buying-signal analysis."""

    polarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    label: Literal["Positive", "Neutral", "Skeptical", "Negative", "Urgent"] = "Neutral"
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    signals: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """Knowledge base chunk returned by the hybrid RAG retriever."""

    doc_id: str
    title: str
    category: str
    content: str
    score: float = Field(ge=0.0, le=1.0)
    matched_terms: list[str] = Field(default_factory=list)


class LeadMemory(BaseModel):
    """Contextual memory extracted and updated across conversation turns."""

    lead_id: Optional[str] = None
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    team_size: Optional[int] = None
    preferred_language: SupportedLang = "en"
    pain_points: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    competitors_mentioned: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    prior_calls_count: int = 0
    returning_lead: bool = False


class ConversationMessage(BaseModel):
    """Single message turn stored in session history."""

    id: Optional[int] = None
    session_id: str
    role: Literal["user", "assistant", "system"]
    text: str
    language: SupportedLang = "en"
    sentiment: Optional[SentimentResult] = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class AppointmentCreate(BaseModel):
    """Request to book an appointment."""

    session_id: Optional[str] = None
    lead_id: Optional[str] = None
    lead_name: str
    lead_email: Optional[str] = None
    company: Optional[str] = None
    slot_iso: str
    duration_minutes: int = 30
    topic: str = "Fortivex Product Demo & Architecture Review"
    timezone: str = "UTC"


class AppointmentRecord(BaseModel):
    """Persisted appointment record."""

    id: str
    lead_id: Optional[str] = None
    session_id: Optional[str] = None
    lead_name: str
    lead_email: Optional[str] = None
    company: Optional[str] = None
    slot_iso: str
    duration_minutes: int = 30
    topic: str
    timezone: str = "UTC"
    status: Literal["Confirmed", "Rescheduled", "Cancelled"] = "Confirmed"
    meeting_link: str
    created_at: str


class TurnRequest(BaseModel):
    """Input payload for a single voice/text interaction turn."""

    session_id: str
    text: str
    language_override: Optional[SupportedLang] = None
    input_mode: Literal["voice", "text"] = "voice"


class TurnResponse(BaseModel):
    """Complete agent response with NLU, RAG, CRM, and voice metadata."""

    session_id: str
    reply_text: str
    language: SupportedLang
    tts_locale: str
    sentiment: SentimentResult
    qualification: LeadQualification
    memory: LeadMemory
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    booked_appointment: Optional[AppointmentRecord] = None
    proposed_slots: list[str] = Field(default_factory=list)
    crm_synced: bool = False
    webhook_events: list[dict[str, Any]] = Field(default_factory=list)
    ai_engine: Literal["gemini", "local-hybrid"] = "local-hybrid"
    latency_ms: int = 0


class CallSummary(BaseModel):
    """Structured post-call executive summary and business insights."""

    session_id: str
    lead_id: Optional[str] = None
    lead_name: str
    company: Optional[str] = None
    language: SupportedLang = "en"
    turn_count: int
    duration_estimate_sec: int
    executive_summary: str
    key_pain_points: list[str]
    objections_raised: list[str]
    qualification: LeadQualification
    average_sentiment: float
    sentiment_trend: Literal["Improving", "Stable", "Declining"]
    next_steps: list[str]
    booked_appointment: Optional[AppointmentRecord] = None
    generated_at: str


class KnowledgeDocumentInput(BaseModel):
    """Input schema for adding a custom document to the RAG knowledge base."""

    title: str
    category: str = "Custom"
    content: str


class LeadRecord(BaseModel):
    """CRM Lead record."""

    id: str
    name: str
    email: Optional[str] = None
    company: Optional[str] = None
    role: Optional[str] = None
    preferred_language: SupportedLang = "en"
    score: int = 0
    tier: str = "Cold"
    stage: str = "Discovery"
    estimated_acv_usd: int = 0
    bant: BANTProfile = Field(default_factory=BANTProfile)
    pain_points: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    last_sentiment: float = 0.0
    updated_at: str
    created_at: str


class BusinessInsights(BaseModel):
    """Aggregate business intelligence metrics across all recorded sales calls."""

    total_calls: int
    total_leads: int
    hot_leads: int
    warm_leads: int
    cold_leads: int
    booked_demos: int
    conversion_rate_pct: float
    average_lead_score: float
    average_sentiment: float
    total_pipeline_usd: int
    language_distribution: dict[str, int]
    top_pain_points: list[dict[str, Any]]
    top_objections: list[dict[str, Any]]
    stage_funnel: dict[str, int]


class RuntimeConfigUpdate(BaseModel):
    """Allows setting Gemini API key or model dynamically at runtime."""

    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    crm_webhook_url: Optional[str] = None
