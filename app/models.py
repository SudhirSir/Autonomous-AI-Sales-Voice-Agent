"""Pydantic data models for Fortivex V2 AI Sales Voice Agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal, Optional
from pydantic import BaseModel, Field


def utc_now() -> datetime:
    """Return timezone-aware UTC datetime."""
    return datetime.now(timezone.utc)


def utc_now_iso() -> str:
    """Return ISO-8601 UTC timestamp string."""
    return utc_now().isoformat()


SupportedLang = Literal["en", "es", "fr", "de", "hi", "hinglish"]

IntentType = Literal[
    "product_question",
    "pricing_question",
    "feature_question",
    "availability_request",
    "booking_request",
    "booking_confirmation",
    "reschedule_request",
    "cancellation_request",
    "objection",
    "follow_up_request",
    "opt_out",
    "general_conversation",
]


class IntentClassification(BaseModel):
    """Contextual intent classification result."""

    intent: IntentType = "general_conversation"
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    requires_confirmation: bool = False
    reasoning: str = ""


class BANTProfile(BaseModel):
    """Structured BANT lead qualification dimensions with evidence & confidence."""

    budget: Optional[str] = None
    budget_score: int = Field(default=0, ge=0, le=25)
    budget_evidence: Optional[str] = None
    budget_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    authority: Optional[str] = None
    authority_score: int = Field(default=0, ge=0, le=25)
    authority_evidence: Optional[str] = None
    authority_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    need: Optional[str] = None
    need_score: int = Field(default=0, ge=0, le=25)
    need_evidence: Optional[str] = None
    need_confidence: float = Field(default=0.0, ge=0.0, le=1.0)

    timeline: Optional[str] = None
    timeline_score: int = Field(default=0, ge=0, le=25)
    timeline_evidence: Optional[str] = None
    timeline_confidence: float = Field(default=0.0, ge=0.0, le=1.0)


class LeadQualification(BaseModel):
    """Computed lead score, tier, stage, priority, and explainable BANT rationale."""

    score: int = Field(default=0, ge=0, le=100)
    tier: Literal["Hot", "Warm", "Cold", "Disqualified"] = "Cold"
    stage: Literal[
        "Discovery", "Qualification", "Proposal", "Demo Scheduled", "Nurture", "Disqualified"
    ] = "Discovery"
    lead_priority: Literal["P1-Urgent", "P2-High", "P3-Standard", "P4-Low", "Disqualified"] = "P3-Standard"
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    bant: BANTProfile = Field(default_factory=BANTProfile)
    missing_dimensions: list[str] = Field(default_factory=lambda: ["budget", "authority", "need", "timeline"])
    estimated_acv_usd: int = 0
    score_rationale: list[str] = Field(default_factory=list)
    evidence_summary: dict[str, str] = Field(default_factory=dict)


class SentimentResult(BaseModel):
    """Per-turn emotional sentiment separated from commercial purchase intent."""

    polarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    label: Literal["Positive", "Neutral", "Negative", "Frustrated", "Skeptical", "Mixed"] = "Neutral"
    confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    intent_score: int = Field(default=20, ge=0, le=100)
    purchase_readiness: Literal["High", "Medium", "Low", "None"] = "Low"
    frustration_detected: bool = False
    trust_concern: bool = False
    price_objection: bool = False
    competitor_objection: bool = False
    sarcasm_detected: bool = False
    mixed_sentiment: bool = False
    negation_adjusted: bool = False
    engagement_level: Literal["High", "Medium", "Low"] = "Medium"
    signals: list[str] = Field(default_factory=list)


class RetrievedChunk(BaseModel):
    """Knowledge base chunk returned by the hybrid RAG retriever with full diagnostics."""

    doc_id: str
    title: str
    category: str
    source_file: str = ""
    content: str
    score: float = Field(ge=0.0, le=1.0)
    rank: int = 1
    sparse_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    reranker_score: float = 0.0
    final_confidence: float = 0.0
    embedding_backend: str = "local"
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
    timezone: str = "UTC"
    pain_points: list[str] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    competitors_mentioned: list[str] = Field(default_factory=list)
    buying_signals: list[str] = Field(default_factory=list)
    interests: list[str] = Field(default_factory=list)
    prior_calls_count: int = 0
    returning_lead: bool = False


class ThreeTierMemorySnapshot(BaseModel):
    """Explicit 3-tier memory snapshot retrieved before every agent response."""

    short_term_turns: list[dict[str, str]] = Field(default_factory=list)
    session_facts: dict[str, Any] = Field(default_factory=dict)
    long_term_profile: Optional[dict[str, Any]] = None
    previous_interactions: list[str] = Field(default_factory=list)
    appointment_history: list[dict[str, Any]] = Field(default_factory=list)


class ConversationMessage(BaseModel):
    """Single message turn stored in session history."""

    id: Optional[int] = None
    session_id: str
    role: Literal["user", "assistant", "system"]
    text: str
    language: SupportedLang = "en"
    sentiment: Optional[SentimentResult] = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    created_at: str = Field(default_factory=utc_now_iso)


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


class AppointmentReschedule(BaseModel):
    """Request to reschedule an existing appointment."""

    slot_iso: str
    timezone: Optional[str] = None


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


class ToolExecutionLog(BaseModel):
    """Audited tool invocation executed by the autonomous agent orchestrator."""

    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
    status: Literal["success", "error", "denied"] = "success"
    summary: str
    result_preview: Optional[str] = None
    error: Optional[str] = None
    duration_ms: int = 0
    executed_at: str = Field(default_factory=utc_now_iso)


class CRMSyncStatusRecord(BaseModel):
    """Structured CRM synchronization status for local and external providers."""

    provider: str
    operation: str = "lead.upsert"
    local_persistence_status: Literal["saved", "failed"] = "saved"
    external_sync_status: Literal["not_configured", "pending", "synced", "failed", "mock"] = "not_configured"
    external_record_id: Optional[str] = None
    http_status: Optional[int] = None
    retry_count: int = 0
    error_code: Optional[str] = None
    failure_reason: Optional[str] = None
    last_sync_timestamp: str = Field(default_factory=utc_now_iso)
    idempotency_key: str = ""


class SessionMetrics(BaseModel):
    """Real event-driven call metrics tracked from session lifecycle timestamps."""

    call_started_at: str = Field(default_factory=utc_now_iso)
    call_connected_at: Optional[str] = None
    call_ended_at: Optional[str] = None
    actual_duration_seconds: float = 0.0
    speaking_duration_seconds: float = 0.0
    silence_duration_seconds: float = 0.0
    interruption_count: int = 0
    user_turn_count: int = 0
    agent_turn_count: int = 0
    average_response_latency_ms: float = 0.0
    stt_processing_ms: int = 0
    tts_processing_ms: int = 0


class TurnRequest(BaseModel):
    """Input payload for a single voice/text interaction turn."""

    session_id: str
    text: str
    language_override: Optional[SupportedLang] = None
    input_mode: Literal["voice", "text"] = "voice"
    timezone: str = "UTC"
    stt_latency_ms: int = 0
    confirm_action: Optional[bool] = None


class TurnResponse(BaseModel):
    """Complete agent response with NLU, RAG, CRM, Tools, Metrics, and Voice metadata."""

    session_id: str
    reply_text: str
    language: SupportedLang
    tts_locale: str
    stt_provider: str = "browser"
    tts_provider: str = "browser"
    intent: Optional[IntentClassification] = None
    sentiment: SentimentResult
    qualification: LeadQualification
    memory: LeadMemory
    memory_snapshot: Optional[ThreeTierMemorySnapshot] = None
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    kb_grounded: bool = True
    contradiction_detected: bool = False
    contradiction_note: Optional[str] = None
    booked_appointment: Optional[AppointmentRecord] = None
    proposed_slots: list[str] = Field(default_factory=list)
    booking_error: Optional[str] = None
    requires_confirmation: bool = False
    pending_action: Optional[dict[str, Any]] = None
    crm_synced: bool = False
    local_persistence_status: Literal["saved", "failed"] = "saved"
    external_crm_status: Literal["not_configured", "pending", "synced", "failed", "mock"] = "not_configured"
    crm_mode: str = "local-sqlite"
    crm_sync_results: list[CRMSyncStatusRecord] = Field(default_factory=list)
    webhook_events: list[dict[str, Any]] = Field(default_factory=list)
    tool_logs: list[ToolExecutionLog] = Field(default_factory=list)
    ai_engine: Literal["gemini", "local-hybrid"] = "local-hybrid"
    latency_ms: int = 0
    latency_breakdown: dict[str, int] = Field(default_factory=dict)
    session_metrics: Optional[SessionMetrics] = None


class CallSummary(BaseModel):
    """Structured post-call executive summary and real measured call metrics."""

    session_id: str
    lead_id: Optional[str] = None
    lead_name: str
    customer_email: Optional[str] = None
    customer_phone: Optional[str] = None
    company: Optional[str] = None
    language: SupportedLang = "en"
    turn_count: int
    duration_estimate_sec: int  # Set from actual_duration_seconds for API backward-compatibility
    actual_duration_seconds: float = 0.0
    call_started_at: str = ""
    call_ended_at: Optional[str] = None
    speaking_duration_seconds: float = 0.0
    silence_duration_seconds: float = 0.0
    interruption_count: int = 0
    user_turn_count: int = 0
    agent_turn_count: int = 0
    average_response_latency_ms: float = 0.0
    stt_processing_ms: int = 0
    tts_processing_ms: int = 0
    executive_summary: str
    key_pain_points: list[str]
    buying_signals: list[str] = Field(default_factory=list)
    objections_raised: list[str]
    competitor_mentions: list[str] = Field(default_factory=list)
    knowledge_sources_used: list[str] = Field(default_factory=list)
    qualification: LeadQualification
    average_sentiment: float
    sentiment_trend: Literal["Improving", "Stable", "Declining"]
    appointment_outcome: str = "No demo scheduled"
    next_steps: list[str]
    follow_up_date: str = ""
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
    phone: Optional[str] = None
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
    external_crm_ids: dict[str, str] = Field(default_factory=dict)
    updated_at: str
    created_at: str


class BusinessInsights(BaseModel):
    """Aggregate business intelligence metrics across stored sales calls."""

    total_calls: int
    unique_leads: int
    total_leads: int
    qualified_leads: int
    hot_leads: int
    warm_leads: int
    cold_leads: int
    disqualified_leads: int = 0
    booked_demos: int
    conversion_rate_pct: float
    booking_rate_pct: float
    average_lead_score: float
    average_sentiment: float
    average_call_duration_sec: float = 0.0
    total_pipeline_usd: int
    language_distribution: dict[str, int]
    top_pain_points: list[dict[str, Any]]
    top_objections: list[dict[str, Any]]
    frequently_asked_questions: list[dict[str, Any]] = Field(default_factory=list)
    stage_funnel: dict[str, int]
    agent_performance: dict[str, Any] = Field(default_factory=dict)


class RuntimeConfigUpdate(BaseModel):
    """Allows updating provider keys and runtime config safely from backend/settings."""

    gemini_api_key: Optional[str] = None
    gemini_model: Optional[str] = None
    deepgram_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    elevenlabs_api_key: Optional[str] = None
    stt_provider: Optional[str] = None
    tts_provider: Optional[str] = None
    crm_webhook_url: Optional[str] = None
    hubspot_access_token: Optional[str] = None
    salesforce_instance_url: Optional[str] = None
    salesforce_access_token: Optional[str] = None
    crm_mock_mode: Optional[bool] = None
    business_timezone: Optional[str] = None
