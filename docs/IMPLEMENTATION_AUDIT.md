# Existing Codebase Implementation Audit

**Audit Date:** 2026-11-25  
**Repository:** `https://github.com/SudhirSir/Autonomous-AI-Sales-Voice-Agent`  
**Audited Revision:** `752e8d5` (plus redesigned light SaaS UI in `static/`)

## 1. File-by-File Inventory

| File | Purpose | Current State |
|---|---|---|
| `app/config.py` | Environment & settings via `pydantic-settings` | Working; lacks external STT/TTS/CRM provider keys, auth token, rate-limit, and CORS settings. |
| `app/models.py` | Pydantic request/response schemas | Working; lacks Hinglish (`hinglish`), `intent_score`, `score_history`, `tool_logs`, `reschedule`, and filtered analytics models. |
| `app/rag.py` | TF-IDF + cross-lingual keyword expansion RAG | Working for `.md`/`.txt` in memory; lacks PDF/DOCX ingestion, dense semantic vector retrieval, persistent vector store, document deletion, and evaluation harness. |
| `app/nlp.py` | Language detection, sentiment, entity extraction, BANT, slot parsing | Working for `en/es/fr/de/hi`; lacks Hinglish code-switching, `phone` extraction, `Disqualified` tier rules, separate sentiment vs purchase intent, and strict business-hours date validation. |
| `app/database.py` | SQLite CRM, sessions, messages, appointments, summaries | Working; lacks schema migrations, `score_history`, `tool_execution_logs`, `crm_sync_logs`, atomic `BEGIN IMMEDIATE` booking lock, rescheduling, and filtered analytics queries. |
| `app/agent.py` | Turn orchestration & Gemini/local-hybrid reply generation | Working; procedural pipeline without structured tool-calling registry, lacks unknown-KB refusal guardrail, and uses `datetime.utcnow()` (deprecated in Python 3.13). |
| `app/main.py` | FastAPI REST + WebSocket server | Working; `/api/stt` only handles Gemini audio upload, `/api/tts` missing, `CRM_WEBHOOK_URL` not actually dispatched via HTTP, no API auth or rate limiting, open `allow_origins=["*"]`. |
| `static/index.html`, `static/styles.css`, `static/app.js` | Single-page SaaS dashboard | Clean UI working for Live Call, CRM & Calendar, KB, Insights; lacks file upload (PDF/DOCX), KB delete, appointment reschedule, Call History view, VAD sensitivity, and analytics filters. |
| `tests/test_sales_agent.py` | Automated pytest suite | 5 tests passing; lacks concurrent booking race test, retry/backoff CRM mock tests, Hinglish tests, RAG grounding refusal tests, and auth/rate-limit tests. |

## 2. Feature-by-Feature Gap Analysis

### 1. Real-Time Voice Interaction (STT & TTS)
- **What Exists:** WebSocket `/ws/call/{session_id}`, browser `SpeechRecognition` & `speechSynthesis`, Web Audio waveform canvas, barge-in (`speechSynthesis.cancel()` + WS `barge_in`), and `/api/stt` Gemini audio endpoint.
- **Gaps Identified:**
  - Relies solely on browser Web Speech API for live STT/TTS in practice; no unified `STTProvider` / `TTSProvider` abstraction for pluggable backends (Deepgram, Whisper-compatible, Google Cloud TTS, ElevenLabs, Gemini, Local WAV/gTTS-free synthesizer fallback).
  - No client-side Voice Activity Detection (RMS energy threshold + silence hang-time) with `MediaRecorder` audio capture fallback to `/api/stt`.
  - No `/api/tts` endpoint returning synthesized audio (`audio/wav` or `audio/mpeg`) for non-WebSpeech clients.
  - No streaming sentence-chunked TTS playback or per-stage latency telemetry (`rag_ms`, `llm_ms`, `tts_ms`).

### 2. Contextual Memory & Multilingual Communication
- **What Exists:** Per-session message history, `LeadMemory` JSON state, and cross-call lookup by `email` or `name + company`. Supports `en`, `es`, `fr`, `de`, `hi`.
- **Gaps Identified:**
  - No explicit 3-tier memory retrieval report (`short_term`, `session`, `long_term`) including `phone`, `previous_interactions`, and `appointment_history`.
  - Missing **Hinglish** (`hinglish`) Romanized Hindi-English code-switching detection and response generation.
  - Returning lead matching does not handle phone number lookup.

### 3. Intelligent Lead Qualification & Sentiment Analysis
- **What Exists:** Deterministic BANT scorer (0–100), `Hot`/`Warm`/`Cold` tiers, ACV estimation, per-turn polarity & signal tags.
- **Gaps Identified:**
  - No `Disqualified` tier assignment or explicit disqualification rules.
  - No persisted `score_history` table tracking score evolution per turn.
  - Does not separate `sentiment` (emotional tone + frustration) from `purchase_intent` (buying readiness 0–100) cleanly.

### 4. RAG-Based Knowledge Retrieval
- **What Exists:** `scikit-learn` TF-IDF + lexical bonus over 5 curated `.md` files and `custom_docs`.
- **Gaps Identified:**
  - Sparse TF-IDF only—no dense semantic vector embeddings (`TruncatedSVD` LSA + Gemini `text-embedding-004` / `sentence-transformers` adapter) or Reciprocal Rank Fusion (RRF) + reranking.
  - No `.pdf` or `.docx` file parsing/upload endpoint.
  - No `DELETE /api/knowledge/{doc_id}` endpoint.
  - No strict refusal when KB has no relevant grounded answer (`min_score` gating).
  - No quantitative RAG evaluation script measuring Precision@K, Recall@K, and groundedness.

### 5. Automated Appointment Booking & CRM Integration
- **What Exists:** SQLite `appointments` & `leads`, conflict auto-shifting (+1 hr), cancellation, CSV export, and local `webhook_events` logging.
- **Gaps Identified:**
  - Booking silently shifts conflicting slots instead of rejecting conflicts atomically and returning clear conflict/business-hour errors; lacks `BEGIN IMMEDIATE` transaction locking against concurrent race conditions.
  - Missing appointment **rescheduling** (`PATCH /api/appointments/{id}` and conversational reschedule).
  - `CRM_WEBHOOK_URL` is stored in config but never actually called via HTTP; missing HubSpot & Salesforce adapters, retry with exponential backoff, timeout handling, idempotency keys, and `crm_sync_logs`.

### 6. Call Summarisation & Business Insights
- **What Exists:** Structured `CallSummary` and aggregate `BusinessInsights`.
- **Gaps Identified:**
  - Summaries omit `knowledge_sources_used`, `competitor_mentions`, `buying_signals`, `follow_up_date`, and `customer_phone`.
  - Analytics lack `frequently_asked_questions`, `agent_performance` metrics, and query filters (`date_from`, `date_to`, `tier`, `outcome`).

### 7. Agent Orchestration, Security & Reliability
- **What Exists:** Procedural turn flow in `app/agent.py`, Pydantic models, parameterized SQLite queries.
- **Gaps Identified:**
  - No structured Tool Registry / execution audit log (`tool_execution_logs`) separating LLM reasoning from tool execution.
  - Open CORS (`*`), no optional API auth (`X-API-Key` / Bearer) for admin/CRM routes, no rate limiting, no PII redaction mode, no DB backup endpoint, and `datetime.utcnow()` Python 3.13 deprecation warnings across 4 modules.
