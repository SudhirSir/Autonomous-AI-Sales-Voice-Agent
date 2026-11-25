# System Architecture — Fortivex Autonomous AI Sales Voice Agent

## 1. High-Level Architecture

Fortivex is organized as a modular FastAPI + WebSocket voice & revenue automation platform backed by SQLite, a Hybrid Dense+Sparse RAG engine, deterministic BANT + Sentiment NLP, an audited Tool Orchestrator, and pluggable Voice & CRM provider adapters.

```mermaid
flowchart TB
    User["Buyer (Browser Mic / Audio / Text)"] -->|WebRTC Audio + WebSocket / REST| API["FastAPI Gateway + Rate Limiter + Auth"]
    API --> Voice["Voice Pipeline (VAD + STT/TTS Provider Registry)"]
    API --> Orch["Autonomous Agent Orchestrator & Tool Registry"]

    Orch --> Mem["3-Tier Memory Engine (Short-Term / Session / Long-Term DB)"]
    Orch --> NLP["Multilingual NLP (EN/HI/Hinglish/ES/FR/DE + BANT + Sentiment)"]
    Orch --> RAG["Hybrid RAG Engine (TF-IDF Sparse + Dense Vector + RRF Rerank)"]
    Orch --> Booking["Atomic Appointment Engine (BEGIN IMMEDIATE + Business Hours)"]
    Orch --> CRM["CRM Adapter Layer (SQLite + Webhook + HubSpot + Salesforce + Retries)"]

    Mem --> DB[("SQLite Database + Migrations + Backups")]
    Booking --> DB
    CRM --> DB
    Orch --> Analytics["Call Summarizer & Filtered Business Intelligence"]
    Analytics --> DB
```

## 2. End-to-End Conversational Turn Workflow

```mermaid
sequenceDiagram
    participant B as Buyer (Voice/Text)
    participant WS as WebSocket /ws/call/{id}
    participant STT as STT Provider
    participant Orch as Agent Orchestrator
    participant Mem as 3-Tier Memory
    participant RAG as Hybrid RAG
    participant DB as SQLite / CRM / Calendar
    participant TTS as TTS Provider

    B->>WS: Audio chunk or transcript (barge-in cancels prior TTS)
    WS->>STT: Transcribe (if audio) + detect language (en/hi/hinglish/es/fr/de)
    STT->>Orch: Normalized utterance + latency_stt_ms
    Orch->>Mem: Retrieve short-term, session & cross-call long-term memory
    Orch->>Orch: Extract BANT, sentiment, buying signals, objections
    Orch->>RAG: Hybrid search (Sparse TF-IDF + Dense Embeddings + RRF)
    RAG-->>Orch: Grounded chunks + citation scores (or unknown-answer flag)
    Orch->>DB: Execute validated tools (upsert CRM, check/book/reschedule slot)
    Orch->>Orch: Generate grounded multilingual response (Gemini or Local-Hybrid)
    Orch->>TTS: Synthesize voice stream / locale metadata
    WS-->>B: Stream response, TTS audio/locale, BANT score, RAG citations & booking state
```

## 3. Core Modules

- `app/config.py`: Environment-driven configuration (`pydantic-settings`), provider keys, security/CORS/rate-limit settings.
- `app/voice.py`: Abstract `STTProvider` & `TTSProvider` interfaces with implementations for Gemini Multimodal, OpenAI/Whisper-compatible, Deepgram, ElevenLabs, and Local/Browser fallback.
- `app/rag.py`: Document ingestion (`.pdf`, `.docx`, `.txt`, `.md`), semantic chunking, sparse TF-IDF + dense semantic embeddings (`sentence-transformers` / Gemini / local `TruncatedSVD` LSI), Reciprocal Rank Fusion, and grounded citation enforcement.
- `app/nlp.py`: 6-language + Hinglish code-switching detection, deterministic BANT + `Disqualified` scoring, sentiment vs purchase-intent separation, and timezone-aware appointment parsing.
- `app/tools.py`: Structured Pydantic tool definitions, permission checks, execution wrapper, and persistent `tool_execution_logs`.
- `app/crm.py`: Pluggable CRM integration layer (`LocalSQLiteCRM`, `WebhookCRM`, `HubSpotCRM`, `SalesforceCRM`) with exponential backoff retries, timeouts, idempotency keys, and delivery audit logs.
- `app/database.py`: SQLite schema management with versioned migrations, atomic `BEGIN IMMEDIATE` appointment locking, `score_history`, `call_activities`, filtered analytics, and backup utilities.
- `app/agent.py`: Autonomous orchestrator uniting memory retrieval, tool selection, grounded LLM/hybrid generation, and structured call report generation.
