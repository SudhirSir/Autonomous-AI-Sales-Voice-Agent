# Fortivex V2 — Autonomous AI Sales Voice Agent

An end-to-end AI Sales Voice Agent built with **FastAPI**, **WebSockets (Streaming & Barge-In)**, **11-Tool Gemini Function Calling**, **Hybrid RAG + Persistent Cache + Reranker + Contradiction Guardrail**, **Calibrated BANT & Sentiment**, **DST-Safe Timezone Booking**, **Idempotent CRM Upserts (HubSpot / Salesforce / Webhook / SQLite + DLQ)**, **Genuine STT/TTS**, and **Real Lifecycle Call Metrics**.

## Architecture & Documentation

- [System Architecture Diagram (SVG)](docs/system_architecture.svg)
- [System Architecture & Sequence Flows](docs/ARCHITECTURE.md)
- [REST & WebSocket API Reference](docs/API_DOCUMENTATION.md)
- [Codebase Implementation Audit](docs/IMPLEMENTATION_AUDIT.md)

## Six Core Requirements Compliance Matrix

| Requirement | V2 Implementation Summary | Verification Status |
|---|---|---|
| **1. Real-Time Voice (STT & TTS)** | Unified `STTProvider` (`Deepgram`, `OpenAI Whisper`, `Gemini Audio`, `Faster-Whisper` w/ timestamps, `Browser WebSpeech`) & `TTSProvider` (`ElevenLabs`, `Google Cloud TTS`, `Piper` `.onnx`, `Windows System.Speech` offline WAV, `Browser SpeechSynthesis`). No synthetic audio placeholders; WebSocket async streaming + `asyncio.Task` / `threading.Event` barge-in cancellation. | **VERIFIED** (`Windows System.Speech` WAV + WebSocket streaming/barge-in verified live; `Deepgram`/`OpenAI`/`ElevenLabs`/`Google TTS` verified via mocks; `Faster-Whisper`/`Piper` require optional model install) |
| **2. Contextual Memory & Multilingual AI** | 3-tier memory (`short_term_turns`, `session_facts`, `long_term_profile` via email/phone/company) + **EN, HI, Hinglish, ES, FR, DE** detection, contextual 12-class intent classifier, and zero hardcoded commercial claims. | **VERIFIED** (Unit, E2E & live Gemini Turn 1 verified) |
| **3. Lead Qualification & Sentiment** | Calibrated BANT (25x4 = 0–100) with evidence, confidence & self-correction (no booking score inflation) + sentiment separated from `intent_score` with negation, sarcasm, and mixed-sentiment handling. | **VERIFIED** |
| **4. Hybrid RAG Knowledge Engine** | `.pdf/.docx/.txt/.md` chunking, Sparse TF-IDF + Dense (`SentenceTransformer` / `Gemini` / `local-tfidf-lsa` fallback) + `DiskEmbeddingCache` + RRF + Reranker + Contradiction Detection + Out-of-Scope & Prompt-Injection Refusal + 100-query benchmark (`scripts/eval_rag.py`). | **VERIFIED** (`SentenceTransformer` neural backend requires optional install due to disk/model size; local hybrid + cache + reranker + contradiction verified) |
| **5. Automated Appointment Booking & CRM** | 12-class intent gating (informational demo questions do not book), DST-safe `ZoneInfo` + natural TZ parsing, `BEGIN IMMEDIATE` atomic booking (`409`/`422`), reschedule/cancel, plus HubSpot & Salesforce upsert, deterministic SHA-256 idempotency, transient-only retries, and Dead-Letter Queue (`/api/crm/failed-jobs`, `/api/crm/retry/{job_id}`). Honest `local_persistence_status` vs `external_crm_status`. | **VERIFIED** (SQLite atomic concurrency verified live; HubSpot/Salesforce/Webhook verified via HTTP mocks) |
| **6. Call Summarisation & Business Insights** | Real event-driven `SessionMetrics` (`call_started_at`, `call_ended_at`, `actual_duration_seconds`, `speaking_duration_seconds`, `silence_duration_seconds`, `interruption_count`, `average_response_latency_ms`) + filtered BI insights (`date_from`, `date_to`, `tier`, `outcome`). | **VERIFIED** |

## Installation & Quick Start

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Configure environment (optional external keys)
Copy-Item .env.example .env

# 3. Run Alembic schema migrations (optional for SQLite / PostgreSQL)
python -m alembic upgrade head

# 4. Start the server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in Chrome or Edge.

## Testing, RAG Benchmark & Acceptance Demo

```powershell
# Run full test suite (20 tests across voice, tools, grounding, CRM, intent, BANT, TZ, auth, E2E)
python -m pytest -v

# Run 100-query RAG evaluation benchmark
python scripts/eval_rag.py

# Run Phase 15 end-to-end acceptance demonstration
python scripts/verify_e2e_v2.py
```
