# Fortivex — Autonomous AI Sales Voice Agent

An end-to-end, production-conscious AI Sales Voice Agent built with **FastAPI**, **WebSockets**, **Hybrid Dense + Sparse RAG**, **SQLite Atomic Transactions & Migrations**, **Pluggable STT/TTS & CRM Adapters**, and a **Modern SaaS Dashboard**.

## Architecture & Documentation

- [System Architecture & Diagrams](docs/ARCHITECTURE.md)
- [Existing Codebase Audit](docs/IMPLEMENTATION_AUDIT.md)
- [REST & WebSocket API Reference](docs/API_DOCUMENTATION.md)

## Six Core Requirements Implemented

| Requirement | Implementation Summary | Status |
|---|---|---|
| **1. Real-Time Voice (STT & TTS)** | Pluggable `VoiceProviderRegistry` (`Deepgram`, `Whisper`, `Gemini Audio`, `ElevenLabs`, `Google Cloud TTS`, `Local WAV`, plus Browser Web Speech + MediaRecorder fallback), WebSocket streaming (`turn_delta`), barge-in (`tts_stop`), and latency breakdown. | **IMPLEMENTED AND VERIFIED** (External cloud voice APIs require keys; local WAV + Browser STT/TTS verified live) |
| **2. Contextual Memory & Multilingual AI** | 3-tier memory (`short_term_turns`, `session_facts`, `long_term_profile` across calls via email/phone/company) + **English, Hindi, Hinglish, Spanish, French, German** detection and response generation. | **IMPLEMENTED AND VERIFIED** |
| **3. Lead Qualification & Sentiment** | Deterministic BANT scoring (0–100) with explainable rationale, `Hot`/`Warm`/`Cold`/`Disqualified` tiers, `score_history` persistence, and sentiment separated from `intent_score` & `purchase_readiness`. | **IMPLEMENTED AND VERIFIED** |
| **4. Hybrid RAG Knowledge Engine** | `.pdf`, `.docx`, `.txt`, `.md` ingestion + chunking + Sparse TF-IDF + Dense LSA/Gemini embeddings + Reciprocal Rank Fusion + strict out-of-scope refusal guardrail + RAG benchmark (`scripts/eval_rag.py`). | **IMPLEMENTED AND VERIFIED** |
| **5. Automated Appointment Booking** | Natural-language date/time parsing, Mon–Fri business-hour validation, `BEGIN IMMEDIATE` atomic locking preventing concurrent double-booking (`409 Conflict`), rescheduling (`PATCH`), and cancellation (`DELETE`). | **IMPLEMENTED AND VERIFIED** |
| **6. Call Summarisation & CRM / BI** | Modular CRM layer (`SQLite`, `Webhook` with exponential backoff retries, `HubSpot`, `Salesforce`, `Mock`) + structured `CallSummary` reports + filtered analytics (`date_from`, `date_to`, `tier`, `outcome`). | **IMPLEMENTED AND VERIFIED** (HubSpot/Salesforce require external credentials; webhook retries & SQLite verified) |

## Installation & Quick Start

```powershell
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Optional: configure API keys (system runs 100% locally without external keys)
Copy-Item .env.example .env

# 3. Start the server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000** in Chrome or Edge.

## Testing & RAG Evaluation

```powershell
# Run full unit & integration test suite (11 tests)
python -m pytest -v

# Run Hybrid RAG vs Sparse TF-IDF evaluation benchmark
python scripts/eval_rag.py
```

### Observed Benchmark Results (`scripts/eval_rag.py`)
- `sparse_only`: `recall@3 = 1.0`, `precision@3 = 0.738`, `mrr = 0.857`, `unknown_rejection_rate = 0.5`
- `hybrid_rrf`: `recall@3 = 1.0`, `precision@3 = 0.619`, `mrr = 1.0`, `unknown_rejection_rate = 1.0`

## Known Limitations
- Live cloud STT/TTS (`Deepgram`, `Whisper`, `ElevenLabs`, `Google Cloud TTS`) and live external CRMs (`HubSpot`, `Salesforce`) require valid API credentials in `.env` or runtime Settings; without them, the system operates in `local-hybrid` + `browser-webspeech` + `local-sqlite` mode.
- Rate limiting and WebSocket state use single-process in-memory buckets; multi-worker deployment would require Redis-backed state.
