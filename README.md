# Fortivex Autonomous AI Sales Voice Agent

A functional, real-time AI-powered Autonomous Sales Voice Agent prototype built with **FastAPI**, **WebSockets**, **scikit-learn Hybrid RAG**, **SQLite CRM**, and a modern **Web Speech API + Audio Visualizer** frontend.

## Core Capabilities

1. **Real-Time Voice Interaction (STT & TTS)**
   - Live browser-based Speech-to-Text (`Web Speech API`) & Text-to-Speech (`SpeechSynthesis`) with barge-in interruption, push-to-talk, and continuous hands-free mode.
   - Real-time WebSocket streaming (`/ws/call/{session_id}`) with Web Audio API waveform visualization and server-side `/api/stt` & `/api/tts` endpoints.
2. **Contextual Memory & Multilingual Communication**
   - Persistent per-session & cross-call entity memory (name, company, role, budget, timeline, team size, pain points, objections).
   - Automatic language detection and fluent responses across **English (`en`)**, **Spanish (`es`)**, **French (`fr`)**, **German (`de`)**, and **Hindi (`hi`)**, with synchronized STT/TTS voice locales.
3. **Intelligent Lead Qualification & Sentiment Analysis**
   - Real-time **BANT** (Budget, Authority, Need, Timeline) extraction, dynamic 0–100 lead scoring, tier classification (`Hot`, `Warm`, `Cold`, `Disqualified`), and adaptive next-best-question steering.
   - Per-turn sentiment polarity (`-1.0` to `+1.0`), emotion/signal detection (`buying_signal`, `price_objection`, `security_concern`, `competitor_comparison`), and live sentiment trajectory.
4. **RAG-Based Knowledge Retrieval**
   - Hybrid TF-IDF + multilingual keyword-bridge retriever (`scikit-learn`) over curated Fortivex product, pricing, compliance, integration, and competitor battlecard documents.
   - Live RAG Inspector showing retrieved chunks, sources, and similarity scores, plus dynamic document ingestion via UI/API.
5. **Automated Appointment Booking & CRM Integration**
   - SQLite-backed availability calendar with conflict detection, natural-language date/time parsing, automated demo booking, and rescheduling.
   - Full CRM pipeline (`leads`, `deals`, `activities`, `appointments`) with outbound CRM webhook payload logging (Salesforce/HubSpot schema).
6. **Call Summarisation & Business Insights**
   - Automated structured call summaries (TL;DR, BANT scorecard, objections, action items) and executive analytics dashboard (conversion funnel, sentiment trends, top objections, pipeline value).

## Quick Start

```powershell
# Optional: configure Gemini API key for LLM-enhanced generation (works 100% out-of-the-box without it)
Copy-Item .env.example .env

# Run the FastAPI server
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open `http://127.0.0.1:8000` in Chrome or Edge (recommended for full Web Speech API microphone support).

## Running Tests

```powershell
python -m pytest -v
```
