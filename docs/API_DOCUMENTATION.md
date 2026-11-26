# API Documentation — Fortivex Autonomous AI Sales Voice Agent

Base URL (local): `http://127.0.0.1:8000`

## Authentication
When `FORTIVEX_API_TOKEN` is unset (default local dev), endpoints are accessible without a token. When `FORTIVEX_API_TOKEN` is configured, protected admin/CRM/analytics endpoints require `Authorization: Bearer <token>` or `X-API-Key: <token>`.

---

## Health & Configuration

### `GET /api/health`
Returns server status, active AI/STT/TTS/CRM engines, and indexed knowledge chunk count.

### `POST /api/config`
Dynamically updates runtime keys in memory (never exposed back to the frontend).
```json
{
  "gemini_api_key": "optional-key",
  "crm_webhook_url": "https://example.com/webhook"
}
```

---

## Real-Time Voice & Conversation

### `WebSocket /ws/call/{session_id}`
Bidirectional real-time call channel.
- **Client → Server messages:**
  - `{"type": "turn", "text": "...", "language_override": "hi", "input_mode": "voice"}`
  - `{"type": "audio_chunk", "audio_b64": "...", "mime_type": "audio/webm", "language_override": null}`
  - `{"type": "barge_in"}`
  - `{"type": "end_call"}`
  - `{"type": "ping"}`
- **Server → Client messages:**
  - `session_ready`
  - `transcript_stt`
  - `agent_thinking`
  - `turn_delta` (streamed response chunks)
  - `turn_result` (full `TurnResponse` payload + latency breakdown)
  - `tts_stop`
  - `call_summary`

### `POST /api/turn`
HTTP fallback for a single conversation turn.
```json
{
  "session_id": "call-demo-1",
  "text": "Hi, I'm Alex Rivera at ApexCloud. What is your Scale plan pricing?",
  "language_override": null,
  "input_mode": "text"
}
```

### `POST /api/stt`
Multipart audio upload (`audio` file + optional `language` form field). Routes through configured STT provider (`gemini`, `deepgram`, `whisper`, or `fallback`).

### `POST /api/tts`
Generates spoken audio (`audio/wav` or `audio/mpeg`) via configured TTS provider (`elevenlabs`, `google`, `local-wav`) for text + language/rate.

---

## Memory, Calls & Summaries

### `GET /api/sessions`
Lists recent call sessions with language, turn count, lead link, and last activity.

### `GET /api/sessions/{session_id}`
Returns full session transcript, 3-tier memory snapshot, BANT score history, tool execution logs, booked appointment, and call summary.

### `POST /api/sessions/{session_id}/summarize`
Generates and persists a structured `CallSummary` report.

### `GET /api/summaries?date_from=&date_to=&tier=&outcome=`
Returns persisted call summaries with optional filters.

---

## Knowledge Base (Hybrid RAG)

### `GET /api/knowledge`
Lists indexed documents and chunks.

### `GET /api/knowledge/search?q=...&top_k=4`
Runs hybrid Dense + Sparse TF-IDF retrieval with RRF scoring and returns citations.

### `POST /api/knowledge`
Ingests JSON text knowledge (`title`, `category`, `content`).

### `POST /api/knowledge/upload`
Multipart upload supporting `.pdf`, `.docx`, `.txt`, and `.md` files.

### `DELETE /api/knowledge/{doc_id}`
Removes a custom knowledge document and rebuilds the sparse + dense index.

---

## Appointments & CRM Integration

### `GET /api/appointments/slots?timezone=UTC&date=`
Returns available conflict-free business-hour slots.

### `POST /api/appointments`
Atomically books a conflict-free appointment (`BEGIN IMMEDIATE`). Returns `409 Conflict` if already booked or `422` if outside business hours / invalid past date.

### `PATCH /api/appointments/{appt_id}`
Reschedules an existing appointment atomically with conflict & business-hour validation.

### `DELETE /api/appointments/{appt_id}`
Cancels an appointment.

### `GET /api/leads` & `GET /api/leads/export`
Lists CRM leads (including score history) and exports CSV.

### `GET /api/crm/status`, `GET /api/crm/failed-jobs`, `POST /api/crm/retry/{job_id}` & `GET /api/webhooks`
Reports active CRM adapters (`local-sqlite`, `webhook`, `hubspot`, `salesforce`, `mock`), delivery audit logs, dead-letter failed jobs, and idempotent job replay.

---

## Voice, RAG Diagnostics, Auth & Reliability

### `GET /api/voice/status` & `GET /api/rag/diagnostics`
Exposes active STT/TTS provider names/voices and RAG embedding/reranker backend diagnostics without exposing credentials.

### `GET /api/auth/status`, `POST /api/auth/session`, `DELETE /api/auth/session`
Manages HttpOnly session cookie authentication when `FORTIVEX_API_TOKEN` is configured.

### `GET /api/insights?date_from=&date_to=&tier=&outcome=`
Computes real-time analytics from SQLite records (calls, unique leads, qualified leads, conversion rate, real average call duration, pipeline value, sentiment trends, top objections, FAQs, and agent performance).

### `POST /api/admin/backup` & `POST /api/admin/retention-cleanup`
Creates a timestamped SQLite backup snapshot in `data/backups/` and enforces configurable data retention / temp audio cleanup.
