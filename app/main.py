"""FastAPI application and WebSocket real-time voice pipeline for Fortivex."""

from __future__ import annotations

import base64
import csv
import io
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.agent import SalesVoiceAgent
from app.config import get_runtime_gemini_key, settings
from app.crm import CRMIntegrationManager
from app.database import BookingConflictError, BookingValidationError, DatabaseStore
from app.models import (
    AppointmentCreate,
    AppointmentReschedule,
    CallSummary,
    KnowledgeDocumentInput,
    RuntimeConfigUpdate,
    TurnRequest,
    TurnResponse,
)
from app.nlp import TTS_LOCALES
from app.rag import HybridRAGRetriever
from app.security import enforce_rate_limit, require_api_auth
from app.voice import VoiceProviderRegistry

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


class TTSRequest(BaseModel):
    text: str
    language: str = "en"
    rate: float = 1.0


def create_app(
    db_path: str | Path | None = None,
    knowledge_dir: str | Path | None = None,
) -> FastAPI:
    """Create and configure the Fortivex FastAPI app."""
    application = FastAPI(title=settings.app_name, version="2.0.0")

    allowed_origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    application.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins or ["http://127.0.0.1:8000"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    db = DatabaseStore(db_path=db_path)
    rag = HybridRAGRetriever(knowledge_dir=knowledge_dir)
    crm = CRMIntegrationManager(db=db)
    voice = VoiceProviderRegistry()
    agent = SalesVoiceAgent(db=db, rag=rag, crm=crm, voice=voice)

    application.state.db = db
    application.state.rag = rag
    application.state.crm = crm
    application.state.voice = voice
    application.state.agent = agent

    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/", include_in_schema=False)
    async def serve_ui():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return {"message": "Fortivex AI Sales Voice Agent API is running."}
        return FileResponse(str(index_path))

    @application.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "app": settings.app_name,
            "ai_engine": "gemini" if get_runtime_gemini_key() else "local-hybrid",
            "gemini_model": settings.gemini_model,
            "stt_provider": voice.active_stt_name(),
            "tts_provider": voice.active_tts_name(),
            "crm_mode": crm.active_mode(),
            "knowledge_chunks": len(rag.chunks),
        }

    @application.post("/api/config", dependencies=[Depends(require_api_auth), Depends(enforce_rate_limit)])
    async def update_runtime_config(payload: RuntimeConfigUpdate) -> dict[str, Any]:
        if payload.gemini_api_key is not None:
            os.environ["GEMINI_API_KEY"] = payload.gemini_api_key.strip()
        if payload.gemini_model:
            settings.gemini_model = payload.gemini_model.strip()
        if payload.deepgram_api_key is not None:
            os.environ["DEEPGRAM_API_KEY"] = payload.deepgram_api_key.strip()
        if payload.openai_api_key is not None:
            os.environ["OPENAI_API_KEY"] = payload.openai_api_key.strip()
        if payload.elevenlabs_api_key is not None:
            os.environ["ELEVENLABS_API_KEY"] = payload.elevenlabs_api_key.strip()
        if payload.stt_provider:
            os.environ["STT_PROVIDER"] = payload.stt_provider.strip()
        if payload.tts_provider:
            os.environ["TTS_PROVIDER"] = payload.tts_provider.strip()
        if payload.crm_webhook_url is not None:
            os.environ["CRM_WEBHOOK_URL"] = payload.crm_webhook_url.strip()
        if payload.hubspot_access_token is not None:
            os.environ["HUBSPOT_ACCESS_TOKEN"] = payload.hubspot_access_token.strip()
        if payload.crm_mock_mode is not None:
            settings.crm_mock_mode = payload.crm_mock_mode
        return {
            "status": "updated",
            "ai_engine": "gemini" if get_runtime_gemini_key() else "local-hybrid",
            "stt_provider": voice.active_stt_name(),
            "tts_provider": voice.active_tts_name(),
            "crm_mode": crm.active_mode(),
        }

    # ---------------- Real-Time Voice & Turn Endpoints ----------------

    @application.post("/api/turn", response_model=TurnResponse, dependencies=[Depends(enforce_rate_limit)])
    async def process_turn_http(req: TurnRequest) -> TurnResponse:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Utterance text cannot be empty.")
        return agent.process_turn(req)

    @application.websocket("/ws/call/{session_id}")
    async def websocket_voice_call(websocket: WebSocket, session_id: str) -> None:
        if len(session_id) > 80 or not session_id.replace("-", "").replace("_", "").isalnum():
            await websocket.close(code=1008)
            return

        await websocket.accept()
        sess = db.get_or_create_session(session_id)
        await websocket.send_json(
            {
                "type": "session_ready",
                "session_id": session_id,
                "language": sess["language"],
                "memory": sess["memory"].model_dump(),
                "qualification": sess["qualification"].model_dump(),
                "ai_engine": "gemini" if get_runtime_gemini_key() else "local-hybrid",
                "stt_provider": voice.active_stt_name(),
                "tts_provider": voice.active_tts_name(),
            }
        )
        try:
            while True:
                data = await websocket.receive_json()
                msg_type = data.get("type", "turn")

                if msg_type == "ping":
                    await websocket.send_json({"type": "pong"})
                    continue

                if msg_type == "barge_in":
                    await websocket.send_json({"type": "tts_stop", "session_id": session_id})
                    continue

                if msg_type == "end_call":
                    summary = agent.generate_call_summary(session_id)
                    await websocket.send_json({"type": "call_summary", "summary": summary.model_dump()})
                    continue

                text = ""
                if msg_type == "audio_chunk":
                    raw_b64 = data.get("audio_b64") or ""
                    mime = data.get("mime_type") or "audio/webm"
                    try:
                        audio_bytes = base64.b64decode(raw_b64)
                    except Exception:
                        await websocket.send_json({"type": "error", "detail": "Invalid base64 audio payload."})
                        continue
                    stt_res = voice.transcribe(audio_bytes, mime_type=mime, language=data.get("language_override"))
                    await websocket.send_json(
                        {
                            "type": "transcript_stt",
                            "transcript": stt_res.transcript,
                            "provider": stt_res.provider,
                            "latency_ms": stt_res.latency_ms,
                            "fallback_used": stt_res.fallback_used,
                            "error": stt_res.error,
                        }
                    )
                    text = stt_res.transcript.strip()
                    if not text:
                        continue
                elif msg_type == "turn":
                    text = (data.get("text") or "").strip()
                    if not text:
                        continue
                else:
                    continue

                await websocket.send_json({"type": "agent_thinking", "session_id": session_id})
                req = TurnRequest(
                    session_id=session_id,
                    text=text,
                    language_override=data.get("language_override"),
                    input_mode=data.get("input_mode", "voice"),
                    timezone=data.get("timezone", "UTC"),
                )
                resp = agent.process_turn(req)

                # Stream response words incrementally before final structured payload
                words = resp.reply_text.split(" ")
                chunk_size = max(3, len(words) // 4)
                for idx in range(0, len(words), chunk_size):
                    delta = " ".join(words[idx : idx + chunk_size])
                    await websocket.send_json({"type": "turn_delta", "delta": delta})

                await websocket.send_json({"type": "turn_result", "data": resp.model_dump()})
        except WebSocketDisconnect:
            pass

    @application.post("/api/stt", dependencies=[Depends(enforce_rate_limit)])
    async def server_stt(
        audio: UploadFile = File(...),
        language: Optional[str] = Form(default=None),
    ) -> dict[str, Any]:
        raw_bytes = await audio.read()
        res = voice.transcribe(
            raw_bytes,
            mime_type=audio.content_type or "audio/webm",
            language=language,
        )
        return {
            "transcript": res.transcript,
            "provider": res.provider,
            "latency_ms": res.latency_ms,
            "fallback_used": res.fallback_used,
            "error": res.error,
            "bytes": len(raw_bytes),
        }

    @application.post("/api/tts", dependencies=[Depends(enforce_rate_limit)])
    async def server_tts(payload: TTSRequest) -> Response:
        locale = TTS_LOCALES.get(payload.language, "en-US")  # type: ignore[arg-type]
        res = voice.synthesize(payload.text, locale=locale, rate=payload.rate)
        return Response(
            content=res.audio_bytes,
            media_type=res.content_type,
            headers={
                "X-TTS-Provider": res.provider,
                "X-TTS-Latency-Ms": str(res.latency_ms),
                "X-TTS-Fallback": str(res.fallback_used).lower(),
            },
        )

    # ---------------- Session, Memory & Summary Endpoints ----------------

    @application.get("/api/sessions", dependencies=[Depends(require_api_auth)])
    async def list_sessions() -> list[dict[str, Any]]:
        return db.list_sessions()

    @application.get("/api/sessions/{session_id}", dependencies=[Depends(require_api_auth)])
    async def get_session(session_id: str) -> dict[str, Any]:
        sess = db.get_or_create_session(session_id)
        messages = db.get_messages(session_id)
        appt = db.get_session_appointment(session_id)
        summary = db.get_call_summary(session_id)
        mem_snap = db.get_three_tier_memory(session_id, sess["memory"])
        tool_logs = db.get_tool_logs(session_id)
        score_hist = db.get_score_history(sess["lead_id"]) if sess["lead_id"] else []
        return {
            "session_id": session_id,
            "language": sess["language"],
            "memory": sess["memory"].model_dump(),
            "memory_snapshot": mem_snap.model_dump(),
            "qualification": sess["qualification"].model_dump(),
            "score_history": score_hist,
            "tool_logs": [t.model_dump() for t in tool_logs],
            "messages": [m.model_dump() for m in messages],
            "booked_appointment": appt.model_dump() if appt else None,
            "summary": summary.model_dump() if summary else None,
        }

    @application.post("/api/sessions/{session_id}/summarize", response_model=CallSummary)
    async def summarize_session(session_id: str) -> CallSummary:
        return agent.generate_call_summary(session_id)

    @application.get("/api/summaries", response_model=list[CallSummary], dependencies=[Depends(require_api_auth)])
    async def list_summaries(
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        tier: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> list[CallSummary]:
        return db.list_call_summaries(date_from=date_from, date_to=date_to, tier=tier, outcome=outcome)

    # ---------------- CRM & Appointments Endpoints ----------------

    @application.get("/api/leads", dependencies=[Depends(require_api_auth)])
    async def list_leads() -> list[dict[str, Any]]:
        return [l.model_dump() for l in db.list_leads()]

    @application.get("/api/leads/{lead_id}/history", dependencies=[Depends(require_api_auth)])
    async def get_lead_history(lead_id: str) -> dict[str, Any]:
        lead = db.get_lead(lead_id)
        if not lead:
            raise HTTPException(status_code=404, detail="Lead not found")
        return {"lead": lead.model_dump(), "score_history": db.get_score_history(lead_id)}

    @application.get("/api/leads/export", response_class=PlainTextResponse, dependencies=[Depends(require_api_auth)])
    async def export_leads_csv() -> StreamingResponse:
        leads = db.list_leads()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "name", "email", "phone", "company", "role", "language", "score", "tier",
            "stage", "estimated_acv_usd", "budget", "authority", "need", "timeline",
            "pain_points", "objections", "last_sentiment", "updated_at",
        ])
        for l in leads:
            writer.writerow([
                l.id, l.name, l.email or "", l.phone or "", l.company or "", l.role or "",
                l.preferred_language, l.score, l.tier, l.stage, l.estimated_acv_usd,
                l.bant.budget or "", l.bant.authority or "", l.bant.need or "", l.bant.timeline or "",
                " | ".join(l.pain_points), " | ".join(l.objections), l.last_sentiment, l.updated_at,
            ])
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=fortivex_crm_leads.csv"},
        )

    @application.get("/api/crm/status", dependencies=[Depends(require_api_auth)])
    async def get_crm_status() -> dict[str, Any]:
        return crm.get_status()

    @application.get("/api/appointments", dependencies=[Depends(require_api_auth)])
    async def list_appointments() -> list[dict[str, Any]]:
        return [a.model_dump() for a in db.list_appointments()]

    @application.get("/api/appointments/slots")
    async def get_available_slots(target_date: Optional[str] = None) -> dict[str, list[str]]:
        return {"slots": db.get_available_slots(count=6, target_date=target_date)}

    @application.post("/api/appointments", dependencies=[Depends(require_api_auth), Depends(enforce_rate_limit)])
    async def create_appointment(payload: AppointmentCreate) -> dict[str, Any]:
        try:
            record = db.book_appointment(payload)
        except BookingConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except BookingValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        webhooks = crm.sync_appointment(record, "calendar.appointment.booked")
        return {"appointment": record.model_dump(), "webhook": webhooks[0] if webhooks else None}

    @application.patch("/api/appointments/{appt_id}", dependencies=[Depends(require_api_auth)])
    async def reschedule_appointment_api(appt_id: str, payload: AppointmentReschedule) -> dict[str, Any]:
        try:
            updated = db.reschedule_appointment(appt_id, payload)
        except BookingConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc))
        except BookingValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        if not updated:
            raise HTTPException(status_code=404, detail="Appointment not found")
        webhooks = crm.sync_appointment(updated, "calendar.appointment.rescheduled")
        return {"appointment": updated.model_dump(), "webhook": webhooks[0] if webhooks else None}

    @application.delete("/api/appointments/{appt_id}", dependencies=[Depends(require_api_auth)])
    async def delete_appointment(appt_id: str) -> dict[str, bool]:
        ok = db.cancel_appointment(appt_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Appointment not found")
        return {"cancelled": True}

    # ---------------- Knowledge Base & Insights Endpoints ----------------

    @application.get("/api/knowledge")
    async def list_knowledge() -> dict[str, Any]:
        return {"chunks": rag.list_chunks()}

    @application.get("/api/knowledge/search")
    async def search_knowledge(q: str, top_k: int = 4) -> dict[str, Any]:
        hits = rag.retrieve(q, top_k=top_k)
        return {
            "query": q,
            "grounded": bool(hits and hits[0].score >= 0.22),
            "results": [h.model_dump() for h in hits],
        }

    @application.post("/api/knowledge", dependencies=[Depends(require_api_auth)])
    async def add_knowledge(doc: KnowledgeDocumentInput) -> dict[str, Any]:
        if not doc.title.strip() or not doc.content.strip():
            raise HTTPException(status_code=400, detail="Title and content are required.")
        doc_id = db.save_custom_doc(doc.title, doc.category, doc.content)
        chunks = rag.add_document(doc.title, doc.category, doc.content, parent_id=doc_id)
        return {"doc_id": doc_id, "added_chunks": len(chunks), "total_chunks": len(rag.chunks)}

    @application.post("/api/knowledge/upload", dependencies=[Depends(require_api_auth)])
    async def upload_knowledge_file(
        file: UploadFile = File(...),
        category: str = Form(default="Uploaded Document"),
    ) -> dict[str, Any]:
        raw = await file.read()
        if not raw:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")
        try:
            extracted_text, chunks = rag.ingest_file(
                filename=file.filename or "uploaded.txt",
                raw_bytes=raw,
                category=category,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

        parent_id = chunks[0].parent_id if chunks else None
        db.save_custom_doc(
            title=Path(file.filename or "Uploaded").stem,
            category=category,
            content=extracted_text,
            doc_id=parent_id,
        )
        return {
            "doc_id": parent_id,
            "filename": file.filename,
            "added_chunks": len(chunks),
            "total_chunks": len(rag.chunks),
        }

    @application.delete("/api/knowledge/{doc_id}", dependencies=[Depends(require_api_auth)])
    async def delete_knowledge_doc(doc_id: str) -> dict[str, Any]:
        removed = rag.delete_document(doc_id)
        db_deleted = db.delete_custom_doc(doc_id)
        if removed == 0 and not db_deleted:
            raise HTTPException(status_code=404, detail="Knowledge document not found")
        return {"deleted": True, "removed_chunks": removed, "total_chunks": len(rag.chunks)}

    @application.get("/api/webhooks", dependencies=[Depends(require_api_auth)])
    async def list_webhooks() -> list[dict[str, Any]]:
        return db.list_webhooks()

    @application.get("/api/insights", dependencies=[Depends(require_api_auth)])
    async def get_insights(
        date_from: Optional[str] = None,
        date_to: Optional[str] = None,
        tier: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> dict[str, Any]:
        return db.get_business_insights(
            date_from=date_from, date_to=date_to, tier=tier, outcome=outcome
        ).model_dump()

    @application.post("/api/admin/backup", dependencies=[Depends(require_api_auth)])
    async def create_db_backup() -> dict[str, str]:
        backup_path = db.create_backup()
        return {"status": "ok", "backup_file": backup_path.name}

    return application


app = create_app()
