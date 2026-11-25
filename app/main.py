"""FastAPI application and WebSocket real-time voice pipeline for Fortivex."""

from __future__ import annotations

import csv
import io
import os
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from app.agent import SalesVoiceAgent
from app.config import get_runtime_gemini_key, settings
from app.database import DatabaseStore
from app.models import (
    AppointmentCreate,
    CallSummary,
    KnowledgeDocumentInput,
    RuntimeConfigUpdate,
    TurnRequest,
    TurnResponse,
)
from app.rag import HybridRAGRetriever

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


def create_app(
    db_path: str | Path | None = None,
    knowledge_dir: str | Path | None = None,
) -> FastAPI:
    """Create and configure the Fortivex FastAPI app (factory supports isolated test DBs)."""
    application = FastAPI(title=settings.app_name, version="1.0.0")
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    db = DatabaseStore(db_path=db_path)
    rag = HybridRAGRetriever(knowledge_dir=knowledge_dir)
    agent = SalesVoiceAgent(db=db, rag=rag)

    application.state.db = db
    application.state.rag = rag
    application.state.agent = agent

    if STATIC_DIR.exists():
        application.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    @application.get("/", include_in_schema=False)
    async def serve_ui():
        index_path = STATIC_DIR / "index.html"
        if not index_path.exists():
            return {"message": "Fortivex AI Sales Voice Agent API is running. UI not found."}
        return FileResponse(str(index_path))

    @application.get("/api/health")
    async def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "app": settings.app_name,
            "ai_engine": "gemini" if get_runtime_gemini_key() else "local-hybrid",
            "gemini_model": settings.gemini_model,
            "knowledge_chunks": len(rag.chunks),
        }

    @application.post("/api/config")
    async def update_runtime_config(payload: RuntimeConfigUpdate) -> dict[str, Any]:
        if payload.gemini_api_key is not None:
            os.environ["GEMINI_API_KEY"] = payload.gemini_api_key.strip()
        if payload.gemini_model:
            settings.gemini_model = payload.gemini_model.strip()
        if payload.crm_webhook_url is not None:
            settings.crm_webhook_url = payload.crm_webhook_url.strip()
        return {
            "status": "updated",
            "ai_engine": "gemini" if get_runtime_gemini_key() else "local-hybrid",
            "gemini_model": settings.gemini_model,
        }

    # ---------------- Real-Time Voice & Turn Endpoints ----------------

    @application.post("/api/turn", response_model=TurnResponse)
    async def process_turn_http(req: TurnRequest) -> TurnResponse:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail="Utterance text cannot be empty.")
        return agent.process_turn(req)

    @application.websocket("/ws/call/{session_id}")
    async def websocket_voice_call(websocket: WebSocket, session_id: str) -> None:
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

                if msg_type == "turn":
                    text = (data.get("text") or "").strip()
                    if not text:
                        continue
                    await websocket.send_json({"type": "agent_thinking", "session_id": session_id})
                    req = TurnRequest(
                        session_id=session_id,
                        text=text,
                        language_override=data.get("language_override"),
                        input_mode=data.get("input_mode", "voice"),
                    )
                    resp = agent.process_turn(req)
                    await websocket.send_json({"type": "turn_result", "data": resp.model_dump()})
        except WebSocketDisconnect:
            pass

    @application.post("/api/stt")
    async def server_stt(
        audio: UploadFile = File(...),
        language: Optional[str] = Form(default=None),
    ) -> dict[str, Any]:
        """Server-side audio transcription endpoint (uses Gemini Audio when key is present)."""
        raw_bytes = await audio.read()
        api_key = get_runtime_gemini_key()
        if api_key and raw_bytes:
            try:
                from google import genai
                from google.genai import types

                client = genai.Client(api_key=api_key)
                mime = audio.content_type or "audio/webm"
                prompt = "Transcribe this spoken sales prospect audio verbatim. Return only the transcript text."
                if language:
                    prompt += f" Expected language code: {language}."
                resp = client.models.generate_content(
                    model=settings.gemini_model,
                    contents=[
                        types.Part.from_bytes(data=raw_bytes, mime_type=mime),
                        prompt,
                    ],
                )
                transcript = (resp.text or "").strip()
                return {"transcript": transcript, "engine": "gemini-audio", "bytes": len(raw_bytes)}
            except Exception as exc:
                return {
                    "transcript": "",
                    "engine": "fallback",
                    "error": str(exc),
                    "bytes": len(raw_bytes),
                }

        return {
            "transcript": "",
            "engine": "browser-webspeech-recommended",
            "bytes": len(raw_bytes),
            "message": "No GEMINI_API_KEY configured for server audio STT; browser Web Speech API handles real-time STT.",
        }

    # ---------------- Session & Summary Endpoints ----------------

    @application.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict[str, Any]:
        sess = db.get_or_create_session(session_id)
        messages = db.get_messages(session_id)
        appt = db.get_session_appointment(session_id)
        summary = db.get_call_summary(session_id)
        return {
            "session_id": session_id,
            "language": sess["language"],
            "memory": sess["memory"].model_dump(),
            "qualification": sess["qualification"].model_dump(),
            "messages": [m.model_dump() for m in messages],
            "booked_appointment": appt.model_dump() if appt else None,
            "summary": summary.model_dump() if summary else None,
        }

    @application.post("/api/sessions/{session_id}/summarize", response_model=CallSummary)
    async def summarize_session(session_id: str) -> CallSummary:
        return agent.generate_call_summary(session_id)

    @application.get("/api/summaries", response_model=list[CallSummary])
    async def list_summaries() -> list[CallSummary]:
        return db.list_call_summaries()

    # ---------------- CRM & Appointments Endpoints ----------------

    @application.get("/api/leads")
    async def list_leads() -> list[dict[str, Any]]:
        return [l.model_dump() for l in db.list_leads()]

    @application.get("/api/leads/export", response_class=PlainTextResponse)
    async def export_leads_csv() -> StreamingResponse:
        leads = db.list_leads()
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "id", "name", "email", "company", "role", "language", "score", "tier",
            "stage", "estimated_acv_usd", "budget", "authority", "need", "timeline",
            "pain_points", "objections", "last_sentiment", "updated_at",
        ])
        for l in leads:
            writer.writerow([
                l.id, l.name, l.email or "", l.company or "", l.role or "", l.preferred_language,
                l.score, l.tier, l.stage, l.estimated_acv_usd,
                l.bant.budget or "", l.bant.authority or "", l.bant.need or "", l.bant.timeline or "",
                " | ".join(l.pain_points), " | ".join(l.objections), l.last_sentiment, l.updated_at,
            ])
        buf.seek(0)
        return StreamingResponse(
            iter([buf.getvalue()]),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=fortivex_crm_leads.csv"},
        )

    @application.get("/api/appointments")
    async def list_appointments() -> list[dict[str, Any]]:
        return [a.model_dump() for a in db.list_appointments()]

    @application.get("/api/appointments/slots")
    async def get_available_slots() -> dict[str, list[str]]:
        return {"slots": db.get_available_slots(count=6)}

    @application.post("/api/appointments")
    async def create_appointment(payload: AppointmentCreate) -> dict[str, Any]:
        record, webhook = db.book_appointment(payload)
        return {"appointment": record.model_dump(), "webhook": webhook}

    @application.delete("/api/appointments/{appt_id}")
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
    async def search_knowledge(q: str) -> dict[str, Any]:
        hits = rag.retrieve(q, top_k=4)
        return {"query": q, "results": [h.model_dump() for h in hits]}

    @application.post("/api/knowledge")
    async def add_knowledge(doc: KnowledgeDocumentInput) -> dict[str, Any]:
        if not doc.title.strip() or not doc.content.strip():
            raise HTTPException(status_code=400, detail="Title and content are required.")
        db.save_custom_doc(doc.title, doc.category, doc.content)
        chunks = rag.add_document(doc.title, doc.category, doc.content)
        return {"added_chunks": len(chunks), "total_chunks": len(rag.chunks)}

    @application.get("/api/webhooks")
    async def list_webhooks() -> list[dict[str, Any]]:
        return db.list_webhooks()

    @application.get("/api/insights")
    async def get_insights() -> dict[str, Any]:
        return db.get_business_insights().model_dump()

    return application


app = create_app()
