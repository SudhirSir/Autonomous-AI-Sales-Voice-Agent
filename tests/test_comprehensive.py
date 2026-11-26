"""Comprehensive Phase 12 test suite covering Voice, Memory, BANT/Sentiment, RAG, Booking, CRM, Tools & Security."""

from __future__ import annotations

import base64
import io
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import docx
from fastapi.testclient import TestClient

from app.database import BookingConflictError, BookingValidationError, DatabaseStore
from app.main import create_app
from app.models import AppointmentCreate
from app.security import RateLimiter


def _make_client(tmp_path: Path) -> TestClient:
    os.environ["GEMINI_API_KEY"] = ""
    app = create_app(db_path=tmp_path / "test_fortivex.db")
    return TestClient(app)


def _next_weekday_iso(hour: int = 14) -> str:
    dt = datetime.now(timezone.utc) + timedelta(days=2)
    while dt.weekday() >= 5:
        dt += timedelta(days=1)
    return dt.replace(hour=hour, minute=0, second=0, microsecond=0).isoformat().replace("+00:00", "Z")


# ---------------- 1. VOICE (STT, TTS, WebSocket Barge-in & Streaming) ----------------

def test_voice_tts_stt_and_ws_barge_in(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # Local WAV TTS generation returns valid RIFF WAVE bytes
    tts_res = client.post("/api/tts", json={"text": "Hello from Fortivex", "language": "en", "rate": 1.0})
    assert tts_res.status_code == 200
    assert tts_res.headers["content-type"] == "audio/wav"
    assert tts_res.content[:4] == b"RIFF"

    # STT fallback when no external STT key is configured
    stt_res = client.post(
        "/api/stt",
        files={"audio": ("sample.webm", b"dummy-audio-bytes", "audio/webm")},
        data={"language": "en"},
    )
    assert stt_res.status_code == 200
    assert stt_res.json()["fallback_used"] is True

    # Mocked Deepgram STT via env key
    with patch.dict(os.environ, {"DEEPGRAM_API_KEY": "dg-test", "STT_PROVIDER": "deepgram"}):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "results": {"channels": [{"alternatives": [{"transcript": "Hello pricing please"}]}]}
        }
        mock_resp.raise_for_status = MagicMock()
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.return_value = mock_resp
        with patch("app.voice.httpx.Client", return_value=mock_client):
            dg_res = client.post(
                "/api/stt",
                files={"audio": ("speech.wav", b"RIFFDATA", "audio/wav")},
            )
            assert dg_res.status_code == 200
            assert dg_res.json()["transcript"] == "Hello pricing please"
            assert dg_res.json()["provider"] == "deepgram"

    # WebSocket streaming turn_delta + barge_in
    with client.websocket_connect("/ws/call/ws-voice-1") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "session_ready"

        ws.send_json({"type": "barge_in"})
        assert ws.receive_json()["type"] == "tts_stop"

        ws.send_json({"type": "turn", "text": "What is your pricing?"})
        assert ws.receive_json()["type"] == "agent_thinking"
        deltas = []
        while True:
            evt = ws.receive_json()
            if evt["type"] == "turn_delta":
                deltas.append(evt["delta"])
            elif evt["type"] == "turn_result":
                assert len(deltas) >= 1
                assert evt["data"]["reply_text"]
                break


# ---------------- 2. MEMORY, HINGLISH & CUSTOMER ISOLATION ----------------

def test_hinglish_three_tier_memory_and_customer_isolation(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # Hinglish conversation with phone number
    r1 = client.post(
        "/api/turn",
        json={
            "session_id": "sess-hinglish-1",
            "text": "Namaste, mera naam Vikram Mehta hai. Mujhe Fortivex ke pricing aur security details chahiye, humari 45 logon ki team hai aur phone +91 98765 43210 hai.",
        },
    ).json()
    assert r1["language"] == "hinglish"
    assert r1["memory"]["name"] == "Vikram Mehta"
    assert r1["memory"]["phone"] == "+91 98765 43210"
    assert r1["memory"]["team_size"] == 45
    assert r1["memory_snapshot"]["session_facts"]["team_size"] == 45

    # Generate summary for call 1
    client.post("/api/sessions/sess-hinglish-1/summarize")

    # Returning call matched by phone number retrieves long-term memory
    r2 = client.post(
        "/api/turn",
        json={
            "session_id": "sess-hinglish-2",
            "text": "Hi again, my phone is +91 98765 43210. Let's schedule a demo tomorrow at 11am.",
        },
    ).json()
    assert r2["memory"]["returning_lead"] is True
    assert r2["memory"]["name"] == "Vikram Mehta"
    assert len(r2["memory_snapshot"]["previous_interactions"]) >= 1

    # Different customer isolation: must NOT inherit Vikram's details
    r3 = client.post(
        "/api/turn",
        json={
            "session_id": "sess-isolated-3",
            "text": "Hello, I'm Sarah Jenkins at CloudNine. We need Salesforce integration.",
        },
    ).json()
    assert r3["memory"]["name"] == "Sarah Jenkins"
    assert r3["memory"]["phone"] is None
    assert r3["memory"]["returning_lead"] is False


# ---------------- 3. BANT DISQUALIFICATION & SENTIMENT VS INTENT ----------------

def test_bant_disqualification_and_sentiment_vs_intent(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # Frustrated tone + high purchase intent should remain distinct!
    r_frust = client.post(
        "/api/turn",
        json={
            "session_id": "sess-frust",
            "text": "I am frustrated with how slow Gong is, it's too expensive, but we are ready to start a 50 seat pilot this week if pricing works.",
        },
    ).json()
    assert r_frust["sentiment"]["frustration_detected"] is True
    assert r_frust["sentiment"]["label"] == "Frustrated"
    assert r_frust["sentiment"]["intent_score"] >= 60
    assert r_frust["sentiment"]["purchase_readiness"] in {"Medium", "High"}

    # Disqualified lead
    r_disq = client.post(
        "/api/turn",
        json={
            "session_id": "sess-disq",
            "text": "Wrong number, we have no budget and are not interested at all. Stop calling.",
        },
    ).json()
    assert r_disq["qualification"]["tier"] == "Disqualified"
    assert r_disq["qualification"]["stage"] == "Disqualified"

    # Score history persisted
    lead_id = r_frust["memory"]["lead_id"]
    hist = client.get(f"/api/leads/{lead_id}/history").json()
    assert len(hist["score_history"]) >= 1


# ---------------- 4. HYBRID RAG: DOCX UPLOAD, DELETION & OUT-OF-SCOPE GUARDRAIL ----------------

def test_rag_docx_upload_delete_and_unknown_guardrail(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # Create in-memory DOCX document
    doc = docx.Document()
    doc.add_heading("Fortivex Healthcare HIPAA SLA", level=1)
    doc.add_paragraph("Fortivex provides a signed HIPAA BAA and 99.995% healthcare SLA for clinical voice deployments.")
    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)

    up = client.post(
        "/api/knowledge/upload",
        files={"file": ("healthcare_sla.docx", buf.getvalue(), "application/vnd.openxmlformats-officedocument.wordprocessingml.document")},
        data={"category": "Healthcare"},
    )
    assert up.status_code == 200
    doc_id = up.json()["doc_id"]
    assert up.json()["added_chunks"] >= 1

    # Search retrieves uploaded DOCX chunk with both sparse & dense scores
    srch = client.get("/api/knowledge/search?q=HIPAA BAA clinical voice SLA").json()
    assert srch["grounded"] is True
    assert any("HIPAA" in r["content"] for r in srch["results"])

    # Out-of-scope guardrail refuses to invent answers
    oos = client.post(
        "/api/turn",
        json={
            "session_id": "sess-oos",
            "text": "What is the price of your electric autonomous delivery trucks?",
        },
    ).json()
    assert oos["kb_grounded"] is False
    assert "don't have documented details" in oos["reply_text"].lower() or "knowledge base" in oos["reply_text"].lower()

    # Delete uploaded document
    del_res = client.delete(f"/api/knowledge/{doc_id}")
    assert del_res.status_code == 200
    assert del_res.json()["deleted"] is True


# ---------------- 5. ATOMIC CONCURRENT BOOKING, VALIDATION, RESCHEDULE & CANCEL ----------------

def test_atomic_concurrent_booking_reschedule_and_validation(tmp_path: Path) -> None:
    db_path = tmp_path / "concurrent.db"
    client = TestClient(create_app(db_path=db_path))

    # Reject past date
    past_res = client.post(
        "/api/appointments",
        json={"lead_name": "Past User", "slot_iso": "2020-01-01T14:00:00Z"},
    )
    assert past_res.status_code == 422

    # Reject outside business hours (e.g. 22:00 UTC)
    late_slot = _next_weekday_iso(hour=22)
    late_res = client.post(
        "/api/appointments",
        json={"lead_name": "Late User", "slot_iso": late_slot},
    )
    assert late_res.status_code == 422

    # Concurrent double-booking protection using raw DatabaseStore across threads
    db = DatabaseStore(db_path=db_path)
    target_slot = _next_weekday_iso(hour=14)

    def attempt_book(idx: int) -> str:
        try:
            rec = db.book_appointment(
                AppointmentCreate(lead_name=f"Concurrent {idx}", slot_iso=target_slot)
            )
            return f"ok:{rec.id}"
        except BookingConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(attempt_book, range(4)))

    ok_outcomes = [o for o in outcomes if o.startswith("ok:")]
    conflict_outcomes = [o for o in outcomes if o == "conflict"]
    assert len(ok_outcomes) == 1
    assert len(conflict_outcomes) == 3

    appt_id = ok_outcomes[0].split(":", 1)[1]

    # Reschedule to another valid slot
    new_slot = _next_weekday_iso(hour=16)
    resched = client.patch(f"/api/appointments/{appt_id}", json={"slot_iso": new_slot})
    assert resched.status_code == 200
    assert resched.json()["appointment"]["status"] == "Rescheduled"
    assert resched.json()["appointment"]["slot_iso"] == new_slot

    # Cancel appointment
    canc = client.delete(f"/api/appointments/{appt_id}")
    assert canc.status_code == 200
    assert canc.json()["cancelled"] is True


# ---------------- 6. CRM ADAPTER RETRIES, SECURITY AUTH, RATE LIMIT & FILTERED INSIGHTS ----------------

def test_crm_webhook_retries_security_auth_and_filtered_insights(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # Test webhook retries (503 -> 503 -> 200) via mocked httpx.Client.post
    resp_503 = MagicMock(status_code=503, text="Service Unavailable")
    resp_200 = MagicMock(status_code=200, text="OK")
    with patch.dict(os.environ, {"CRM_WEBHOOK_URL": "https://crm.example.test/hook"}):
        mock_client = MagicMock()
        mock_client.__enter__.return_value.post.side_effect = [resp_503, resp_503, resp_200]
        with patch("app.crm.httpx.Client", return_value=mock_client):
            turn = client.post(
                "/api/turn",
                json={"session_id": "sess-crm-retry", "text": "Hi, I'm Priya Nair at FinCore. We want a demo."},
            )
            assert turn.status_code == 200
            assert "webhook" in turn.json()["crm_mode"]

        status = client.get("/api/crm/status").json()
        assert any(log["attempts"] == 3 and log["status"] == "success" for log in status["recent_sync_logs"])

    # API Token authentication enforcement
    with patch.dict(os.environ, {"FORTIVEX_API_TOKEN": "super-secret-token"}):
        unauth = client.get("/api/leads")
        assert unauth.status_code == 401
        auth_ok = client.get("/api/leads", headers={"Authorization": "Bearer super-secret-token"})
        assert auth_ok.status_code == 200

    # Rate limiter enforcement
    rl = RateLimiter(max_per_minute=2)
    assert rl.check("1.2.3.4") is True
    assert rl.check("1.2.3.4") is True
    assert rl.check("1.2.3.4") is False

    # Filtered Business Insights & Backup
    client.post("/api/sessions/sess-crm-retry/summarize")
    ins_hot = client.get("/api/insights?tier=Hot").json()
    assert ins_hot["hot_leads"] >= 1
    assert "agent_performance" in ins_hot

    bkp = client.post("/api/admin/backup")
    assert bkp.status_code == 200
    assert bkp.json()["status"] == "ok"
