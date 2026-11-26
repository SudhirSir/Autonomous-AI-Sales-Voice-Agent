"""Phase 15 End-to-End Acceptance Demonstration & Live Gemini Integration Check."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import get_runtime_gemini_key  # noqa: E402
from app.main import create_app  # noqa: E402


def run_demo(use_live_gemini: bool = False) -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        if not use_live_gemini:
            os.environ["GEMINI_API_KEY"] = ""

        app = create_app(db_path=Path(tmp) / "phase15_demo.db")
        client = TestClient(app)

        print("1. Health & Voice Providers:", client.get("/api/health").json())

        # Step 1: Hindi/Hinglish Voice/Text Discovery + RAG Pricing + BANT Extraction
        t1 = client.post(
            "/api/turn",
            json={
                "session_id": "demo-hindi-1",
                "text": "Namaste, mera naam Rohan Verma hai, phone +91 98200 11223. Humari 45 logon ki sales team hai aur hum Fortivex ke pricing plans aur SOC2 security details jaanna chahte hain.",
            },
        ).json()
        print("2. Turn 1 [lang, engine, score, kb_grounded]:", t1["language"], t1["ai_engine"], t1["qualification"]["score"], t1["kb_grounded"])
        print("   Reply:", t1["reply_text"])

        # Step 2: Genuine TTS Synthesis (Windows SAPI / Piper / Cloud)
        tts_res = client.post("/api/tts", json={"text": t1["reply_text"][:140], "language": t1["language"]})
        print("3. TTS Synthesis:", tts_res.status_code, tts_res.headers.get("content-type"), "provider=", tts_res.headers.get("x-tts-provider"), "bytes=", len(tts_res.content))

        # Step 3: Genuine Demo Booking Request in IST -> UTC Atomic Persistence
        t2 = client.post(
            "/api/turn",
            json={
                "session_id": "demo-hindi-1",
                "text": "Bahut badhiya, please kal 3:00pm IST par ek live demo book kar dijiye. Email rohan@bharatsales.in hai.",
                "timezone": "Asia/Kolkata",
            },
        ).json()
        print("4. Turn 2 [booked_appointment]:", t2["booked_appointment"])

        # Step 4: Call Summary & Real-Time Analytics
        summary = client.post("/api/sessions/demo-hindi-1/summarize").json()
        print("5. Summary [duration_sec, user_turns, outcome]:", summary["actual_duration_seconds"], summary["user_turn_count"], summary["appointment_outcome"])

        # Step 5: Returning Call Cross-Session Memory Retrieval
        t3 = client.post(
            "/api/turn",
            json={"session_id": "demo-hindi-2", "text": "Hi again, mera phone +91 98200 11223 hai."},
        ).json()
        print("6. Returning Call [returning_lead, name]:", t3["memory"]["returning_lead"], t3["memory"]["name"])

        # Step 6: WebSocket Barge-In Interruption Demonstration
        with client.websocket_connect("/ws/call/demo-hindi-ws") as ws:
            ws.receive_json()
            ws.send_json({"type": "barge_in"})
            stop_msg = ws.receive_json()
            print("7. WebSocket Barge-In:", stop_msg)

        # Step 7: Failed External CRM Request Reported Honestly (Never Hardcoded Success)
        fail_401 = MagicMock(status_code=401, text="Unauthorized HubSpot Token")
        mock_http = MagicMock()
        mock_http.__enter__.return_value.post.return_value = fail_401
        with patch.dict(os.environ, {"HUBSPOT_ACCESS_TOKEN": "invalid-demo-token"}):
            with patch("app.crm.httpx.Client", return_value=mock_http):
                t_fail = client.post(
                    "/api/turn",
                    json={"session_id": "demo-crm-fail", "text": "Hi, email test@failcorp.io."},
                ).json()
                print("8. Failed CRM Check [crm_synced, local, external]:", t_fail["crm_synced"], t_fail["local_persistence_status"], t_fail["external_crm_status"])


if __name__ == "__main__":
    live = "--live-gemini" in sys.argv and bool(get_runtime_gemini_key())
    run_demo(use_live_gemini=live)
