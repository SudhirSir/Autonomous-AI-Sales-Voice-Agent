"""Comprehensive pytest suite for Fortivex Autonomous AI Sales Voice Agent."""

from __future__ import annotations

import os
from pathlib import Path
from fastapi.testclient import TestClient

from app.main import create_app


def _make_client(tmp_path: Path) -> TestClient:
    os.environ["GEMINI_API_KEY"] = ""
    app = create_app(db_path=tmp_path / "test_fortivex.db")
    return TestClient(app)


def test_health_and_ui(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    res = client.get("/api/health")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "ok"
    assert body["knowledge_chunks"] >= 5

    ui = client.get("/")
    assert ui.status_code == 200
    assert "Fortivex" in ui.text
    assert "Autonomous AI Sales Voice Agent" in ui.text


def test_multilingual_rag_and_bant_progression(tmp_path: Path) -> None:
    client = _make_client(tmp_path)
    session_id = "test-sess-en"

    # Turn 1: Rich enterprise discovery
    r1 = client.post(
        "/api/turn",
        json={
            "session_id": session_id,
            "text": "Hi, I'm Alex Rivera, VP of Sales at ApexCloud. We have 60 sales reps wasting hours on manual CRM entry and want to roll out this month with a $40k budget.",
        },
    )
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["language"] == "en"
    assert d1["memory"]["name"] == "Alex Rivera"
    assert d1["memory"]["company"] == "ApexCloud"
    assert d1["memory"]["team_size"] == 60
    assert d1["qualification"]["score"] >= 70
    assert d1["qualification"]["tier"] == "Hot"
    assert d1["local_persistence_status"] == "saved"
    assert d1["external_crm_status"] == "not_configured"
    assert d1["crm_synced"] is False

    # Turn 2: Objection + competitor comparison -> RAG battlecard retrieval
    r2 = client.post(
        "/api/turn",
        json={
            "session_id": session_id,
            "text": "That sounds expensive compared to Gong, and we care about SOC2 security.",
        },
    )
    assert r2.status_code == 200
    d2 = r2.json()
    assert "price_objection" in d2["sentiment"]["signals"]
    assert "competitor_comparison" in d2["sentiment"]["signals"]
    assert len(d2["retrieved_chunks"]) >= 1

    # Turn 3: Automated appointment booking
    r3 = client.post(
        "/api/turn",
        json={
            "session_id": session_id,
            "text": "Sounds great, let's book a demo tomorrow at 2pm. Email me at alex@apexcloud.io.",
        },
    )
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["booked_appointment"] is not None
    assert d3["booked_appointment"]["status"] == "Confirmed"
    assert d3["qualification"]["stage"] == "Demo Scheduled"

    # Turn 4: Call summarization & business insights
    sum_res = client.post(f"/api/sessions/{session_id}/summarize")
    assert sum_res.status_code == 200
    summary = sum_res.json()
    assert summary["lead_name"] == "Alex Rivera"
    assert summary["booked_appointment"] is not None

    ins_res = client.get("/api/insights")
    assert ins_res.status_code == 200
    insights = ins_res.json()
    assert insights["booked_demos"] >= 2
    assert insights["total_pipeline_usd"] > 0


def test_multilingual_spanish_french_hindi_detection_and_rag(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    es = client.post(
        "/api/turn",
        json={
            "session_id": "sess-es",
            "text": "Hola, soy Sofía Gómez. ¿Cuáles son sus precios y cumplen con GDPR y seguridad?",
        },
    ).json()
    assert es["language"] == "es"
    assert es["tts_locale"] == "es-ES"
    assert len(es["retrieved_chunks"]) >= 1

    fr = client.post(
        "/api/turn",
        json={
            "session_id": "sess-fr",
            "text": "Bonjour, je m'appelle Claire chez DataFlow. Pouvez-vous vous intégrer à Salesforce ?",
        },
    ).json()
    assert fr["language"] == "fr"
    assert fr["tts_locale"] == "fr-FR"
    assert any("Integration" in c["category"] or "Salesforce" in c["content"] for c in fr["retrieved_chunks"])

    hi = client.post(
        "/api/turn",
        json={
            "session_id": "sess-hi",
            "text": "नमस्ते, मेरा नाम रोहन है। हम कल सुबह 10 बजे डेमो बुक करना चाहते हैं।",
        },
    ).json()
    assert hi["language"] == "hi"
    assert hi["tts_locale"] == "hi-IN"
    assert hi["booked_appointment"] is not None


def test_websocket_real_time_call_and_custom_kb(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # Add custom knowledge document
    add_res = client.post(
        "/api/knowledge",
        json={
            "title": "FedRAMP GovCloud Readiness",
            "category": "Government",
            "content": "Fortivex GovCloud supports FedRAMP High authorization with dedicated US-East air-gapped VPCs.",
        },
    )
    assert add_res.status_code == 200
    assert add_res.json()["added_chunks"] >= 1

    with client.websocket_connect("/ws/call/ws-sess-1") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "session_ready"

        ws.send_json({"type": "turn", "text": "Do you support FedRAMP GovCloud?", "input_mode": "voice"})
        thinking = ws.receive_json()
        assert thinking["type"] == "agent_thinking"

        result = ws.receive_json()
        while result["type"] == "turn_delta":
            result = ws.receive_json()
        assert result["type"] == "turn_result"
        chunks = result["data"]["retrieved_chunks"]
        assert any("FedRAMP" in c["title"] or "FedRAMP" in c["content"] for c in chunks)

        ws.send_json({"type": "end_call"})
        summary_evt = ws.receive_json()
        assert summary_evt["type"] == "call_summary"
        assert summary_evt["summary"]["session_id"] == "ws-sess-1"


def test_german_returning_lead_memory_and_csv_export(tmp_path: Path) -> None:
    client = _make_client(tmp_path)

    # First call by German prospect Lukas
    r1 = client.post(
        "/api/turn",
        json={
            "session_id": "sess-de-1",
            "text": "Hallo, ich bin Lukas bei EuroTech. Wir brauchen mehrsprachige Sprachautomatisierung für 40 Mitarbeiter. Meine E-Mail ist lukas@eurotech.de.",
        },
    ).json()
    assert r1["language"] == "de"
    assert r1["tts_locale"] == "de-DE"
    assert r1["memory"]["email"] == "lukas@eurotech.de"

    # Second call (new session) referencing same email -> cross-call returning lead memory!
    r2 = client.post(
        "/api/turn",
        json={
            "session_id": "sess-de-2",
            "text": "Hallo nochmal, hier ist lukas@eurotech.de. Wie sind die Preise?",
        },
    ).json()
    assert r2["memory"]["returning_lead"] is True
    assert r2["memory"]["prior_calls_count"] >= 1
    assert len(r2["retrieved_chunks"]) >= 1

    # Export CRM leads CSV
    csv_res = client.get("/api/leads/export")
    assert csv_res.status_code == 200
    assert "lukas@eurotech.de" in csv_res.text

