"""100-Query RAG Evaluation Benchmark comparing Sparse TF-IDF vs Hybrid RRF + Reranker."""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag import HybridRAGRetriever  # noqa: E402


def _build_100_queries() -> list[dict]:
    """Return 100 manually labelled evaluation queries across 9 categories."""
    queries: list[dict] = []

    # 1. Exact matches (14)
    exact = [
        ("What is the price of the Growth Plan?", "Pricing & ROI"),
        ("How much does the Scale Plan cost per month?", "Pricing & ROI"),
        ("What is the starting ACV for the Enterprise Plan?", "Pricing & ROI"),
        ("Do you offer a 14-day guided sandbox pilot?", "Pricing & ROI"),
        ("Is Fortivex SOC 2 Type II certified?", "Security & Compliance"),
        ("Are you ISO 27001 and GDPR compliant?", "Security & Compliance"),
        ("Do you support HIPAA BAA and PCI-DSS?", "Security & Compliance"),
        ("What encryption standard is used at rest and in transit?", "Security & Compliance"),
        ("Do you have a native Salesforce managed package?", "Integrations"),
        ("Does Fortivex integrate with HubSpot via OAuth 2.0?", "Integrations"),
        ("Do you support Google Calendar and Microsoft Outlook?", "Integrations"),
        ("How does Fortivex compare to Gong or Chorus?", "Competitive Intel"),
        ("How is Fortivex different from Drift or Intercom?", "Competitive Intel"),
        ("How many business days does standard cloud onboarding take?", "Platform"),
    ]
    for q, cat in exact:
        queries.append({"query": q, "expected_category": cat, "should_ground": True, "bucket": "exact"})

    # 2. Semantic matches (12)
    semantic = [
        ("How fast do teams see return on investment?", "Pricing & ROI"),
        ("Do we save money by paying yearly upfront?", "Pricing & ROI"),
        ("Will our conversation transcripts ever train AI models?", "Security & Compliance"),
        ("Where is European customer data stored?", "Security & Compliance"),
        ("Can we bring our own encryption keys?", "Security & Compliance"),
        ("Which telephony and SIP platforms connect to Fortivex?", "Integrations"),
        ("Can sales teams export data nightly to Snowflake or BigQuery?", "Integrations"),
        ("Does it support enterprise single sign-on via Okta?", "Integrations"),
        ("Why choose Fortivex over building an internal voice bot?", "Competitive Intel"),
        ("What happens to manual CRM logging for sales reps?", "Platform"),
        ("How many concurrent voice sessions can the architecture handle?", "Platform"),
        ("Can we deploy in a private virtual cloud?", "Platform"),
    ]
    for q, cat in semantic:
        queries.append({"query": q, "expected_category": cat, "should_ground": True, "bucket": "semantic"})

    # 3. Paraphrased questions (12)
    paraphrased = [
        ("What is the monthly fee for a team of 25 reps?", "Pricing & ROI"),
        ("For 80 seats with multilingual voice, which subscription tier fits?", "Pricing & ROI"),
        ("How much rep time is recovered each week after rollout?", "Pricing & ROI"),
        ("What lift in demo-to-close conversion do customers typically see?", "Pricing & ROI"),
        ("Tell me about PII redaction and access controls.", "Security & Compliance"),
        ("Can we get automated GDPR deletion workflows?", "Security & Compliance"),
        ("How long does connecting Salesforce take?", "Integrations"),
        ("Do you connect with Zoho CRM, Pipedrive, or Dynamics 365?", "Integrations"),
        ("What makes passive call recorders weaker than Fortivex?", "Competitive Intel"),
        ("Why are rule-based website chat widgets less effective?", "Competitive Intel"),
        ("What latency and uptime guarantees does the voice engine have?", "Platform"),
        ("How long is private cloud setup with a Solutions Architect?", "Platform"),
    ]
    for q, cat in paraphrased:
        queries.append({"query": q, "expected_category": cat, "should_ground": True, "bucket": "paraphrased"})

    # 4. Hindi questions (12)
    hindi = [
        ("Fortivex के प्राइसिंग प्लान और कीमत क्या हैं?", "Pricing & ROI"),
        ("क्या सालाना भुगतान पर कोई छूट मिलती है?", "Pricing & ROI"),
        ("14 दिन का पायलट कितने लोगों की टीम के लिए मिलता है?", "Pricing & ROI"),
        ("क्या आपका प्लेटफॉर्म SOC 2 और GDPR सुरक्षित है?", "Security & Compliance"),
        ("क्या ग्राहक डेटा का उपयोग मॉडल ट्रेनिंग में होता है?", "Security & Compliance"),
        ("क्या आप एन्क्रिप्शन और सुरक्षा प्रदान करते हैं?", "Security & Compliance"),
        ("क्या Salesforce और HubSpot इंटीग्रेशन उपलब्ध है?", "Integrations"),
        ("क्या Google Calendar और Outlook सिंक होता है?", "Integrations"),
        ("Gong की तुलना में Fortivex कैसे बेहतर है?", "Competitive Intel"),
        ("क्लाउड ऑनबोर्डिंग में कितने दिन लगते हैं?", "Platform"),
        ("क्या आप HIPAA BAA और प्राइवेसी सुरक्षा देते हैं?", "Security & Compliance"),
        ("CRM में डेटा एंट्री कैसे ऑटोमेट होती है?", "Integrations"),
    ]
    for q, cat in hindi:
        queries.append({"query": q, "expected_category": cat, "should_ground": True, "bucket": "hindi"})

    # 5. Hinglish questions (12)
    hinglish = [
        ("Fortivex ke Growth aur Scale plan ki keemat kya hai?", "Pricing & ROI"),
        ("Annual billing par kitna discount milta hai?", "Pricing & ROI"),
        ("Kya 14-day pilot aur ROI payback details milengi?", "Pricing & ROI"),
        ("Aapki security aur SOC2 compliance kaisi hai?", "Security & Compliance"),
        ("Kya humara data AI model training mein use hota hai?", "Security & Compliance"),
        ("Suraksha aur encryption ke liye AES-256 aur GDPR hai kya?", "Security & Compliance"),
        ("Salesforce aur HubSpot integration kitne minutes mein hota hai?", "Integrations"),
        ("Google Calendar aur Outlook scheduling sync hota hai kya?", "Integrations"),
        ("Gong aur Drift se Fortivex kaise alag hai?", "Competitive Intel"),
        ("Deployment mein kitne business days lagte hain?", "Platform"),
        ("Okta SSO aur Snowflake integration support hai kya?", "Integrations"),
        ("Manual CRM entry kaise kam hoti hai Fortivex se?", "Platform"),
    ]
    for q, cat in hinglish:
        queries.append({"query": q, "expected_category": cat, "should_ground": True, "bucket": "hinglish"})

    # 6. Multilingual ES / FR / DE (12)
    multilingual = [
        ("¿Cuáles son sus precios y descuentos anuales?", "Pricing & ROI"),
        ("¿Cumplen con seguridad SOC 2, GDPR y privacidad?", "Security & Compliance"),
        ("¿Tienen integración nativa con Salesforce y HubSpot?", "Integrations"),
        ("¿Cómo se comparan frente a Gong y Drift?", "Competitive Intel"),
        ("Quels sont vos tarifs Growth et Scale par mois ?", "Pricing & ROI"),
        ("Êtes-vous conformes au RGPD et certifiés ISO 27001 ?", "Security & Compliance"),
        ("Proposez-vous une intégration Salesforce et HubSpot ?", "Integrations"),
        ("Quelle est la différence avec Gong ?", "Competitive Intel"),
        ("Wie hoch sind die Preise für das Growth- und Scale-Paket?", "Pricing & ROI"),
        ("Sind Sie DSGVO-konform und bieten Sie Sicherheit mit SOC 2?", "Security & Compliance"),
        ("Unterstützen Sie Integrationen für Salesforce und HubSpot?", "Integrations"),
        ("Wie unterscheidet sich Fortivex von Gong oder In-House-Lösungen?", "Competitive Intel"),
    ]
    for q, cat in multilingual:
        queries.append({"query": q, "expected_category": cat, "should_ground": True, "bucket": "multilingual"})

    # 7. Out-of-domain questions (14)
    out_of_domain = [
        "What is the price of your electric autonomous delivery trucks?",
        "Do you sell restaurant kitchen POS hardware?",
        "Can you prescribe antibiotics for strep throat?",
        "What is the weather forecast in Mumbai tomorrow?",
        "Do you manufacture lithium-ion solar inverters?",
        "How do I bake sourdough bread at home?",
        "What is your airline baggage fee for international flights?",
        "Do you offer residential mortgage refinancing rates?",
        "Where can I buy spare tractor tires?",
        "What is the mileage of your hybrid sedan?",
        "Do you offer pet grooming subscriptions?",
        "Can you repair cracked smartphone screens?",
        "What are the shipping rates for refrigerated freight containers?",
        "Do you provide dental insurance plans for individuals?",
    ]
    for q in out_of_domain:
        queries.append({"query": q, "expected_category": None, "should_ground": False, "bucket": "out_of_domain"})

    # 8. Similar-but-incorrect / unsupported product feature distractors (12)
    similar_unsupported = [
        "Does Fortivex include free cryptocurrency wallet custody?",
        "Do you offer FedRAMP High authorization for Pentagon satellites?",
        "Does your Growth plan include 5000 complimentary Android tablets?",
        "Do you have a native integration with SAP Ariba nuclear procurement?",
        "Does Fortivex offer an automated forklift fleet tracker?",
        "What is the pricing for your drone delivery API module?",
        "Do you provide blockchain smart-contract auditing services?",
        "Does Fortivex sell physical Bluetooth headset hardware?",
        "Is there a 90-day money-back warranty on biometric door locks?",
        "Can Fortivex book commercial cargo airline tickets?",
        "Do you offer payroll tax filing software for restaurants?",
        "What is your SLA for quantum cryptographic key distribution?",
    ]
    for q in similar_unsupported:
        queries.append({"query": q, "expected_category": None, "should_ground": False, "bucket": "similar_unsupported"})

    assert len(queries) == 100, f"Expected 100 queries, got {len(queries)}"
    return queries


BENCHMARK_SET = _build_100_queries()


def _dcg(rels: list[int]) -> float:
    return sum((rel / math.log2(i + 2)) for i, rel in enumerate(rels))


def evaluate(k: int = 3) -> dict[str, dict[str, float]]:
    rag = HybridRAGRetriever()
    results: dict[str, dict[str, float]] = {}

    for mode in ("sparse_only", "hybrid_rrf_rerank"):
        hits_at_k = 0
        mrr_sum = 0.0
        precision_sum = 0.0
        ndcg_sum = 0.0
        in_scope = [x for x in BENCHMARK_SET if x["should_ground"]]
        out_of_scope = [x for x in BENCHMARK_SET if not x["should_ground"]]
        correct_rejections = 0

        for item in in_scope:
            chunks = (
                rag.retrieve_sparse_only(item["query"], top_k=k)
                if mode == "sparse_only"
                else rag.retrieve(item["query"], top_k=k)
            )
            cats = [c.category for c in chunks]
            expected = item["expected_category"]
            rels = [1 if c == expected else 0 for c in cats]

            if expected in cats:
                hits_at_k += 1
                rank = cats.index(expected) + 1
                mrr_sum += 1.0 / rank
            precision_sum += sum(rels) / max(1, len(cats))
            dcg = _dcg(rels)
            idcg = _dcg(sorted(rels, reverse=True)) if any(rels) else 1.0
            ndcg_sum += (dcg / idcg) if idcg > 0 else 0.0

        for item in out_of_scope:
            chunks = (
                rag.retrieve_sparse_only(item["query"], top_k=k, min_score=0.22)
                if mode == "sparse_only"
                else rag.retrieve(item["query"], top_k=k, min_score=0.22)
            )
            if not chunks:
                correct_rejections += 1

        n = len(in_scope)
        results[mode] = {
            "total_queries": float(len(BENCHMARK_SET)),
            "in_scope_queries": float(n),
            "out_of_scope_queries": float(len(out_of_scope)),
            f"recall@{k}": round(hits_at_k / n, 3),
            f"precision@{k}": round(precision_sum / n, 3),
            "mrr": round(mrr_sum / n, 3),
            f"ndcg@{k}": round(ndcg_sum / n, 3),
            "unknown_rejection_rate": round(correct_rejections / len(out_of_scope), 3),
        }

    # Evaluate Contradictory Document Detection
    with tempfile.TemporaryDirectory() as tmp:
        kb_dir = Path(tmp)
        (kb_dir / "pricing_a.md").write_text("# Pricing Sheet A\n- Growth Plan ($499/month) for 25 seats.", encoding="utf-8")
        (kb_dir / "pricing_b.md").write_text("# Pricing Sheet B\n- Growth Plan ($799/month) for 25 seats.", encoding="utf-8")
        c_rag = HybridRAGRetriever(knowledge_dir=kb_dir, cache_dir=Path(tmp) / "cache")
        c_hits = c_rag.retrieve("What is the Growth Plan price?", top_k=3)
        detected, note = c_rag.detect_contradictions(c_hits)
        results["contradiction_eval"] = {
            "contradiction_detected": 1.0 if detected else 0.0,
            "note": note or "",
        }

    return results


if __name__ == "__main__":
    report = evaluate(k=3)
    print("=== Fortivex V2 100-Query RAG Evaluation (k=3) ===")
    for mode, metrics in report.items():
        print(f"{mode}: {metrics}")
