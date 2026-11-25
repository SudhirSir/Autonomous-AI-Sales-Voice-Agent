"""RAG Evaluation Benchmark: Compares Sparse TF-IDF vs Hybrid Dense+Sparse RRF Retrieval."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.rag import HybridRAGRetriever  # noqa: E402


BENCHMARK_SET = [
    {
        "query": "What are your pricing plans and annual discounts?",
        "expected_category": "Pricing & ROI",
        "should_ground": True,
    },
    {
        "query": "Are you SOC 2 Type II and GDPR compliant with zero training?",
        "expected_category": "Security & Compliance",
        "should_ground": True,
    },
    {
        "query": "How quickly can you integrate with Salesforce and HubSpot?",
        "expected_category": "Integrations",
        "should_ground": True,
    },
    {
        "query": "How does Fortivex compare against Gong or Drift?",
        "expected_category": "Competitive Intel",
        "should_ground": True,
    },
    {
        "query": "How many days does cloud or VPC deployment take?",
        "expected_category": "Platform",
        "should_ground": True,
    },
    {
        "query": "precios y cumplimiento de seguridad GDPR",
        "expected_category": "Pricing & ROI",
        "should_ground": True,
    },
    {
        "query": "Fortivex keemat aur suraksha details batao",
        "expected_category": "Pricing & ROI",
        "should_ground": True,
    },
    {
        "query": "What is the price of your electric autonomous delivery trucks?",
        "expected_category": None,
        "should_ground": False,
    },
    {
        "query": "Do you sell restaurant kitchen POS hardware?",
        "expected_category": None,
        "should_ground": False,
    },
]


def evaluate(k: int = 3) -> dict[str, dict[str, float]]:
    rag = HybridRAGRetriever()
    results = {}

    for mode in ("sparse_only", "hybrid_rrf"):
        hits_at_k = 0
        mrr_sum = 0.0
        precision_sum = 0.0
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
            if expected in cats:
                hits_at_k += 1
                rank = cats.index(expected) + 1
                mrr_sum += 1.0 / rank
            rel_count = sum(1 for c in cats if c == expected)
            precision_sum += rel_count / max(1, len(cats))

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
            f"recall@{k}": round(hits_at_k / n, 3),
            f"precision@{k}": round(precision_sum / n, 3),
            "mrr": round(mrr_sum / n, 3),
            "unknown_rejection_rate": round(correct_rejections / len(out_of_scope), 3),
        }

    return results


if __name__ == "__main__":
    report = evaluate(k=3)
    print("=== Fortivex RAG Evaluation (k=3) ===")
    for mode, metrics in report.items():
        print(f"{mode}: {metrics}")
