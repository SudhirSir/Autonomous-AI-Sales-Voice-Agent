"""Hybrid TF-IDF + Multilingual Query Expansion RAG Knowledge Retriever."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.config import settings
from app.models import RetrievedChunk


# Cross-lingual term bridge so Spanish, French, German, and Hindi queries retrieve English KB chunks cleanly
MULTILINGUAL_TERM_MAP: dict[str, list[str]] = {
    # Pricing / Cost / Budget
    "precio": ["pricing", "plan", "month", "cost", "discount"],
    "precios": ["pricing", "plan", "month", "cost"],
    "costo": ["pricing", "cost", "roi"],
    "presupuesto": ["pricing", "budget", "plan"],
    "descuento": ["discount", "annual", "pilot"],
    "prix": ["pricing", "plan", "month", "cost"],
    "tarif": ["pricing", "plan", "cost"],
    "coût": ["pricing", "cost", "roi"],
    "remise": ["discount", "annual"],
    "preis": ["pricing", "plan", "month", "cost"],
    "preise": ["pricing", "plan", "cost"],
    "kosten": ["pricing", "cost", "roi"],
    "rabatt": ["discount", "annual"],
    "कीमत": ["pricing", "plan", "cost"],
    "दाम": ["pricing", "cost"],
    "बजट": ["pricing", "budget", "plan"],
    "छूट": ["discount", "pilot"],
    # Security / Compliance
    "seguridad": ["security", "soc", "encryption", "gdpr", "hipaa"],
    "cumplimiento": ["compliance", "soc", "gdpr", "hipaa", "iso"],
    "privacidad": ["privacy", "gdpr", "pii", "zero-training"],
    "sécurité": ["security", "soc", "encryption", "gdpr", "hipaa"],
    "conformité": ["compliance", "soc", "gdpr", "iso"],
    "confidentialité": ["privacy", "gdpr", "zero-training"],
    "sicherheit": ["security", "soc", "encryption", "gdpr", "hipaa"],
    "datenschutz": ["privacy", "gdpr", "pii", "iso"],
    "सुरक्षा": ["security", "soc", "encryption", "gdpr"],
    "गोपनीयता": ["privacy", "gdpr", "zero-training"],
    # Integrations / CRM
    "integración": ["integration", "salesforce", "hubspot", "crm", "api"],
    "integraciones": ["integrations", "salesforce", "hubspot", "crm"],
    "calendario": ["calendar", "google", "outlook", "calendly"],
    "intégration": ["integration", "salesforce", "hubspot", "crm", "api"],
    "calendrier": ["calendar", "google", "outlook"],
    "integrationen": ["integrations", "salesforce", "hubspot", "crm", "api"],
    "एकीकरण": ["integration", "salesforce", "hubspot", "crm"],
    # Pilot / Demo / Deployment
    "prueba": ["pilot", "sandbox", "14-day"],
    "implementación": ["deployment", "onboarding", "business days", "vpc"],
    "essai": ["pilot", "sandbox", "14-day"],
    "déploiement": ["deployment", "onboarding", "vpc"],
    "bereitstellung": ["deployment", "onboarding", "vpc"],
    "टेस्ट": ["pilot", "sandbox"],
    "डेमो": ["pilot", "demo", "sandbox"],
    # Competitors & ROI
    "competencia": ["gong", "drift", "intercom", "in-house"],
    "concurrence": ["gong", "drift", "intercom", "in-house"],
    "wettbewerb": ["gong", "drift", "intercom", "in-house"],
    "retorno": ["roi", "payback", "conversion", "hours"],
}


@dataclass
class KnowledgeChunk:
    doc_id: str
    title: str
    category: str
    content: str


class HybridRAGRetriever:
    """In-memory hybrid TF-IDF + lexical boost retriever with dynamic indexing."""

    def __init__(self, knowledge_dir: str | Path | None = None) -> None:
        self.knowledge_dir = Path(knowledge_dir or settings.knowledge_dir)
        self.chunks: list[KnowledgeChunk] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        self.load_directory(self.knowledge_dir)

    def load_directory(self, directory: Path) -> None:
        """Load all markdown and text files from disk and index their sections."""
        self.chunks.clear()
        if directory.exists():
            for path in sorted(directory.glob("*")):
                if path.suffix.lower() in {".md", ".txt"} and path.is_file():
                    raw = path.read_text(encoding="utf-8")
                    category = self._infer_category(path.stem)
                    self._ingest_markdown(doc_prefix=path.stem, category=category, raw_text=raw)
        self._rebuild_index()

    def _infer_category(self, stem: str) -> str:
        lower = stem.lower()
        if "pricing" in lower or "roi" in lower:
            return "Pricing & ROI"
        if "security" in lower or "compliance" in lower:
            return "Security & Compliance"
        if "integration" in lower or "api" in lower:
            return "Integrations"
        if "competitor" in lower or "battlecard" in lower:
            return "Competitive Intel"
        if "platform" in lower or "overview" in lower:
            return "Platform"
        return "Knowledge Base"

    def _ingest_markdown(self, doc_prefix: str, category: str, raw_text: str) -> list[KnowledgeChunk]:
        """Split a markdown document by level-2 headings into focused chunks."""
        lines = raw_text.splitlines()
        doc_title = doc_prefix.replace("_", " ").title()
        for line in lines:
            if line.startswith("# "):
                doc_title = line[2:].strip()
                break

        sections: list[tuple[str, str]] = []
        current_heading = doc_title
        buffer: list[str] = []

        for line in lines:
            if line.startswith("## "):
                if buffer:
                    body = "\n".join(buffer).strip()
                    if body:
                        sections.append((current_heading, body))
                current_heading = f"{doc_title} — {line[3:].strip()}"
                buffer = []
            elif not line.startswith("# "):
                buffer.append(line)

        if buffer:
            body = "\n".join(buffer).strip()
            if body:
                sections.append((current_heading, body))

        if not sections and raw_text.strip():
            sections.append((doc_title, raw_text.strip()))

        created: list[KnowledgeChunk] = []
        for idx, (heading, body) in enumerate(sections, start=1):
            chunk = KnowledgeChunk(
                doc_id=f"{doc_prefix}#{idx}",
                title=heading,
                category=category,
                content=body,
            )
            self.chunks.append(chunk)
            created.append(chunk)
        return created

    def add_document(self, title: str, category: str, content: str) -> list[KnowledgeChunk]:
        """Add a custom knowledge document at runtime and refresh the vector index."""
        doc_prefix = f"custom_{uuid.uuid4().hex[:8]}"
        markdown_text = f"# {title}\n\n{content}" if not content.lstrip().startswith("#") else content
        created = self._ingest_markdown(doc_prefix=doc_prefix, category=category or "Custom", raw_text=markdown_text)
        self._rebuild_index()
        return created

    def _rebuild_index(self) -> None:
        if not self.chunks:
            self.vectorizer = None
            self.matrix = None
            return
        corpus = [f"{c.title}\n{c.category}\n{c.content}" for c in self.chunks]
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.matrix = self.vectorizer.fit_transform(corpus)

    def _expand_query(self, query: str) -> tuple[str, list[str]]:
        """Expand multilingual or shorthand terms into English KB anchor terms."""
        tokens = re.findall(r"[\w\u0900-\u097F-]+", query.lower())
        expanded_terms: list[str] = list(tokens)
        for tok in tokens:
            if tok in MULTILINGUAL_TERM_MAP:
                expanded_terms.extend(MULTILINGUAL_TERM_MAP[tok])
        # Synonym expansions for common English sales intents
        synonym_rules = {
            "price": ["pricing", "plan", "month", "growth", "scale", "enterprise"],
            "cost": ["pricing", "plan", "roi", "acv"],
            "how much": ["pricing", "plan", "month", "enterprise"],
            "soc2": ["soc 2", "compliance", "security", "type ii"],
            "soc": ["soc 2", "compliance", "security"],
            "safe": ["security", "encryption", "zero data training"],
            "crm": ["salesforce", "hubspot", "pipedrive", "zoho", "dynamics"],
            "gong": ["gong", "chorus", "battlecards", "autonomous"],
            "pilot": ["14-day", "sandbox", "pilot", "trial"],
            "trial": ["14-day", "sandbox", "pilot"],
            "deploy": ["deployment", "onboarding", "business days", "vpc"],
            "onboard": ["deployment", "onboarding", "business days"],
        }
        q_lower = query.lower()
        for trigger, additions in synonym_rules.items():
            if trigger in q_lower:
                expanded_terms.extend(additions)

        deduped = list(dict.fromkeys(expanded_terms))
        return " ".join(deduped), deduped

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.08) -> list[RetrievedChunk]:
        """Return top-k relevant knowledge chunks with normalized similarity scores."""
        if not self.chunks or self.vectorizer is None or self.matrix is None or not query.strip():
            return []

        expanded_query, query_tokens = self._expand_query(query)
        q_vec = self.vectorizer.transform([expanded_query])
        sims = cosine_similarity(q_vec, self.matrix).flatten()

        # Lexical overlap bonus for strong recall on acronyms/entities (SOC2, GDPR, HubSpot, etc.)
        query_token_set = {t for t in query_tokens if len(t) >= 3}
        scored: list[tuple[float, KnowledgeChunk, list[str]]] = []

        for idx, chunk in enumerate(self.chunks):
            chunk_text_lower = f"{chunk.title} {chunk.content}".lower()
            matched = [t for t in query_token_set if t in chunk_text_lower]
            lexical_bonus = min(0.25, 0.05 * len(matched))
            combined = float(min(1.0, sims[idx] + lexical_bonus))
            if combined >= min_score:
                scored.append((combined, chunk, sorted(matched)[:6]))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                category=chunk.category,
                content=chunk.content,
                score=round(score, 4),
                matched_terms=matched,
            )
            for score, chunk, matched in scored[:top_k]
        ]

    def list_chunks(self) -> list[dict[str, str]]:
        """List all indexed chunks for UI inspection."""
        return [
            {
                "doc_id": c.doc_id,
                "title": c.title,
                "category": c.category,
                "preview": c.content[:180] + ("..." if len(c.content) > 180 else ""),
            }
            for c in self.chunks
        ]
