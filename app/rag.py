"""Hybrid Dense Semantic + Sparse TF-IDF RAG Engine with PDF/DOCX/TXT/MD Ingestion."""

from __future__ import annotations

import io
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

from app.config import get_runtime_gemini_key, settings
from app.models import RetrievedChunk

logger = logging.getLogger(__name__)


# Cross-lingual term bridge (ES, FR, DE, HI, Hinglish -> English KB anchor terms)
MULTILINGUAL_TERM_MAP: dict[str, list[str]] = {
    # Pricing / Cost
    "precio": ["pricing", "plan", "month", "cost", "discount"],
    "precios": ["pricing", "plan", "month", "cost"],
    "costo": ["pricing", "cost", "roi"],
    "presupuesto": ["pricing", "budget", "plan"],
    "descuento": ["discount", "annual", "pilot"],
    "prix": ["pricing", "plan", "month", "cost"],
    "tarif": ["pricing", "plan", "cost"],
    "coût": ["pricing", "cost", "roi"],
    "preis": ["pricing", "plan", "month", "cost"],
    "preise": ["pricing", "plan", "cost"],
    "kosten": ["pricing", "cost", "roi"],
    "कीमत": ["pricing", "plan", "cost"],
    "दाम": ["pricing", "cost"],
    "बजट": ["pricing", "budget", "plan"],
    "keemat": ["pricing", "plan", "cost"],
    "paisa": ["pricing", "cost", "roi"],
    "mehnga": ["pricing", "discount", "roi"],
    # Security / Compliance
    "seguridad": ["security", "soc", "encryption", "gdpr", "hipaa"],
    "cumplimiento": ["compliance", "soc", "gdpr", "hipaa", "iso"],
    "privacidad": ["privacy", "gdpr", "pii", "zero-training"],
    "sécurité": ["security", "soc", "encryption", "gdpr", "hipaa"],
    "conformité": ["compliance", "soc", "gdpr", "iso"],
    "sicherheit": ["security", "soc", "encryption", "gdpr", "hipaa"],
    "datenschutz": ["privacy", "gdpr", "pii", "iso"],
    "सुरक्षा": ["security", "soc", "encryption", "gdpr"],
    "suraksha": ["security", "soc", "encryption", "gdpr"],
    # Integrations / CRM
    "integración": ["integration", "salesforce", "hubspot", "crm", "api"],
    "integraciones": ["integrations", "salesforce", "hubspot", "crm"],
    "intégration": ["integration", "salesforce", "hubspot", "crm", "api"],
    "integrationen": ["integrations", "salesforce", "hubspot", "crm", "api"],
    "एकीकरण": ["integration", "salesforce", "hubspot", "crm"],
    # Pilot / Deployment
    "prueba": ["pilot", "sandbox", "14-day"],
    "implementación": ["deployment", "onboarding", "business days", "vpc"],
    "essai": ["pilot", "sandbox", "14-day"],
    "déploiement": ["deployment", "onboarding", "vpc"],
    "bereitstellung": ["deployment", "onboarding", "vpc"],
    "डेमो": ["pilot", "demo", "sandbox"],
    "retorno": ["roi", "payback", "conversion", "hours"],
}


@dataclass
class KnowledgeChunk:
    doc_id: str
    parent_id: str
    title: str
    category: str
    source_file: str
    content: str


def extract_text_from_bytes(filename: str, raw_bytes: bytes) -> str:
    """Extract and normalize text from PDF, DOCX, TXT, or MD files."""
    ext = Path(filename).suffix.lower()
    if ext in {".md", ".txt"}:
        return raw_bytes.decode("utf-8", errors="ignore")

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(raw_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return "\n\n".join(p.strip() for p in pages if p.strip())

    if ext == ".docx":
        from docx import Document

        doc = Document(io.BytesIO(raw_bytes))
        parts = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_txt = " | ".join(cell.text.strip() for cell in row.cells if cell.text.strip())
                if row_txt:
                    parts.append(row_txt)
        return "\n\n".join(parts)

    raise ValueError(f"Unsupported file format '{ext}'. Supported: .pdf, .docx, .txt, .md")


def clean_text(text: str) -> str:
    """Clean extracted text while preserving paragraph boundaries."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class HybridRAGRetriever:
    """Dual Sparse TF-IDF + Dense Semantic Vector Retriever with RRF & Reranking."""

    def __init__(self, knowledge_dir: str | Path | None = None) -> None:
        self.knowledge_dir = Path(knowledge_dir or settings.knowledge_dir)
        self.chunks: list[KnowledgeChunk] = []
        self.sparse_vectorizer: TfidfVectorizer | None = None
        self.sparse_matrix = None
        self.dense_char_vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.dense_matrix: Optional[np.ndarray] = None
        self.gemini_embeddings: Optional[np.ndarray] = None
        self.load_directory(self.knowledge_dir)

    def load_directory(self, directory: Path) -> None:
        """Load initial curated knowledge documents from disk."""
        self.chunks.clear()
        if directory.exists():
            for path in sorted(directory.glob("*")):
                if path.suffix.lower() in {".md", ".txt", ".pdf", ".docx"} and path.is_file():
                    raw = extract_text_from_bytes(path.name, path.read_bytes())
                    category = self._infer_category(path.stem)
                    self._ingest_text(
                        parent_id=path.stem,
                        title_default=path.stem.replace("_", " ").title(),
                        category=category,
                        source_file=path.name,
                        raw_text=raw,
                    )
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

    def _chunk_long_body(self, text: str, max_chars: int = 750, overlap: int = 120) -> list[str]:
        """Split long sections into overlapping semantic chunks."""
        cleaned = clean_text(text)
        if len(cleaned) <= max_chars:
            return [cleaned] if cleaned else []
        chunks: list[str] = []
        start = 0
        while start < len(cleaned):
            end = min(len(cleaned), start + max_chars)
            if end < len(cleaned):
                split_at = cleaned.rfind(" ", start + max_chars // 2, end)
                if split_at != -1:
                    end = split_at
            piece = cleaned[start:end].strip()
            if piece:
                chunks.append(piece)
            if end >= len(cleaned):
                break
            start = max(start + 1, end - overlap)
        return chunks

    def _ingest_text(
        self,
        parent_id: str,
        title_default: str,
        category: str,
        source_file: str,
        raw_text: str,
    ) -> list[KnowledgeChunk]:
        cleaned = clean_text(raw_text)
        lines = cleaned.splitlines()
        doc_title = title_default
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

        if not sections and cleaned:
            sections.append((doc_title, cleaned))

        created: list[KnowledgeChunk] = []
        idx = 1
        for heading, body in sections:
            for sub in self._chunk_long_body(body):
                chunk = KnowledgeChunk(
                    doc_id=f"{parent_id}#{idx}",
                    parent_id=parent_id,
                    title=heading,
                    category=category,
                    source_file=source_file,
                    content=sub,
                )
                self.chunks.append(chunk)
                created.append(chunk)
                idx += 1
        return created

    def add_document(
        self,
        title: str,
        category: str,
        content: str,
        parent_id: Optional[str] = None,
        source_file: str = "custom_input.md",
    ) -> list[KnowledgeChunk]:
        """Add a document at runtime and rebuild both sparse and dense indices."""
        pid = parent_id or f"doc-{uuid.uuid4().hex[:8]}"
        markdown_text = f"# {title}\n\n{content}" if not content.lstrip().startswith("#") else content
        created = self._ingest_text(
            parent_id=pid,
            title_default=title,
            category=category or "Custom",
            source_file=source_file,
            raw_text=markdown_text,
        )
        self._rebuild_index()
        return created

    def ingest_file(
        self, filename: str, raw_bytes: bytes, category: str = "Uploaded Document", parent_id: Optional[str] = None
    ) -> tuple[str, list[KnowledgeChunk]]:
        """Extract text from PDF/DOCX/TXT/MD, chunk, and index into vector store."""
        text = extract_text_from_bytes(filename, raw_bytes)
        if not text.strip():
            raise ValueError("Uploaded file contained no extractable text.")
        title = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        pid = parent_id or f"doc-{uuid.uuid4().hex[:8]}"
        chunks = self._ingest_text(
            parent_id=pid,
            title_default=title,
            category=category,
            source_file=filename,
            raw_text=text,
        )
        self._rebuild_index()
        return text, chunks

    def delete_document(self, doc_or_parent_id: str) -> int:
        """Remove all chunks belonging to a document ID and rebuild indices."""
        before = len(self.chunks)
        self.chunks = [
            c
            for c in self.chunks
            if c.parent_id != doc_or_parent_id and c.doc_id != doc_or_parent_id
        ]
        removed = before - len(self.chunks)
        if removed > 0:
            self._rebuild_index()
        return removed

    def _rebuild_index(self) -> None:
        if not self.chunks:
            self.sparse_vectorizer = None
            self.sparse_matrix = None
            self.dense_char_vectorizer = None
            self.svd = None
            self.dense_matrix = None
            self.gemini_embeddings = None
            return

        corpus = [f"{c.title}\n{c.category}\n{c.content}" for c in self.chunks]

        # 1. Sparse Lexical Index (TF-IDF word n-grams)
        self.sparse_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(corpus)

        # 2. Local Dense Semantic Vector Space (Char+Word Subword LSA via TruncatedSVD)
        self.dense_char_vectorizer = TfidfVectorizer(
            lowercase=True,
            analyzer="char_wb",
            ngram_range=(3, 5),
            sublinear_tf=True,
        )
        char_mat = self.dense_char_vectorizer.fit_transform(corpus)
        n_comp = min(64, max(2, min(char_mat.shape) - 1))
        if n_comp >= 2:
            self.svd = TruncatedSVD(n_components=n_comp, random_state=42)
            dense = self.svd.fit_transform(char_mat)
            self.dense_matrix = normalize(dense)
        else:
            self.svd = None
            self.dense_matrix = normalize(char_mat.toarray())

        # Optional Gemini dense embeddings when key is configured
        self.gemini_embeddings = None
        if get_runtime_gemini_key() and len(corpus) <= 60:
            try:
                from google import genai

                client = genai.Client(api_key=get_runtime_gemini_key())
                vecs = []
                for doc_txt in corpus:
                    res = client.models.embed_content(
                        model=settings.gemini_embedding_model,
                        contents=doc_txt,
                    )
                    vecs.append(res.embeddings[0].values)
                self.gemini_embeddings = normalize(np.array(vecs, dtype=np.float32))
            except Exception as exc:
                logger.debug("Gemini embedding skipped, using local dense SVD: %s", exc)
                self.gemini_embeddings = None

    def _expand_query(self, query: str) -> tuple[str, list[str]]:
        tokens = re.findall(r"[\w\u0900-\u097F-]+", query.lower())
        expanded_terms: list[str] = list(tokens)
        for tok in tokens:
            if tok in MULTILINGUAL_TERM_MAP:
                expanded_terms.extend(MULTILINGUAL_TERM_MAP[tok])

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

    def _dense_scores(self, expanded_query: str, raw_query: str) -> np.ndarray:
        if self.gemini_embeddings is not None and get_runtime_gemini_key():
            try:
                from google import genai

                client = genai.Client(api_key=get_runtime_gemini_key())
                res = client.models.embed_content(
                    model=settings.gemini_embedding_model,
                    contents=raw_query,
                )
                q_emb = normalize(np.array([res.embeddings[0].values], dtype=np.float32))
                return cosine_similarity(q_emb, self.gemini_embeddings).flatten()
            except Exception:
                pass

        if self.dense_char_vectorizer is None or self.dense_matrix is None:
            return np.zeros(len(self.chunks))
        q_char = self.dense_char_vectorizer.transform([expanded_query])
        if self.svd is not None:
            q_dense = normalize(self.svd.transform(q_char))
        else:
            q_dense = normalize(q_char.toarray())
        return np.clip(cosine_similarity(q_dense, self.dense_matrix).flatten(), 0.0, 1.0)

    def retrieve_sparse_only(self, query: str, top_k: int = 3, min_score: float = 0.08) -> list[RetrievedChunk]:
        """Pure sparse TF-IDF baseline (used for RAG evaluation comparison)."""
        if not self.chunks or self.sparse_vectorizer is None or self.sparse_matrix is None:
            return []
        expanded, _ = self._expand_query(query)
        q_vec = self.sparse_vectorizer.transform([expanded])
        sims = cosine_similarity(q_vec, self.sparse_matrix).flatten()
        order = np.argsort(-sims)[:top_k]
        return [
            RetrievedChunk(
                doc_id=self.chunks[i].doc_id,
                title=self.chunks[i].title,
                category=self.chunks[i].category,
                content=self.chunks[i].content,
                score=round(float(sims[i]), 4),
                sparse_score=round(float(sims[i]), 4),
                dense_score=0.0,
            )
            for i in order
            if sims[i] >= min_score
        ]

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.12) -> list[RetrievedChunk]:
        """Hybrid Dense + Sparse TF-IDF retrieval with Reciprocal Rank Fusion & reranking."""
        if not self.chunks or self.sparse_vectorizer is None or self.sparse_matrix is None or not query.strip():
            return []

        expanded_query, query_tokens = self._expand_query(query)
        q_vec = self.sparse_vectorizer.transform([expanded_query])
        sparse_sims = cosine_similarity(q_vec, self.sparse_matrix).flatten()
        dense_sims = self._dense_scores(query, query)

        from app.nlp import detect_language

        is_english = detect_language(query) == "en"
        # Out-of-scope domain noun coverage check (for English queries with >=2 domain words)
        generic_words = {
            "what", "when", "where", "which", "your", "does", "have", "about",
            "with", "from", "this", "that", "they", "their", "there", "could",
            "would", "should", "price", "pricing", "cost", "many", "much", "sell",
            "support", "provide", "offer", "fortivex", "compare", "against",
        }
        raw_domain_words = (
            [
                w for w in re.findall(r"[a-z]{4,}", query.lower())
                if w not in generic_words and w not in MULTILINGUAL_TERM_MAP
            ]
            if is_english
            else []
        )
        all_kb_text = " ".join(f"{c.title} {c.content}".lower() for c in self.chunks)
        domain_hits = sum(1 for w in raw_domain_words if w in all_kb_text or w.rstrip("s") in all_kb_text)
        domain_coverage = (domain_hits / len(raw_domain_words)) if raw_domain_words else 1.0

        # Compute RRF ranks (k=20)
        sparse_ranks = np.argsort(np.argsort(-sparse_sims))
        dense_ranks = np.argsort(np.argsort(-dense_sims))

        query_token_set = {t for t in query_tokens if len(t) >= 3}
        scored: list[tuple[float, float, float, KnowledgeChunk, list[str]]] = []

        for idx, chunk in enumerate(self.chunks):
            s_score = float(sparse_sims[idx])
            d_score = float(dense_sims[idx])
            chunk_text_lower = f"{chunk.title} {chunk.content}".lower()
            matched = [t for t in query_token_set if t in chunk_text_lower]
            lexical_bonus = min(0.20, 0.05 * len(matched))

            # Hybrid fusion: weighted similarity + signal-scaled RRF boost
            rrf = (1.0 / (20 + sparse_ranks[idx])) + (1.0 / (20 + dense_ranks[idx]))
            rrf_norm = min(0.16, rrf * 1.6) * min(1.0, (s_score + d_score) * 1.4)
            hybrid_score = min(1.0, (0.55 * s_score) + (0.28 * d_score) + lexical_bonus + rrf_norm)

            if is_english and len(raw_domain_words) >= 2 and domain_coverage < 0.45:
                hybrid_score *= 0.35

            if (s_score > 0.03 or len(matched) > 0 or d_score > 0.45) and hybrid_score >= min_score:
                scored.append((hybrid_score, s_score, d_score, chunk, sorted(matched)[:6]))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                category=chunk.category,
                content=chunk.content,
                score=round(score, 4),
                sparse_score=round(s_score, 4),
                dense_score=round(d_score, 4),
                matched_terms=matched,
            )
            for score, s_score, d_score, chunk, matched in scored[:top_k]
        ]

    def is_knowledge_query(self, text: str) -> bool:
        """Heuristic identifying whether user is asking a factual product/policy question."""
        lower = text.lower()
        question_markers = [
            "?", "what", "how", "when", "where", "which", "does", "do you", "can you",
            "price", "pricing", "cost", "soc2", "gdpr", "hipaa", "security", "integrat",
            "gong", "chorus", "drift", "sla", "uptime", "policy", "refund", "warranty",
            "qué", "cómo", "cuál", "precio", "quoi", "comment", "prix", "wie", "was", "preis",
            "क्या", "कैसे", "कीमत", "kya", "kitna", "keemat",
        ]
        return any(m in lower for m in question_markers)

    def list_chunks(self) -> list[dict[str, str]]:
        """List all indexed chunks with source metadata."""
        return [
            {
                "doc_id": c.doc_id,
                "parent_id": c.parent_id,
                "title": c.title,
                "category": c.category,
                "source_file": c.source_file,
                "preview": c.content[:180] + ("..." if len(c.content) > 180 else ""),
            }
            for c in self.chunks
        ]
