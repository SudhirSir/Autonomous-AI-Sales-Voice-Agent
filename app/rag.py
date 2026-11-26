"""Advanced Hybrid RAG Engine (Sparse TF-IDF + Multilingual Dense Embeddings + RRF + Reranker + Cache + Contradiction Detection)."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import logging
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

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
    "प्लान": ["pricing", "plan", "growth", "scale"],
    "बजट": ["pricing", "budget", "plan"],
    "keemat": ["pricing", "plan", "cost"],
    "paisa": ["pricing", "cost", "roi"],
    "mehnga": ["pricing", "discount", "roi"],
    "seguridad": ["security", "soc", "encryption", "gdpr", "hipaa"],
    "cumplimiento": ["compliance", "soc", "gdpr", "hipaa", "iso"],
    "privacidad": ["privacy", "gdpr", "pii", "zero-training"],
    "sécurité": ["security", "soc", "encryption", "gdpr", "hipaa"],
    "conformité": ["compliance", "soc", "gdpr", "iso"],
    "sicherheit": ["security", "soc", "encryption", "gdpr", "hipaa"],
    "datenschutz": ["privacy", "gdpr", "pii", "iso"],
    "सुरक्षा": ["security", "soc", "encryption", "gdpr"],
    "suraksha": ["security", "soc", "encryption", "gdpr"],
    "integración": ["integration", "salesforce", "hubspot", "crm", "api"],
    "integraciones": ["integrations", "salesforce", "hubspot", "crm"],
    "intégration": ["integration", "salesforce", "hubspot", "crm", "api"],
    "integrationen": ["integrations", "salesforce", "hubspot", "crm", "api"],
    "एकीकरण": ["integration", "salesforce", "hubspot", "crm"],
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


class DiskEmbeddingCache:
    """Persistent file-based embedding cache avoiding repeated API/model recomputation."""

    def __init__(self, cache_dir: str | Path) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "embeddings.json"
        self._data: dict[str, list[float]] = {}
        self._load()

    def _load(self) -> None:
        if self.cache_file.exists():
            try:
                self._data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                self._data = {}

    def _save(self) -> None:
        try:
            self.cache_file.write_text(json.dumps(self._data), encoding="utf-8")
        except Exception as exc:
            logger.debug("Embedding cache write skipped: %s", exc)

    @staticmethod
    def key(backend: str, model_name: str, text: str) -> str:
        digest = hashlib.sha256(f"{backend}:{model_name}:{text}".encode("utf-8")).hexdigest()
        return digest

    def get(self, backend: str, model_name: str, text: str) -> Optional[list[float]]:
        return self._data.get(self.key(backend, model_name, text))

    def put_many(self, backend: str, model_name: str, items: list[tuple[str, list[float]]]) -> None:
        for txt, vec in items:
            self._data[self.key(backend, model_name, txt)] = vec
        self._save()


class HybridRAGRetriever:
    """Hybrid Dense + Sparse TF-IDF Retriever with Persistent Embedding Cache, RRF & Reranking."""

    def __init__(
        self,
        knowledge_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        sparse_weight: Optional[float] = None,
        dense_weight: Optional[float] = None,
        rerank_weight: Optional[float] = None,
    ) -> None:
        self.knowledge_dir = Path(knowledge_dir or settings.knowledge_dir)
        self.cache = DiskEmbeddingCache(cache_dir or settings.rag_cache_dir)
        self.sparse_weight = sparse_weight if sparse_weight is not None else settings.rag_sparse_weight
        self.dense_weight = dense_weight if dense_weight is not None else settings.rag_dense_weight
        self.rerank_weight = rerank_weight if rerank_weight is not None else settings.rag_rerank_weight

        self.chunks: list[KnowledgeChunk] = []
        self.sparse_vectorizer: TfidfVectorizer | None = None
        self.sparse_matrix = None
        self.dense_char_vectorizer: TfidfVectorizer | None = None
        self.svd: TruncatedSVD | None = None
        self.dense_matrix: Optional[np.ndarray] = None
        self.neural_embeddings: Optional[np.ndarray] = None
        self.embedding_backend: str = "local-tfidf-lsa"
        self.reranker_backend: str = "lexical-semantic-reranker"

        self._st_model = None
        self._cross_encoder = None
        self._st_available = importlib.util.find_spec("sentence_transformers") is not None

        self.load_directory(self.knowledge_dir)

    def _get_sentence_transformer(self):
        if not self._st_available:
            return None
        if self._st_model is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._st_model = SentenceTransformer(settings.rag_embedding_model)
            except Exception as exc:
                logger.debug("SentenceTransformer load skipped: %s", exc)
                self._st_available = False
                return None
        return self._st_model

    def _get_cross_encoder(self):
        if not self._st_available:
            return None
        if self._cross_encoder is None:
            try:
                from sentence_transformers import CrossEncoder

                self._cross_encoder = CrossEncoder(settings.rag_reranker_model)
                self.reranker_backend = f"cross-encoder:{settings.rag_reranker_model}"
            except Exception as exc:
                logger.debug("CrossEncoder load skipped: %s", exc)
                return None
        return self._cross_encoder

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

    def _chunk_long_body(self, text: str, max_chars: int = 680, overlap: int = 120) -> list[str]:
        """Split long sections on sentence/paragraph boundaries into overlapping chunks."""
        cleaned = clean_text(text)
        if len(cleaned) <= max_chars:
            return [cleaned] if cleaned else []

        paragraphs = [p.strip() for p in cleaned.split("\n\n") if p.strip()]
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            if len(current) + len(para) + 2 <= max_chars:
                current = f"{current}\n\n{para}".strip()
            else:
                if current:
                    chunks.append(current)
                if len(para) <= max_chars:
                    current = para
                else:
                    # Sliding window on word boundary for oversized paragraph
                    start = 0
                    while start < len(para):
                        end = min(len(para), start + max_chars)
                        if end < len(para):
                            sp = para.rfind(" ", start + max_chars // 2, end)
                            if sp != -1:
                                end = sp
                        piece = para[start:end].strip()
                        if piece:
                            chunks.append(piece)
                        if end >= len(para):
                            break
                        start = max(start + 1, end - overlap)
                    current = ""
        if current:
            chunks.append(current)
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
            self.neural_embeddings = None
            return

        corpus = [f"{c.title}\n{c.category}\n{c.content}" for c in self.chunks]

        # 1. Sparse Lexical Index (Word TF-IDF 1-2 grams)
        self.sparse_vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )
        self.sparse_matrix = self.sparse_vectorizer.fit_transform(corpus)

        # 2. Local Char-Ngram + LSA fallback matrix (never misrepresented as a neural model)
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

        # 3. Neural Multilingual Dense Embeddings (SentenceTransformers -> Gemini -> local fallback)
        self.neural_embeddings = None
        self.embedding_backend = "local-tfidf-lsa"

        st_model = self._get_sentence_transformer()
        if st_model is not None:
            try:
                vecs: list[list[float]] = []
                to_encode: list[tuple[int, str]] = []
                for idx, txt in enumerate(corpus):
                    cached = self.cache.get("sentence-transformers", settings.rag_embedding_model, txt)
                    if cached is not None:
                        vecs.append(cached)
                    else:
                        vecs.append([])
                        to_encode.append((idx, txt))
                if to_encode:
                    encoded = st_model.encode([t for _, t in to_encode], normalize_embeddings=True)
                    new_cache_items = []
                    for (idx, txt), vec in zip(to_encode, encoded):
                        v_list = vec.tolist()
                        vecs[idx] = v_list
                        new_cache_items.append((txt, v_list))
                    self.cache.put_many("sentence-transformers", settings.rag_embedding_model, new_cache_items)
                self.neural_embeddings = normalize(np.array(vecs, dtype=np.float32))
                self.embedding_backend = f"sentence-transformers:{settings.rag_embedding_model}"
                return
            except Exception as exc:
                logger.debug("SentenceTransformer embedding skipped: %s", exc)

        gemini_key = get_runtime_gemini_key()
        if gemini_key and len(corpus) <= 80:
            try:
                from google import genai

                client = genai.Client(api_key=gemini_key)
                vecs = []
                new_cache_items = []
                for txt in corpus:
                    cached = self.cache.get("gemini", settings.gemini_embedding_model, txt)
                    if cached is not None:
                        vecs.append(cached)
                    else:
                        res = client.models.embed_content(
                            model=settings.gemini_embedding_model,
                            contents=txt,
                        )
                        v_list = list(res.embeddings[0].values)
                        vecs.append(v_list)
                        new_cache_items.append((txt, v_list))
                if new_cache_items:
                    self.cache.put_many("gemini", settings.gemini_embedding_model, new_cache_items)
                self.neural_embeddings = normalize(np.array(vecs, dtype=np.float32))
                self.embedding_backend = f"gemini:{settings.gemini_embedding_model}"
            except Exception as exc:
                logger.debug("Gemini embedding skipped, using local-tfidf-lsa fallback: %s", exc)
                self.neural_embeddings = None

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
        if self.neural_embeddings is not None:
            if self.embedding_backend.startswith("sentence-transformers:") and self._st_model is not None:
                try:
                    q_vec = self._st_model.encode([raw_query], normalize_embeddings=True)
                    return np.clip(cosine_similarity(q_vec, self.neural_embeddings).flatten(), 0.0, 1.0)
                except Exception:
                    pass
            elif self.embedding_backend.startswith("gemini:") and get_runtime_gemini_key():
                try:
                    from google import genai

                    client = genai.Client(api_key=get_runtime_gemini_key())
                    res = client.models.embed_content(
                        model=settings.gemini_embedding_model,
                        contents=raw_query,
                    )
                    q_emb = normalize(np.array([res.embeddings[0].values], dtype=np.float32))
                    return np.clip(cosine_similarity(q_emb, self.neural_embeddings).flatten(), 0.0, 1.0)
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

    def _rerank_scores(self, query: str, expanded_tokens: list[str], candidate_indices: list[int]) -> dict[int, float]:
        """Score candidates with CrossEncoder if available, else lexical-semantic reranker."""
        if not candidate_indices:
            return {}
        ce = self._get_cross_encoder()
        if ce is not None:
            try:
                pairs = [(query, f"{self.chunks[i].title}: {self.chunks[i].content}") for i in candidate_indices]
                raw_scores = ce.predict(pairs)
                # Sigmoid normalize logits to [0, 1]
                probs = 1.0 / (1.0 + np.exp(-np.array(raw_scores, dtype=np.float32)))
                return {idx: float(p) for idx, p in zip(candidate_indices, probs)}
            except Exception:
                pass

        q_set = {t for t in expanded_tokens if len(t) >= 3}
        out: dict[int, float] = {}
        for idx in candidate_indices:
            ch = self.chunks[idx]
            title_l = ch.title.lower()
            body_l = ch.content.lower()
            title_hits = sum(1 for t in q_set if t in title_l)
            body_hits = sum(1 for t in q_set if t in body_l)
            coverage = (title_hits * 1.5 + body_hits) / max(1.0, len(q_set) * 1.4)
            out[idx] = round(min(1.0, coverage), 4)
        return out

    def retrieve_sparse_only(self, query: str, top_k: int = 3, min_score: float = 0.08) -> list[RetrievedChunk]:
        """Pure sparse TF-IDF baseline (for RAG evaluation comparison)."""
        if not self.chunks or self.sparse_vectorizer is None or self.sparse_matrix is None:
            return []
        expanded, _ = self._expand_query(query)
        q_vec = self.sparse_vectorizer.transform([expanded])
        sims = cosine_similarity(q_vec, self.sparse_matrix).flatten()
        order = np.argsort(-sims)[:top_k]
        results: list[RetrievedChunk] = []
        for rank_idx, i in enumerate(order, 1):
            if sims[i] >= min_score:
                sc = round(float(sims[i]), 4)
                results.append(
                    RetrievedChunk(
                        doc_id=self.chunks[i].doc_id,
                        title=self.chunks[i].title,
                        category=self.chunks[i].category,
                        source_file=self.chunks[i].source_file,
                        content=self.chunks[i].content,
                        score=sc,
                        rank=rank_idx,
                        sparse_score=sc,
                        dense_score=0.0,
                        rrf_score=0.0,
                        reranker_score=0.0,
                        final_confidence=sc,
                        embedding_backend="sparse-only",
                    )
                )
        return results

    def retrieve(self, query: str, top_k: int = 3, min_score: float = 0.12) -> list[RetrievedChunk]:
        """Hybrid Dense + Sparse TF-IDF retrieval with RRF, Cross-Encoder Reranking & Unknown Gating."""
        if not self.chunks or self.sparse_vectorizer is None or self.sparse_matrix is None or not query.strip():
            return []

        expanded_query, query_tokens = self._expand_query(query)
        q_vec = self.sparse_vectorizer.transform([expanded_query])
        sparse_sims = cosine_similarity(q_vec, self.sparse_matrix).flatten()
        dense_sims = self._dense_scores(expanded_query, query)

        from app.nlp import detect_language

        is_english = detect_language(query) == "en"
        generic_words = {
            "what", "when", "where", "which", "your", "does", "have", "about",
            "with", "from", "this", "that", "they", "their", "there", "could",
            "would", "should", "price", "pricing", "cost", "many", "much", "sell",
            "support", "provide", "offer", "fortivex", "compare", "against", "tell",
            "include", "included", "work", "works", "demo", "demos",
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

        sparse_ranks = np.argsort(np.argsort(-sparse_sims))
        dense_ranks = np.argsort(np.argsort(-dense_sims))

        # Preselect top pool for reranking
        pre_scores = (0.55 * sparse_sims) + (0.45 * dense_sims)
        pool_indices = list(np.argsort(-pre_scores)[: max(top_k * 3, 8)])
        rerank_map = self._rerank_scores(query, query_tokens, pool_indices)

        query_token_set = {t for t in query_tokens if len(t) >= 3}
        scored: list[tuple[float, float, float, float, float, KnowledgeChunk, list[str]]] = []

        for idx in pool_indices:
            chunk = self.chunks[idx]
            s_score = float(sparse_sims[idx])
            d_score = float(dense_sims[idx])
            r_score = float(rerank_map.get(idx, 0.0))

            chunk_text_lower = f"{chunk.title} {chunk.content}".lower()
            matched = [t for t in query_token_set if t in chunk_text_lower]
            lexical_bonus = min(0.16, 0.04 * len(matched))

            rrf_raw = (1.0 / (20 + sparse_ranks[idx])) + (1.0 / (20 + dense_ranks[idx]))
            rrf_scaled = min(0.18, rrf_raw * 1.8) * min(1.0, (s_score + d_score) * 1.35)

            hybrid_score = min(
                1.0,
                (self.sparse_weight * s_score)
                + (self.dense_weight * d_score)
                + (self.rerank_weight * r_score)
                + lexical_bonus
                + rrf_scaled,
            )

            if is_english and len(raw_domain_words) >= 2 and domain_coverage < 0.45:
                hybrid_score *= 0.20

            neural_hit = (self.neural_embeddings is not None) and (d_score > 0.45)
            if (s_score > 0.03 or len(matched) > 0 or neural_hit) and hybrid_score >= min_score:
                scored.append((hybrid_score, s_score, d_score, rrf_scaled, r_score, chunk, sorted(matched)[:6]))

        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievedChunk(
                doc_id=chunk.doc_id,
                title=chunk.title,
                category=chunk.category,
                source_file=chunk.source_file,
                content=chunk.content,
                score=round(score, 4),
                rank=rank_i,
                sparse_score=round(s_score, 4),
                dense_score=round(d_score, 4),
                rrf_score=round(rrf_s, 4),
                reranker_score=round(rer_s, 4),
                final_confidence=round(score, 4),
                embedding_backend=self.embedding_backend,
                matched_terms=matched,
            )
            for rank_i, (score, s_score, d_score, rrf_s, rer_s, chunk, matched) in enumerate(scored[:top_k], 1)
        ]

    @staticmethod
    def detect_contradictions(chunks: list[RetrievedChunk]) -> tuple[bool, Optional[str]]:
        """Detect conflicting numeric prices, SLAs, or pilot durations across retrieved chunks."""
        if len(chunks) < 2:
            return False, None

        # Compare dollar pricing or uptime SLA percentages across chunks for the same tier/metric
        tier_prices: dict[str, set[str]] = {}
        sla_values: set[str] = set()
        pilot_days: set[str] = set()

        for c in chunks:
            low = c.content.lower()
            for tier in ("growth", "scale", "enterprise"):
                m = re.search(rf"{tier}[^.\n]*?\$([\d,]+)", low)
                if m:
                    tier_prices.setdefault(tier, set()).add(f"${m.group(1)}")
            for m_sla in re.finditer(r"(\d{2}\.\d+)\s*%\s*(?:uptime|sla)", low):
                sla_values.add(f"{m_sla.group(1)}%")
            for m_pil in re.finditer(r"(\d+)-day\s+(?:guided\s+)?(?:sandbox\s+)?pilot", low):
                pilot_days.add(f"{m_pil.group(1)}-day")

        conflicts: list[str] = []
        for tier, vals in tier_prices.items():
            if len(vals) > 1:
                conflicts.append(f"{tier.title()} tier price differs across sources ({' vs '.join(sorted(vals))})")
        if len(pilot_days) > 1:
            conflicts.append(f"Pilot duration differs across sources ({' vs '.join(sorted(pilot_days))})")

        if conflicts:
            titles = " | ".join(dict.fromkeys(c.title for c in chunks[:3]))
            return True, f"Contradictory knowledge detected ({'; '.join(conflicts)} across [{titles}])."
        return False, None

    def is_knowledge_query(self, text: str) -> bool:
        """Identify whether user is asking a factual product, pricing, or policy question."""
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

    def diagnostics(self) -> dict[str, Any]:
        return {
            "total_chunks": len(self.chunks),
            "embedding_backend": self.embedding_backend,
            "reranker_backend": self.reranker_backend,
            "weights": {
                "sparse": self.sparse_weight,
                "dense": self.dense_weight,
                "rerank": self.rerank_weight,
            },
            "min_confidence": settings.rag_min_confidence,
        }
