"""
Hybrid retrieval and answer generation for multi-tenant Maintenance Copilot.
"""

from __future__ import annotations

import logging
import math
import os
import pickle
import re
import uuid
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import HashingVectorizer

from core.tenant import TenantContext, migrate_legacy_tenant_storage, resolve_tenant
from database.database import database_enabled
from repositories.document_repository import load_chunk_metadata
from repositories.vector_repository import VectorRepository

logger = logging.getLogger(__name__)

DEBUG = os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if os.getenv("ALLOW_MODEL_DOWNLOADS", "false").lower() not in {"1", "true", "yes"}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

FAISS_CANDIDATES = int(os.getenv("FAISS_CANDIDATES", "30"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "8"))
HYBRID_CANDIDATES = int(os.getenv("HYBRID_CANDIDATES", str(max(FAISS_CANDIDATES, 30))))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GEMINI_FALLBACK_MODELS = [
    model.strip()
    for model in os.getenv("GEMINI_FALLBACK_MODELS", "gemini-2.0-flash").split(",")
    if model.strip()
]
GENERATION_MODELS = list(dict.fromkeys([GEMINI_MODEL, *GEMINI_FALLBACK_MODELS]))
MAX_GENERATION_ATTEMPTS = int(os.getenv("GEMINI_MAX_ATTEMPTS", "3"))
BI_ENCODER_MODEL = os.getenv("BI_ENCODER_MODEL", "all-MiniLM-L6-v2")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
MANUAL_ANSWER_MIN_CONFIDENCE = int(os.getenv("MANUAL_ANSWER_MIN_CONFIDENCE", "55"))

_ABBREV_SAFE_SPLIT = re.compile(
    r"(?<!Rev)(?<!Fig)(?<!approx)(?<!SOP)(?<!min)(?<!max)(?<=[.!?])\s+",
    re.IGNORECASE,
)

INDUSTRIAL_GLOSSARY = {
    "bearing": "bearing lubrication spindle shaft radial axial grease preload noise heat",
    "bearings": "bearing lubrication spindle shaft radial axial grease preload noise heat",
    "calibrate": "calibration adjust verify tolerance zero span reference standard",
    "calibration": "calibrate adjust verify tolerance zero span reference standard",
    "hot": "overheating heat thermal temperature spike warm casing",
    "overheating": "hot heat thermal temperature spike cooling airflow overload",
    "leak": "leaking fluid lubrication seep drip oil hydraulic seal gasket",
    "leaking": "leak fluid lubrication seep drip oil hydraulic seal gasket",
    "noise": "vibration rattle knock grinding squeal abnormal sound bearing",
    "vibration": "vibrating alignment unbalance loose coupling bearing resonance",
    "vibrating": "vibration alignment unbalance loose coupling bearing resonance",
    "pressure": "psi bar hydraulic pneumatic gauge flow valve regulator",
    "stopped": "stall jam trip fault overload interlock power failure",
    "jam": "stuck blocked seized obstruction feed stall trip",
    "smoke": "burning overheating insulation electrical fault thermal fire",
    "lubricate": "lubrication grease oil interval bearing spindle service",
    "lubrication": "lubricate grease oil interval bearing spindle service",
}


class HashingEmbeddingModel:
    def __init__(self, n_features: int = 384) -> None:
        self.vectorizer = HashingVectorizer(
            n_features=n_features,
            alternate_sign=False,
            norm="l2",
            ngram_range=(1, 2),
            lowercase=True,
        )

    def encode(self, texts: str | list[str]) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        return self.vectorizer.transform(texts).toarray().astype("float32")


class KeywordReranker:
    _token_pattern = re.compile(r"\b[a-zA-Z][a-zA-Z0-9_-]{2,}\b")

    def _tokens(self, text: str) -> set[str]:
        return set(self._token_pattern.findall(text.lower()))

    def predict(self, pairs: list[list[str]]) -> list[float]:
        scores = []
        for question, candidate in pairs:
            question_tokens = self._tokens(question)
            candidate_tokens = self._tokens(candidate)
            if not question_tokens or not candidate_tokens:
                scores.append(-1.0)
                continue
            overlap = len(question_tokens & candidate_tokens)
            coverage = overlap / max(len(question_tokens), 1)
            density = overlap / max(len(candidate_tokens), 1)
            phrase_bonus = 0.2 if question.lower() in candidate.lower() else 0.0
            scores.append(float((coverage * 0.8) + (density * 0.2) + phrase_bonus))
        return scores


class BM25Index:
    token_pattern = re.compile(r"\b[\w./+-]+\b")

    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.chunks = chunks
        self.doc_tokens: list[list[str]] = []
        self.term_freqs: list[Counter[str]] = []
        self.doc_freqs: Counter[str] = Counter()
        self.doc_lengths: list[int] = []
        self.avgdl = 0.0
        self.k1 = 1.5
        self.b = 0.75
        self._build()

    def tokenize(self, text: str) -> list[str]:
        return [token.lower() for token in self.token_pattern.findall(text)]

    def _build(self) -> None:
        total_length = 0
        for chunk in self.chunks:
            metadata_text = " ".join(
                str(chunk.get(field, ""))
                for field in ["source_file", "section_title", "machine_type", "manufacturer", "revision"]
            )
            tokens = self.tokenize(f"{metadata_text}\n{chunk.get('text', '')}")
            term_freq = Counter(tokens)
            self.doc_tokens.append(tokens)
            self.term_freqs.append(term_freq)
            self.doc_lengths.append(len(tokens))
            total_length += len(tokens)
            self.doc_freqs.update(term_freq.keys())
        self.avgdl = total_length / max(len(self.chunks), 1)

    def score(self, query: str, index: int) -> float:
        query_terms = self.tokenize(query)
        if not query_terms or index >= len(self.term_freqs):
            return 0.0
        score = 0.0
        doc_len = self.doc_lengths[index] or 1
        term_freq = self.term_freqs[index]
        total_docs = max(len(self.chunks), 1)
        for term in query_terms:
            tf = term_freq.get(term, 0)
            if tf == 0:
                continue
            df = self.doc_freqs.get(term, 0)
            idf = math.log(1 + ((total_docs - df + 0.5) / (df + 0.5)))
            denom = tf + self.k1 * (1 - self.b + self.b * doc_len / max(self.avgdl, 1))
            score += idf * ((tf * (self.k1 + 1)) / denom)
        return float(score)

    def search(self, query: str, top_k: int) -> list[tuple[int, float]]:
        scored = [(index, self.score(query, index)) for index in range(len(self.chunks))]
        scored = [(index, score) for index, score in scored if score > 0]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]


class RetrievalService:
    """Tenant-scoped hybrid retrieval and grounded answer generation."""

    def __init__(self, tenant: TenantContext) -> None:
        self.tenant = tenant
        self.vectors = VectorRepository(tenant)
        self._client: genai.Client | None = None
        self._bi_encoder: Any | None = None
        self._reranker: Any | None = None
        self._embedding_backend_name = ""
        self._reranker_backend_name = ""
        self._index: Any | None = None
        self._metadata_chunks: list[dict[str, Any]] = []
        self._database_stamp: tuple[float, float] | None = None
        self._bm25_index: BM25Index | None = None
        self._document_profiles: dict[str, dict[str, Any]] = {}

    def reload(self, force: bool = False) -> dict[str, Any]:
        migrate_legacy_tenant_storage(self.tenant)
        stamp = self.vectors.stamp
        if stamp is None and not self.tenant.index_path.exists():
            self._reset_state()
            return self.summary()

        if not force and self._index is not None and self._database_stamp == stamp:
            return self.summary()

        self._index = self.vectors.read_index()
        metadata = load_chunk_metadata(self.tenant) if database_enabled() else None
        if metadata is None and self.tenant.mapping_path.exists():
            with self.tenant.mapping_path.open("rb") as handle:
                metadata = pickle.load(handle)

        self._metadata_chunks = metadata or []
        self._database_stamp = stamp
        self._build_retrieval_caches()
        return self.summary()

    def summary(self) -> dict[str, Any]:
        files = sorted({chunk.get("source_file", "Unknown") for chunk in self._metadata_chunks})
        extraction_counts: dict[str, int] = {}
        document_type_files: dict[str, set[str]] = {}
        indexed_backends: set[str] = set()
        ocr_page_keys: set[tuple[Any, Any]] = set()

        for chunk in self._metadata_chunks:
            method = str(chunk.get("extraction", "native"))
            extraction_counts[method] = extraction_counts.get(method, 0) + 1
            source_file = str(chunk.get("source_file", ""))
            suffix = Path(source_file).suffix.lower() or "unknown"
            document_type_files.setdefault(suffix, set()).add(source_file)
            if chunk.get("embedding_backend"):
                indexed_backends.add(str(chunk.get("embedding_backend")))
            if "ocr" in method:
                ocr_page_keys.add((chunk.get("source_file"), chunk.get("page")))

        return {
            "organization_id": str(self.tenant.organization_id),
            "available": self._index is not None and bool(self._metadata_chunks),
            "chunks": len(self._metadata_chunks),
            "vectors": self._index.ntotal if self._index is not None else 0,
            "files": files,
            "file_count": len(files),
            "document_count": len(self._document_profiles),
            "extraction_counts": extraction_counts,
            "document_types": {
                suffix: len(source_files) for suffix, source_files in document_type_files.items()
            },
            "ocr_pages": len(ocr_page_keys),
            "bm25_ready": self._bm25_index is not None,
            "database_sync": database_enabled(),
            "indexed_embedding_backends": sorted(indexed_backends),
            "runtime_embedding_backend": self._embedding_backend_name or "not-loaded",
            "runtime_reranker_backend": self._reranker_backend_name or "not-loaded",
            "api_key_configured": bool(GEMINI_API_KEY),
            "generation_models": GENERATION_MODELS,
        }

    def ask(
        self,
        user_question: str,
        *,
        conversation_history: list[dict[str, Any]] | None = None,
        user_memory: list[str] | str | None = None,
        force_mode: str | None = None,
    ) -> tuple[str, list[dict[str, Any]]]:
        clean_question, resolved_mode = self._extract_forced_mode(user_question.strip(), force_mode)
        if not clean_question:
            return "[ANSWER: GENERAL SAFETY GUIDANCE]\nPlease enter a maintenance question or fault description.", []

        loaded = self.ensure_loaded()
        if loaded:
            expanded_query = self.expand_query_locally(clean_question)
            results = self.search_and_rerank(clean_question, expanded_query)
        else:
            results = []

        context, evidence = self._build_context_and_evidence(results)
        manual_confidence = self._manual_confidence(results)
        retrieval_confidence_pct = self.compute_confidence(results)

        if not loaded:
            context = (
                "No manuals are indexed yet. Do not answer machine-specific questions. "
                "Ask the user to upload the relevant manual, SOP, service bulletin, or troubleshooting guide."
            )
            manual_confidence = "no_indexed_manuals"
            retrieval_confidence_pct = 0

        if self._should_use_fallback(manual_confidence, retrieval_confidence_pct, context):
            return self._build_fallback_answer(clean_question, resolved_mode), []

        prompt = self._build_prompt(
            clean_question,
            context,
            self._build_conversation_memory(conversation_history, user_memory),
            manual_confidence,
            retrieval_confidence_pct,
            resolved_mode,
        )

        try:
            answer = self._generate_with_fallback(prompt)
            return self._ensure_source_tag(answer, "[ANSWER: VERIFIED FROM MANUAL]"), evidence
        except Exception as exc:
            logger.exception("Answer generation failed for tenant %s", self.tenant.organization_id)
            return (
                "[ANSWER: GENERAL SAFETY GUIDANCE]\nConnection error while generating the answer. "
                f"Details: {exc}\n\nCheck GEMINI_API_KEY, model access, and network connectivity.",
                evidence,
            )

    def ensure_loaded(self) -> bool:
        self.reload()
        return self._index is not None and bool(self._metadata_chunks)

    def _reset_state(self) -> None:
        self._index = None
        self._metadata_chunks = []
        self._database_stamp = None
        self._bm25_index = None
        self._document_profiles = {}

    def _document_key(self, chunk: dict[str, Any]) -> str:
        return str(chunk.get("document_id") or chunk.get("source_file") or "unknown")

    def _build_retrieval_caches(self) -> None:
        self._bm25_index = BM25Index(self._metadata_chunks) if self._metadata_chunks else None
        profiles: dict[str, dict[str, Any]] = {}
        for index, chunk in enumerate(self._metadata_chunks):
            key = self._document_key(chunk)
            profile = profiles.setdefault(
                key,
                {
                    "document_id": key,
                    "source_file": chunk.get("source_file", "Unknown"),
                    "machine_type": chunk.get("machine_type", "Unknown"),
                    "manufacturer": chunk.get("manufacturer", "Unknown"),
                    "revision": chunk.get("revision", "unknown"),
                    "language": chunk.get("language", "unknown"),
                    "chunk_indices": [],
                },
            )
            profile["chunk_indices"].append(index)
        self._document_profiles = profiles

    def get_client(self) -> genai.Client:
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env before asking questions.")
        if self._client is None:
            self._client = genai.Client(
                api_key=GEMINI_API_KEY,
                http_options=types.HttpOptions(
                    retry_options=types.HttpRetryOptions(
                        attempts=3,
                        initial_delay=2.0,
                        http_status_codes=[429, 503],
                    )
                ),
            )
        return self._client

    def get_bi_encoder(self) -> Any:
        if self._bi_encoder is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("Loading retrieval encoder: %s", BI_ENCODER_MODEL)
                self._bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
                self._embedding_backend_name = f"sentence-transformers:{BI_ENCODER_MODEL}"
            except Exception as exc:
                logger.warning("Falling back to hashing embeddings: %s", exc)
                self._bi_encoder = HashingEmbeddingModel(
                    n_features=int(os.getenv("HASH_EMBEDDING_FEATURES", "384"))
                )
                self._embedding_backend_name = "hashing-fallback"
        return self._bi_encoder

    def get_reranker(self) -> Any:
        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                logger.info("Loading reranker: %s", RERANKER_MODEL)
                self._reranker = CrossEncoder(RERANKER_MODEL)
                self._reranker_backend_name = f"cross-encoder:{RERANKER_MODEL}"
            except Exception as exc:
                logger.warning("Falling back to keyword reranker: %s", exc)
                self._reranker = KeywordReranker()
                self._reranker_backend_name = "keyword-fallback"
        return self._reranker

    def expand_query_locally(self, user_query: str) -> str:
        words = re.findall(r"\b\w+\b", user_query.lower())
        expanded_terms = [
            INDUSTRIAL_GLOSSARY[word] for word in words if word in INDUSTRIAL_GLOSSARY
        ]
        if expanded_terms:
            return f"{user_query} {' '.join(sorted(set(expanded_terms)))}"
        return user_query

    def search_and_rerank(
        self,
        original_query: str,
        expanded_query: str,
        score_threshold: float = -2.0,
    ) -> list[dict[str, Any]]:
        if not self.ensure_loaded():
            return []

        semantic_ranked: list[tuple[int, float]] = []
        if self._index is not None:
            query_vec = np.array([self.get_bi_encoder().encode(expanded_query)]).astype("float32")
            if len(query_vec.shape) == 3:
                query_vec = query_vec.reshape(query_vec.shape[0], query_vec.shape[-1])
            faiss.normalize_L2(query_vec)
            if hasattr(self._index, "d") and int(self._index.d) != int(query_vec.shape[1]):
                logger.error(
                    "Embedding dimension mismatch. Index=%s query=%s",
                    self._index.d,
                    query_vec.shape[1],
                )
            else:
                try:
                    distances, indices = self._index.search(query_vec, HYBRID_CANDIDATES)
                    semantic_ranked = [
                        (int(index), float(score))
                        for index, score in zip(indices[0], distances[0])
                        if index != -1 and index < len(self._metadata_chunks)
                    ]
                except Exception:
                    logger.exception("Vector search failed")

        bm25_ranked = self._bm25_index.search(expanded_query, HYBRID_CANDIDATES) if self._bm25_index else []
        semantic_rank = {index: rank for rank, (index, _) in enumerate(semantic_ranked, 1)}
        bm25_rank = {index: rank for rank, (index, _) in enumerate(bm25_ranked, 1)}
        semantic_scores = dict(semantic_ranked)
        bm25_scores = dict(bm25_ranked)

        document_boosts: dict[str, float] = defaultdict(float)
        for rank, (index, _) in enumerate(semantic_ranked[:FAISS_CANDIDATES], 1):
            document_boosts[self._document_key(self._metadata_chunks[index])] += 1.0 / (50 + rank)
        for rank, (index, _) in enumerate(bm25_ranked[:FAISS_CANDIDATES], 1):
            document_boosts[self._document_key(self._metadata_chunks[index])] += 1.0 / (50 + rank)

        fused_candidates = []
        for index in set(semantic_rank) | set(bm25_rank):
            score = 0.0
            if index in semantic_rank:
                score += 1.0 / (60 + semantic_rank[index])
            if index in bm25_rank:
                score += 1.0 / (60 + bm25_rank[index])
            score += 0.15 * document_boosts.get(self._document_key(self._metadata_chunks[index]), 0.0)
            fused_candidates.append((index, score))
        fused_candidates.sort(key=lambda item: item[1], reverse=True)

        candidates = []
        for index, fused_score in fused_candidates[:HYBRID_CANDIDATES]:
            candidates.append(
                {
                    **self._metadata_chunks[index],
                    "hybrid_score": float(fused_score),
                    "semantic_score": float(semantic_scores.get(index, 0.0)),
                    "bm25_score": float(bm25_scores.get(index, 0.0)),
                    "semantic_rank": semantic_rank.get(index),
                    "bm25_rank": bm25_rank.get(index),
                }
            )

        if not candidates:
            return []

        pairs = [[original_query, candidate["text"]] for candidate in candidates]
        scores = self.get_reranker().predict(pairs)
        ranked = sorted(
            [{**candidate, "rerank_score": float(score)} for candidate, score in zip(candidates, scores)],
            key=lambda item: item["rerank_score"],
            reverse=True,
        )

        if DEBUG:
            for rank, item in enumerate(ranked[:RERANK_TOP_K], 1):
                logger.debug(
                    "%s. %s page %s score=%.3f",
                    rank,
                    item.get("source_file"),
                    item.get("page"),
                    item.get("rerank_score", 0.0),
                )

        return [item for item in ranked if item["rerank_score"] >= score_threshold][:RERANK_TOP_K]

    def compute_confidence(self, results: list[dict[str, Any]]) -> int:
        if not results:
            return 0
        top = float(results[0].get("rerank_score", 0.0))
        if self._reranker_backend_name == "keyword-fallback":
            base = max(0.0, min(1.0, top))
        else:
            base = 1 / (1 + math.exp(-top))
        multi_source_bonus = min(len({item.get("source_file") for item in results}) - 1, 2) * 0.03
        lexical_bonus = min(float(results[0].get("bm25_score", 0.0)) / 20, 0.08)
        confidence = int(round(100 * min(base + multi_source_bonus + lexical_bonus, 0.99)))
        return max(0, min(confidence, 99))

    def _manual_confidence(self, results: list[dict[str, Any]]) -> str:
        if not results:
            return "no_manual_match"
        if self.compute_confidence(results) < MANUAL_ANSWER_MIN_CONFIDENCE:
            return "weak_manual_match"
        return "strong_manual_match"

    @staticmethod
    def _should_use_fallback(manual_confidence: str, retrieval_confidence_pct: int, context: str) -> bool:
        if manual_confidence != "strong_manual_match":
            return True
        if retrieval_confidence_pct < MANUAL_ANSWER_MIN_CONFIDENCE:
            return True
        empty_markers = (
            "No direct manual segments matched this query.",
            "No manuals are indexed yet.",
        )
        return any(marker in context for marker in empty_markers)

    @staticmethod
    def _ensure_source_tag(answer: str, required_tag: str) -> str:
        cleaned = str(answer or "").strip()
        cleaned = re.sub(
            r"^\s*\[(?:SOURCE:\s*(?:MANUAL|FALLBACK)|ANSWER:\s*(?:VERIFIED FROM MANUAL|GENERAL SAFETY GUIDANCE))\]\s*",
            "",
            cleaned,
            flags=re.IGNORECASE,
        )
        return f"{required_tag}\n{cleaned}".strip()

    def _build_fallback_answer(self, user_question: str, force_mode: str | None) -> str:
        mode_line = {
            "MODE A": "Use this as a diagnostic safety checklist, not an OEM-specific repair procedure.",
            "MODE B": "Use this as a generic maintenance safety checklist, not an OEM-specific procedure.",
            "MODE C": "Use this as a general engineering explanation, not a machine-specific specification.",
        }.get(force_mode or "", "Use this as general industrial troubleshooting guidance, not an OEM-specific instruction.")

        topic = self._infer_fallback_topic(user_question)
        checks = self._fallback_checks_for_topic(topic)
        return f"""[ANSWER: GENERAL SAFETY GUIDANCE]
The specific issue was not found in the uploaded manuals with enough confidence to provide a manual-grounded answer. {mode_line}

### Immediate Safety Controls
1. Stop the machine only through the approved operator control or emergency stop if there is immediate danger.
2. Apply LOTO before opening guards, panels, junction boxes, hydraulic cabinets, pneumatic lines, or rotating assemblies.
3. Verify zero hazardous energy: electrical isolation, discharged capacitors/VFD DC bus, relieved hydraulic and pneumatic pressure, blocked gravity loads, and stopped rotating parts.
4. Wear the required PPE for electrical, hydraulic, thermal, chemical, and sharp-edge hazards.
5. Do not bypass interlocks, overloads, pressure switches, guards, or safety relays to continue production.

### Generic Troubleshooting Path
{checks}

### Before Restart
1. Reinstall guards, covers, electrical-panel hardware, hose clamps, and lock tabs.
2. Remove tools, rags, and loose parts from the machine envelope.
3. Clear alarms using the OEM/HMI procedure only after the fault cause is corrected.
4. Restart at low speed or no-load where possible and monitor current draw, pressure, temperature, vibration, and abnormal noise.

### What To Upload Or Check Next
Upload the OEM troubleshooting section, electrical/hydraulic schematic, alarm-code table, lubrication schedule, or maintenance SOP for this machine so I can provide a source-backed answer.
"""

    @staticmethod
    def _infer_fallback_topic(user_question: str) -> str:
        text = user_question.lower()
        if any(word in text for word in ("overheat", "temperature", "hot", "thermal", "smoke", "burn")):
            return "overheating"
        if any(word in text for word in ("hydraulic", "pressure", "leak", "hose", "cylinder", "valve")):
            return "hydraulic"
        if any(word in text for word in ("electrical", "power", "breaker", "vfd", "motor", "voltage", "current")):
            return "electrical"
        if any(word in text for word in ("vibration", "noise", "bearing", "spindle", "shaft", "coupling")):
            return "rotating"
        if any(word in text for word in ("lubric", "grease", "oil")):
            return "lubrication"
        return "general"

    @staticmethod
    def _fallback_checks_for_topic(topic: str) -> str:
        checks_by_topic = {
            "overheating": [
                "Check for blocked airflow, clogged filters, failed fans, coolant loss, dirty heat exchangers, and excessive ambient temperature.",
                "Measure motor current against nameplate rating; high current can indicate overload, binding, poor lubrication, or phase imbalance.",
                "Inspect lubrication level/condition only after LOTO and stored-energy release.",
                "Look for recent process changes: higher load, longer duty cycle, tool wear, tighter material, or changed cycle timing.",
            ],
            "hydraulic": [
                "Relieve hydraulic pressure before loosening fittings; trapped pressure can inject fluid through skin.",
                "Check reservoir level, oil temperature, filter differential indicator, suction restrictions, and aeration/foaming.",
                "Inspect hoses, seals, cylinders, pump coupling, and valve blocks for leakage or heat discoloration.",
                "Compare standby and working pressure to normal plant baseline; do not adjust relief valves without OEM/SOP limits.",
            ],
            "electrical": [
                "Use electrically qualified personnel for live testing; otherwise isolate and verify absence of voltage.",
                "Check incoming power, phase loss/imbalance, loose terminals, overheated contactors, overload trips, fuses, and ground faults.",
                "Inspect VFD/servo drive alarms, cooling fans, cabinet filters, and DC bus discharge time before touching conductors.",
                "Do not upsize fuses, bypass overloads, or reset repeatedly without finding the root cause.",
            ],
            "rotating": [
                "Keep guards installed during observation and apply LOTO before touching belts, couplings, bearings, spindles, or shafts.",
                "Check lubrication condition, belt/chain tension, coupling alignment, loose mounts, imbalance, worn bearings, and abnormal runout.",
                "Trend vibration, temperature, and noise from a safe measurement point.",
                "Stop operation if grinding, metal contact, smoke, rapid temperature rise, or severe vibration is present.",
            ],
            "lubrication": [
                "Apply LOTO before accessing lubrication points near moving parts, pinch points, or hot surfaces.",
                "Verify lubricant type, cleanliness, level, contamination, blocked lines, damaged fittings, and automatic lubricator operation.",
                "Do not mix grease/oil types unless approved by the OEM or plant lubrication standard.",
                "If interval or quantity is unknown, avoid guessing; retrieve the OEM lubrication schedule before servicing.",
            ],
            "general": [
                "Confirm the exact machine model, serial number, alarm code, operating mode, and recent maintenance changes.",
                "Check basic utilities first: electrical supply, compressed air, hydraulic pressure, coolant, lubrication, and guarding/interlocks.",
                "Inspect for visible damage, leaks, loose connectors, abnormal smell, heat, noise, vibration, and contamination.",
                "Compare current readings to normal baseline values before making adjustments.",
            ],
        }
        lines = checks_by_topic.get(topic, checks_by_topic["general"])
        return "\n".join(f"{index}. {line}" for index, line in enumerate(lines, 1))

    def _build_context_and_evidence(
        self,
        results: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        context_parts = []
        evidence = []
        confidence_pct = self.compute_confidence(results)

        for item in results:
            filename = item.get("source_file", "Unknown")
            page = item.get("page", "?")
            section_title = item.get("section_title", "General")
            raw_text = str(item.get("text", "")).strip()
            chunk_id = item.get("chunk_id", "?")
            extraction = item.get("extraction", "native")

            context_parts.append(
                f"[File: {filename} | Page: {page} | Section: {section_title} | "
                f"Extraction: {extraction} | Confidence: {confidence_pct}%]\n{raw_text}"
            )

            sentences = _ABBREV_SAFE_SPLIT.split(raw_text)
            snippet = " ".join(sentences[:4])
            if len(sentences) > 4:
                snippet += "..."

            evidence.append(
                {
                    "file": filename,
                    "page": page,
                    "chunk_id": chunk_id,
                    "document_id": item.get("document_id"),
                    "section": section_title,
                    "machine_type": item.get("machine_type", "Unknown"),
                    "manufacturer": item.get("manufacturer", "Unknown"),
                    "revision": item.get("revision", "unknown"),
                    "source": item.get("source", filename),
                    "snippet": snippet,
                    "extraction": extraction,
                    "score": round(float(item.get("rerank_score", 0)), 3),
                    "semantic_score": round(float(item.get("semantic_score", 0)), 3),
                    "bm25_score": round(float(item.get("bm25_score", 0)), 3),
                    "hybrid_score": round(float(item.get("hybrid_score", 0)), 4),
                    "confidence": confidence_pct,
                }
            )

        context = "\n---\n".join(context_parts) or "No direct manual segments matched this query."
        return context, evidence

    @staticmethod
    def _clean_for_memory(text: str) -> str:
        text = re.sub(r"\s+\[FORCE MODE [ABC]: [^\]]+\]$", "", str(text))
        text = re.sub(r"\s+", " ", text).strip()
        return text[:1200]

    def _build_conversation_memory(
        self,
        conversation_history: list[dict[str, Any]] | None,
        user_memory: list[str] | str | None,
    ) -> str:
        parts = []
        if user_memory:
            notes = [user_memory] if isinstance(user_memory, str) else [str(note) for note in user_memory if str(note).strip()]
            if notes:
                parts.append("Persistent case notes:\n" + "\n".join(f"- {note}" for note in notes))

        if conversation_history:
            lines = []
            for message in conversation_history[-8:]:
                role = str(message.get("role", "user")).title()
                content = self._clean_for_memory(message.get("content", ""))
                if content:
                    lines.append(f"{role}: {content}")
            if lines:
                parts.append("Recent conversation:\n" + "\n".join(lines))

        return "\n\n".join(parts) if parts else "No prior case memory."

    def _build_prompt(
        self,
        user_question: str,
        context: str,
        conversation_memory: str,
        manual_confidence: str,
        retrieval_confidence_pct: int,
        force_mode: str | None = None,
    ) -> str:
        mode_instruction = ""
        if force_mode:
            mode_map = {
                "MODE A": "Use MODE A: Diagnostic / troubleshooting layout.",
                "MODE B": "Use MODE B: Procedural / maintenance layout.",
                "MODE C": "Use MODE C: Conceptual / component explanation layout.",
            }
            mode_instruction = f"\nForced response layout: {mode_map.get(force_mode, force_mode)}\n"

        return f"""
You are Maintenance Copilot, an industrial AI assistant for technicians and engineers.
You answer from uploaded industrial documentation. Use case memory only to resolve references and maintain continuity.
Do not invent machine-specific procedures, intervals, torque values, part numbers, error meanings, or safety limits.
This request has already passed the backend manual-confidence gate. Your response must start with the exact tag:
[ANSWER: VERIFIED FROM MANUAL]

Style:
- Clear, crisp, technical, practical, and professional.
- Sound like a senior maintenance engineer who can reason through faults, not like a document search bot.
- Help the user move toward a fix: isolate likely causes, ask for missing readings, and call out checks.
- Explain your reasoning briefly enough that the technician understands why each check matters.
- If a safety step is uncertain, advise verification in the OEM manual or plant SOP before action.

{mode_instruction}

Response routing:

MODE A - Diagnostic / troubleshooting
Use for faults, symptoms, alarms, errors, overheating, leakage, vibration, noise, trips, or downtime.

MODE B - Procedure / maintenance
Use for replace, install, adjust, service, calibrate, lubricate, inspect, clean, or reset.

MODE C - Concept / component explanation
Use for what is, how it works, purpose, specifications, and component descriptions.

Rules:
1. Start with exactly [ANSWER: VERIFIED FROM MANUAL].
2. Omit sections that are not supported by the retrieved context.
3. Keep source references as file name, page number, and section title.
4. If OCR text looks uncertain, mention that the source came from OCR and should be verified against the page image.
5. For safety-critical work, include lockout, stored energy, pressure, thermal, rotating equipment, or PPE cautions only when relevant.
6. Use only the retrieved manual context for machine-specific values, procedures, intervals, alarm meanings, and limits.
7. Lead with the manual-grounded answer and cite the source references.
8. Include "### Confidence" with the confidence percentage from the retrieved context.
9. Include "### Evidence" with one or more short supporting excerpts.
10. End with "### Follow-Up Questions" containing 2-4 useful next questions.

Manual retrieval confidence:
{manual_confidence}

Confidence percentage:
{retrieval_confidence_pct}%

Case memory:
{conversation_memory}

Retrieved manual context:
{context}

User question:
{user_question}
""".strip()

    @staticmethod
    def _extract_forced_mode(user_question: str, force_mode: str | None) -> tuple[str, str | None]:
        if force_mode in {"MODE A", "MODE B", "MODE C"}:
            return user_question, force_mode
        mode_pattern = re.search(r"\[FORCE (MODE [ABC]): [^\]]+\]", user_question)
        if mode_pattern:
            return user_question[: mode_pattern.start()].strip(), mode_pattern.group(1)
        readable_map = {
            "Diagnostic": "MODE A",
            "Procedural": "MODE B",
            "Procedure": "MODE B",
            "Conceptual": "MODE C",
            "Concept": "MODE C",
        }
        return user_question, readable_map.get(str(force_mode), force_mode)

    def _generate_with_fallback(self, prompt: str) -> str:
        client = self.get_client()
        errors: list[str] = []
        attempts = 0
        for model_name in GENERATION_MODELS:
            if attempts >= MAX_GENERATION_ATTEMPTS:
                break
            attempts += 1
            try:
                chat = client.chats.create(model=model_name)
                response = chat.send_message(prompt)
                if response and getattr(response, "text", None):
                    return response.text
                errors.append(f"{model_name}: empty response")
            except Exception as exc:
                errors.append(f"{model_name}: {exc}")
        raise RuntimeError("; ".join(errors) if errors else "No generation attempt was made.")


_service_cache: dict[uuid.UUID, RetrievalService] = {}


def get_retrieval_service(organization_id: uuid.UUID | str | None = None) -> RetrievalService:
    tenant = resolve_tenant(organization_id)
    service = _service_cache.get(tenant.organization_id)
    if service is None:
        service = RetrievalService(tenant)
        _service_cache[tenant.organization_id] = service
    return service
