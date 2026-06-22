"""
Retrieval and answer generation engine for Maintenance Copilot.

The module loads heavyweight models lazily so the Streamlit UI can start
quickly, show useful status, and index new uploads before the first query.
"""

from __future__ import annotations

import os
import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from sklearn.feature_extraction.text import HashingVectorizer


DEBUG = os.getenv("DEBUG_RETRIEVAL", "false").lower() == "true"

BASE_DIR = Path(__file__).resolve().parent
INDEX_PATH = BASE_DIR / "maintenance_index.faiss"
MAPPING_PATH = BASE_DIR / "chunks_mapping.pkl"

load_dotenv(BASE_DIR / ".env")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if os.getenv("ALLOW_MODEL_DOWNLOADS", "false").lower() not in {"1", "true", "yes"}:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
FAISS_CANDIDATES = int(os.getenv("FAISS_CANDIDATES", "30"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "8"))
GENERAL_GUIDANCE_THRESHOLD = float(os.getenv("GENERAL_GUIDANCE_THRESHOLD", "-0.25"))
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

_client: genai.Client | None = None
_bi_encoder: Any | None = None
_reranker: Any | None = None
_embedding_backend_name = ""
_reranker_backend_name = ""
_index: Any | None = None
_metadata_chunks: list[dict[str, Any]] = []
_database_stamp: tuple[float, float] | None = None


class HashingEmbeddingModel:
    """Dependency-light query embedding fallback for broken Torch/transformers installs."""

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
    """Small fallback reranker based on technical token overlap."""

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

_ABBREV_SAFE_SPLIT = re.compile(
    r"(?<!Rev)(?<!Fig)(?<!approx)(?<!SOP)(?<!min)(?<!max)(?<=[.!?])\s+",
    re.IGNORECASE,
)


def get_client() -> genai.Client:
    global _client
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is missing. Add it to .env before asking questions.")

    if _client is None:
        _client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=types.HttpOptions(
                retry_options=types.HttpRetryOptions(
                    attempts=3,
                    initial_delay=2.0,
                    http_status_codes=[429, 503],
                )
            ),
        )
    return _client


def get_bi_encoder() -> Any:
    global _bi_encoder, _embedding_backend_name
    if _bi_encoder is None:
        try:
            from sentence_transformers import SentenceTransformer

            print(f"Loading retrieval encoder: {BI_ENCODER_MODEL}")
            _bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)
            _embedding_backend_name = f"sentence-transformers:{BI_ENCODER_MODEL}"
        except Exception as exc:
            print(
                "SentenceTransformer retrieval encoder could not start. "
                f"Using local hashing embeddings instead. Details: {exc}"
            )
            _bi_encoder = HashingEmbeddingModel(
                n_features=int(os.getenv("HASH_EMBEDDING_FEATURES", "384"))
            )
            _embedding_backend_name = "hashing-fallback"
    return _bi_encoder


def get_reranker() -> Any:
    global _reranker, _reranker_backend_name
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder

            print(f"Loading reranker: {RERANKER_MODEL}")
            _reranker = CrossEncoder(RERANKER_MODEL)
            _reranker_backend_name = f"cross-encoder:{RERANKER_MODEL}"
        except Exception as exc:
            print(
                "CrossEncoder reranker could not start. "
                f"Using keyword reranker instead. Details: {exc}"
            )
            _reranker = KeywordReranker()
            _reranker_backend_name = "keyword-fallback"
    return _reranker


def _current_database_stamp() -> tuple[float, float] | None:
    if not INDEX_PATH.exists() or not MAPPING_PATH.exists():
        return None
    return (INDEX_PATH.stat().st_mtime, MAPPING_PATH.stat().st_mtime)


def reload_vector_database(force: bool = False) -> dict[str, Any]:
    global _index, _metadata_chunks, _database_stamp

    stamp = _current_database_stamp()
    if stamp is None:
        _index = None
        _metadata_chunks = []
        _database_stamp = None
        return get_database_summary()

    if not force and _index is not None and _database_stamp == stamp:
        return get_database_summary()

    _index = faiss.read_index(str(INDEX_PATH))
    with MAPPING_PATH.open("rb") as file:
        _metadata_chunks = pickle.load(file)
    _database_stamp = stamp
    return get_database_summary()


def get_database_summary() -> dict[str, Any]:
    files = sorted(
        {chunk.get("source_file", "Unknown") for chunk in _metadata_chunks}
    )
    extraction_counts: dict[str, int] = {}
    document_type_files: dict[str, set[str]] = {}
    indexed_backends = set()
    ocr_page_keys = set()
    for chunk in _metadata_chunks:
        method = str(chunk.get("extraction", "native"))
        extraction_counts[method] = extraction_counts.get(method, 0) + 1
        source_file = str(chunk.get("source_file", ""))
        suffix = Path(source_file).suffix.lower() or "unknown"
        document_type_files.setdefault(suffix, set()).add(source_file)
        if chunk.get("embedding_backend"):
            indexed_backends.add(str(chunk.get("embedding_backend")))
        if "ocr" in method:
            ocr_page_keys.add((chunk.get("source_file"), chunk.get("page")))
    document_types = {
        suffix: len(source_files)
        for suffix, source_files in document_type_files.items()
    }

    return {
        "available": _index is not None and bool(_metadata_chunks),
        "chunks": len(_metadata_chunks),
        "vectors": _index.ntotal if _index is not None else 0,
        "files": files,
        "file_count": len(files),
        "extraction_counts": extraction_counts,
        "document_types": document_types,
        "ocr_pages": len(ocr_page_keys),
        "indexed_embedding_backends": sorted(indexed_backends),
        "runtime_embedding_backend": _embedding_backend_name or "not-loaded",
        "runtime_reranker_backend": _reranker_backend_name or "not-loaded",
        "api_key_configured": bool(GEMINI_API_KEY),
        "generation_models": GENERATION_MODELS,
    }


def ensure_vector_database_loaded() -> bool:
    reload_vector_database()
    return _index is not None and bool(_metadata_chunks)


def expand_query_locally(user_query: str) -> str:
    words = re.findall(r"\b\w+\b", user_query.lower())
    expanded_terms = []
    for word in words:
        if word in INDUSTRIAL_GLOSSARY:
            expanded_terms.append(INDUSTRIAL_GLOSSARY[word])
    if expanded_terms:
        return f"{user_query} {' '.join(sorted(set(expanded_terms)))}"
    return user_query


def search_and_rerank(
    original_query: str,
    expanded_query: str,
    score_threshold: float = -2.0,
) -> list[dict[str, Any]]:
    if not ensure_vector_database_loaded():
        return []

    query_vec = np.array([get_bi_encoder().encode(expanded_query)]).astype("float32")
    if len(query_vec.shape) == 3:
        query_vec = query_vec.reshape(query_vec.shape[0], query_vec.shape[-1])
    faiss.normalize_L2(query_vec)
    if hasattr(_index, "d") and int(_index.d) != int(query_vec.shape[1]):
        print(
            "Embedding dimension mismatch. "
            f"Index uses {_index.d}, query backend produced {query_vec.shape[1]}. "
            "Rebuild the index from the app menu."
        )
        return []

    try:
        distances, indices = _index.search(query_vec, FAISS_CANDIDATES)
    except Exception as exc:
        print(f"Vector search failed: {exc}")
        return []
    candidates = [
        _metadata_chunks[index]
        for index in indices[0]
        if index != -1 and index < len(_metadata_chunks)
    ]
    if not candidates:
        return []

    pairs = [[original_query, candidate["text"]] for candidate in candidates]
    scores = get_reranker().predict(pairs)

    ranked = sorted(
        [
            {
                **candidate,
                "rerank_score": float(score),
                "vector_score": float(vector_score),
            }
            for candidate, score, vector_score in zip(candidates, scores, distances[0])
        ],
        key=lambda item: item["rerank_score"],
        reverse=True,
    )

    if DEBUG:
        print("\n[Retrieval debug]")
        for rank, item in enumerate(ranked[:RERANK_TOP_K], 1):
            print(
                f"{rank}. {item.get('source_file')} page {item.get('page')} "
                f"score={item.get('rerank_score'):.3f}"
            )

    return [
        item for item in ranked
        if item["rerank_score"] >= score_threshold
    ][:RERANK_TOP_K]


def _build_context_and_evidence(results: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    context_parts = []
    evidence = []

    for item in results:
        filename = item.get("source_file", "Unknown")
        page = item.get("page", "?")
        raw_text = str(item.get("text", "")).strip()
        chunk_id = item.get("chunk_id", "?")
        extraction = item.get("extraction", "native")

        context_parts.append(
            f"[File: {filename} | Page: {page} | Extraction: {extraction}]\n{raw_text}"
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
                "snippet": snippet,
                "extraction": extraction,
                "score": round(float(item.get("rerank_score", 0)), 3),
            }
        )

    context = "\n---\n".join(context_parts)
    if not context:
        context = "No direct manual segments matched this query."
    return context, evidence


def _manual_confidence(results: list[dict[str, Any]]) -> str:
    if not results:
        return "no_manual_match"
    top_score = float(results[0].get("rerank_score", -999))
    if top_score < GENERAL_GUIDANCE_THRESHOLD:
        return "weak_manual_match"
    return "strong_manual_match"


def _clean_for_memory(text: str) -> str:
    text = re.sub(r"\s+\[FORCE MODE [ABC]: [^\]]+\]$", "", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    return text[:1200]


def _build_conversation_memory(
    conversation_history: list[dict[str, Any]] | None,
    user_memory: list[str] | str | None,
) -> str:
    parts = []

    if user_memory:
        if isinstance(user_memory, str):
            notes = [user_memory]
        else:
            notes = [str(note) for note in user_memory if str(note).strip()]
        if notes:
            parts.append("Persistent case notes:\n" + "\n".join(f"- {note}" for note in notes))

    if conversation_history:
        recent = conversation_history[-8:]
        lines = []
        for message in recent:
            role = str(message.get("role", "user")).title()
            content = _clean_for_memory(message.get("content", ""))
            if content:
                lines.append(f"{role}: {content}")
        if lines:
            parts.append("Recent conversation:\n" + "\n".join(lines))

    return "\n\n".join(parts) if parts else "No prior case memory."


def _build_prompt(
    user_question: str,
    context: str,
    conversation_memory: str,
    manual_confidence: str,
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
You combine document-grounded maintenance knowledge with careful general engineering reasoning.
Use retrieved manuals first. Use case memory for continuity. Use general maintenance knowledge only when the manuals do not answer the question or the user asks a broad/random question.

Style:
- Clear, technical, practical, and conversational.
- Sound like a senior maintenance engineer who can reason through faults, not like a document search bot.
- Help the user move toward a fix: isolate likely causes, ask for missing readings, and call out checks.
- Explain your reasoning briefly enough that the technician understands why each check matters.
- Never invent machine-specific torque values, intervals, part numbers, alarm meanings, or procedures.
- If the manuals do not contain the answer, say that clearly, then provide general engineering guidance separately.
- If a safety step is uncertain, advise verification in the OEM manual or plant SOP before action.

{mode_instruction}

Response routing:

MODE A - Diagnostic / troubleshooting
Use for faults, symptoms, alarms, errors, overheating, leakage, vibration, noise, trips, or downtime.
Use these sections when supported by context:
### Summary
### Immediate Risk
### Possible Causes
### Inspection And Action Steps
### What To Measure Next
### Safety Precautions
### Source References

MODE B - Procedure / maintenance
Use for replace, install, adjust, service, calibrate, lubricate, inspect, clean, or reset.
Use these sections when supported by context:
### Task Overview
### Before You Start
### Required Tools And Materials
### Step By Step Procedure
### Tolerances And Safety Limits
### Source References

MODE C - Concept / component explanation
Use for what is, how it works, purpose, specifications, and component descriptions.
Use these sections when supported by context:
### Definition And Purpose
### How It Works
### Key Specifications
### Source References

Rules:
1. Choose one mode only.
2. Omit sections that are not supported by the retrieved context.
3. Keep source references as file name and page number.
4. If OCR text looks uncertain, mention that the source came from OCR and should be verified against the page image.
5. For safety-critical work, include lockout, stored energy, pressure, thermal, rotating equipment, or PPE cautions only when relevant.
6. When manual confidence is weak/no match, include a section named "### General Engineering Guidance" and do not pretend it came from the manual.
7. When manual confidence is strong, lead with the manual-grounded answer and cite the source references.
8. End with "### Next Best Action" containing one concise practical next step.
9. If the user asks a casual/random question, answer naturally, then connect back to how it affects maintenance work if relevant.

Manual retrieval confidence:
{manual_confidence}

Case memory:
{conversation_memory}

Retrieved manual context:
{context}

User question:
{user_question}
""".strip()


def _extract_forced_mode(user_question: str, force_mode: str | None) -> tuple[str, str | None]:
    if force_mode in {"MODE A", "MODE B", "MODE C"}:
        return user_question, force_mode

    mode_pattern = re.search(r"\[FORCE (MODE [ABC]): [^\]]+\]", user_question)
    if mode_pattern:
        clean_question = user_question[: mode_pattern.start()].strip()
        return clean_question, mode_pattern.group(1)

    readable_map = {
        "Diagnostic": "MODE A",
        "Procedural": "MODE B",
        "Procedure": "MODE B",
        "Conceptual": "MODE C",
        "Concept": "MODE C",
    }
    return user_question, readable_map.get(str(force_mode), force_mode)


def _generate_with_fallback(prompt: str) -> str:
    client = get_client()
    errors = []
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


def ask_copilot(
    user_question: str,
    conversation_history: list[dict[str, Any]] | None = None,
    user_memory: list[str] | str | None = None,
    force_mode: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Return a grounded answer and source evidence for a technician question.
    """
    clean_question, resolved_mode = _extract_forced_mode(user_question.strip(), force_mode)
    if not clean_question:
        return "Please enter a maintenance question or fault description.", []

    if ensure_vector_database_loaded():
        expanded_query = expand_query_locally(clean_question)
        results = search_and_rerank(clean_question, expanded_query)
    else:
        results = []
    context, evidence = _build_context_and_evidence(results)
    manual_confidence = _manual_confidence(results)
    if not ensure_vector_database_loaded():
        context = (
            "No manuals are indexed yet. The answer must be general engineering "
            "guidance unless the user supplied enough case details in memory."
        )
        manual_confidence = "no_indexed_manuals"
    conversation_memory = _build_conversation_memory(conversation_history, user_memory)
    prompt = _build_prompt(
        clean_question,
        context,
        conversation_memory,
        manual_confidence,
        resolved_mode,
    )

    try:
        answer = _generate_with_fallback(prompt)
        return answer, evidence
    except Exception as exc:
        return (
            "Connection error while generating the answer. "
            f"Details: {exc}\n\nCheck GEMINI_API_KEY, model access, and network connectivity.",
            evidence,
        )
