# MaintenanceCopilot AI Enterprise Architecture

This repository currently ships a Streamlit pilot that proves the core RAG
workflow. The production SaaS architecture should preserve the same retrieval
contract while splitting responsibilities into deployable services.

## Current Pilot

- `app.py`: operator-facing upload, document browser, chat, history, and citation UI
- `config.py`: centralized storage paths, directory bootstrap, legacy migration
- `main.py`: document extraction, OCR, section detection, chunking, embeddings, FAISS persistence
- `query.py`: query expansion, hybrid retrieval, reranking, confidence, answer generation

## Target SaaS Architecture

```text
Next.js + React UI
        |
FastAPI API gateway
        |
PostgreSQL metadata + auth + audit logs
        |
Object storage for manuals and page images
        |
Celery workers for extraction, OCR, embeddings, and indexing
        |
Vector store adapter
  |-- FAISS for pilot/on-prem
  |-- pgvector/Qdrant/Milvus/Pinecone/Weaviate for scale
        |
Provider-agnostic LLM and embedding adapters
```

## Production Services

- Auth service: organizations, factories, departments, roles, users, permissions
- Document service: upload validation, versioning, storage, metadata extraction
- Ingestion worker: PDF/DOCX/TXT extraction, OCR, table parsing, section detection
- Index service: embeddings, incremental indexing, rebuilds, deletion, vector-store adapters
- Retrieval service: hierarchical document retrieval, FAISS/vector search, BM25, reranking
- Answer service: grounded LLM prompts, citation formatting, confidence, refusal behavior
- Evaluation service: benchmark questions, recall@K, precision@K, MRR, NDCG, latency, OCR success
- Audit service: uploads, downloads, searches, answers, admin actions

## Data Model

Production tables (Neon PostgreSQL):

- `organizations` — tenant container with `subdomain` and `plan_tier`
- `users` — tenant staff with role-based access (`admin`, `manager`, `operator`)
- `api_keys` — external ingestion credentials
- `documents` — uploaded manual metadata and processing status
- `document_chunks` — chunk text, page numbers, and JSONB retrieval metadata
- `chat_sessions` — conversation threads
- `chat_messages` — user/assistant/system messages with citations
- `audit_logs` — enterprise audit trail

See [`docs/DATABASE_SCHEMA.md`](DATABASE_SCHEMA.md) and [`database/schema.sql`](../database/schema.sql).

## Retrieval Contract

Every answer should be produced from this pipeline:

```text
Question
-> query expansion
-> document-level retrieval
-> chunk-level semantic retrieval
-> BM25 keyword retrieval
-> reciprocal-rank fusion
-> cross-encoder reranking
-> top 5-8 evidence chunks
-> confidence score
-> grounded LLM answer
-> citations, excerpts, follow-up questions
```

## SaaS Readiness Checklist

- Multi-tenant auth and authorization
- Document permissions and private workspaces
- Version-safe uploads
- Background ingestion queue
- Durable metadata database
- Object storage for originals
- Vector-store abstraction
- Audit logs
- Rate limiting
- Evaluation dashboard
- Deployment templates for cloud and on-prem
