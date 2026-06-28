# Data Layout

Maintenance Copilot stores runtime files under `data/`. Path definitions live in
`config.py` and `core/tenant.py` so uploads, vector indexes, and chat history stay
consistent across Streamlit, indexing, and retrieval.

## Single-Tenant Pilot Layout

```text
data/
|-- uploads/                     Legacy pilot uploads (auto-migrated)
|-- faiss/                       Legacy pilot indexes (auto-migrated)
|-- chat_history/
|   `-- chat_history.json
`-- tenants/
    `-- {organization_id}/
        |-- uploads/             Tenant document storage
        `-- faiss/
            |-- maintenance_index.faiss
            |-- chunks_mapping.pkl
            `-- processed_files.pkl
```

## SaaS Tenant Isolation

Each organization receives an isolated directory tree:

| Path | Purpose |
|---|---|
| `data/tenants/{org_id}/uploads/` | Uploaded PDF, DOCX, TXT, Markdown, and image files |
| `data/tenants/{org_id}/faiss/maintenance_index.faiss` | Tenant FAISS vector index |
| `data/tenants/{org_id}/faiss/chunks_mapping.pkl` | Chunk metadata mapped to vectors |
| `data/tenants/{org_id}/faiss/processed_files.pkl` | Incremental indexing cache |

PostgreSQL stores durable document metadata when `DATABASE_URL` is configured:

| Table | Purpose |
|---|---|
| `documents` | File registry, checksum, processing status |
| `document_chunks` | Chunk text, page numbers, vector IDs |

## Configuration

| Variable | Purpose |
|---|---|
| `DEFAULT_ORGANIZATION_ID` | Active tenant for Streamlit pilot mode |
| `DATABASE_URL` | Enables PostgreSQL metadata sync (optional locally) |

## Startup Behavior

1. `config.initialize_storage()` creates legacy `data/` folders and migrates old files.
2. `core.tenant.resolve_tenant()` resolves the active organization.
3. `migrate_legacy_tenant_storage()` moves legacy pilot files into the tenant tree.
4. Indexing and retrieval operate only within the resolved tenant scope.

## Service Layer

| Module | Responsibility |
|---|---|
| `services/indexing_service.py` | Extraction orchestration, PostgreSQL sync, FAISS persistence |
| `services/retrieval_service.py` | Hybrid retrieval, reranking, Gemini answer generation |
| `repositories/document_repository.py` | PostgreSQL document/chunk persistence |
| `repositories/vector_repository.py` | Tenant-scoped FAISS read/write |

`main.py` and `query.py` remain backward-compatible facades for the Streamlit app
and future FastAPI routes.
