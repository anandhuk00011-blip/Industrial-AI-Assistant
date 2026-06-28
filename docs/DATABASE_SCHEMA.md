# Production Database Schema

Maintenance Copilot uses a multi-tenant PostgreSQL schema on Neon. Every row is
scoped by `organization_id` except global auth identifiers such as `users.email`.

## Table Map

| Table | Purpose |
|---|---|
| `organizations` | Tenant container (company / factory) |
| `users` | Operators, managers, and admins within a tenant |
| `api_keys` | External ingestion and automation credentials |
| `documents` | Uploaded manual and SOP metadata |
| `document_chunks` | Chunk text, page numbers, and retrieval metadata |
| `chat_sessions` | Conversation threads (`+ New Chat`) |
| `chat_messages` | Individual user/assistant/system messages |
| `audit_logs` | Enterprise security and compliance audit trail |

## Tenant Isolation Rules

- Every query must filter by `organization_id`.
- `document_chunks.organization_id` is denormalized for fast tenant-scoped retrieval.
- `chat_messages.organization_id` is denormalized for analytics and audit queries.
- Cascading deletes remove tenant data when an organization is removed.

## Provision Neon

### Option A — SQL script (recommended for production)

Run the full script in the Neon SQL Editor:

```bash
# or from CLI
psql "$DATABASE_URL" -f database/schema.sql
```

Source file: [`database/schema.sql`](../database/schema.sql)

### Option B — SQLAlchemy bootstrap (development)

```bash
python database/database.py
```

### Seed demo tenant + operator user

```bash
python database/seed.py
```

Set these in `.env` before seeding:

```text
DATABASE_URL=postgresql+psycopg2://...
DEFAULT_ORGANIZATION_ID=00000000-0000-4000-8000-000000000001
DEFAULT_ORGANIZATION_NAME=Demo Manufacturing Corp
```

## Upgrading From the Previous Pilot Schema

If you already created older tables (`conversations`, `messages`, `billing`, etc.),
provision a fresh Neon branch or run:

```sql
DROP TABLE IF EXISTS messages CASCADE;
DROP TABLE IF EXISTS conversations CASCADE;
DROP TABLE IF EXISTS usage_analytics CASCADE;
DROP TABLE IF EXISTS billing CASCADE;
DROP TABLE IF EXISTS document_chunks CASCADE;
DROP TABLE IF EXISTS documents CASCADE;
DROP TABLE IF EXISTS api_keys CASCADE;
DROP TABLE IF EXISTS audit_logs CASCADE;
DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS organizations CASCADE;
```

Then apply `database/schema.sql`.

## Indexes

Production indexes are defined in `database/schema.sql`:

- `idx_chunks_org` on `document_chunks(organization_id)`
- `idx_messages_session` on `chat_messages(session_id)`
- `idx_docs_checksum` on `documents(md5_checksum)`
- Additional org/user/session indexes for B2B scale

## Chunk Metadata JSONB

Vector and extraction metadata that does not belong in core columns is stored in
`document_chunks.metadata`:

```json
{
  "vector_index_id": "uuid_12",
  "embedding_backend": "sentence-transformers:all-MiniLM-L6-v2",
  "section_title": "Spindle Lubrication",
  "machine_type": "Lathe",
  "manufacturer": "Unknown",
  "revision": "2008",
  "language": "en",
  "extraction": "native+ocr"
}
```

This keeps the SQL schema stable while the RAG pipeline evolves.
