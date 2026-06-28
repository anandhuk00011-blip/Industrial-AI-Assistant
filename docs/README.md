# Maintenance Copilot

Maintenance Copilot is an industrial knowledge assistant for technicians,
reliability engineers, and maintenance teams. It lets users upload equipment
manuals, SOPs, maintenance guides, Word documents, and photos of manual pages,
indexes the content into a tenant-scoped FAISS knowledge base, and answers
maintenance questions with source-backed responses.

The project now includes two product surfaces:

- FastAPI backend for SaaS authentication, tenant-scoped uploads, indexing,
  chat, memory, and audit-ready persistence
- Streamlit frontend client for interviews, local pilots, and quick product demos

## Highlights

- Direct upload from the in-app menu
- FastAPI backend with `/docs` OpenAPI documentation
- Email/password login with signed bearer tokens
- Organization-scoped tenant isolation for uploads, indexes, conversations, and audit trails
- Tenant-scoped document deletion with index rebuild
- Streamed uploads with file count and file size limits
- Supports PDF, DOCX, TXT, Markdown, PNG, JPG, TIFF, BMP, and WebP
- Native PDF text extraction with PyMuPDF
- Table-aware PDF/DOCX extraction for schedules, torque charts, error-code tables, and parts lists
- OCR fallback for scanned pages, embedded DOCX images, and uploaded page photos
- Best-effort OCR support for handwritten notes when Tesseract can read them
- Incremental indexing with FAISS and local sentence-transformer embeddings
- Hybrid retrieval: FAISS semantic search plus BM25 keyword search, fused before reranking
- Hierarchical document-aware boosting before final chunk selection
- Retrieval confidence scoring with visible evidence cards
- Gemini answer generation with source references
- Database-backed case history, uploaded-file audit logs, and answer citations
- Persistent case memory notes for equipment details, symptoms, readings, and constraints
- Response modes for diagnostics, procedures, and component explanations
- Evidence-first responses that refuse unsupported machine-specific claims
- Version-safe uploads that preserve previous revisions instead of overwriting them

## Project Structure

```text
maintenance-copilot/
|-- app.py                 Streamlit frontend client for the FastAPI API
|-- api/
|   |-- main.py            FastAPI SaaS backend
|   |-- schemas.py         API request/response contracts
|   `-- security.py        Bearer token auth helpers
|-- main.py                Document extraction, OCR, chunking, and FAISS indexing
|-- query.py               Retrieval, reranking, prompt building, and Gemini calls
|-- config.py              Central path configuration and legacy storage migration
|-- requirements.txt       Python dependencies
|-- .env.example           Environment variable template
|-- data/
|   |-- uploads/           Uploaded manuals, SOPs, DOCX files, and page photos
|   |-- faiss/             Generated vector index and indexing caches
|   |   |-- maintenance_index.faiss
|   |   |-- chunks_mapping.pkl
|   |   `-- processed_files.pkl
|   `-- chat_history/      Local conversation history
|       `-- chat_history.json
```

Generated indexes, uploaded manuals, logs, secrets, and local databases are ignored
by `.gitignore` so the repository is safe to publish. On startup, any files still
in the legacy `data_input/` folder or project root are moved automatically into
`data/uploads/` and `data/faiss/`.

## Setup

1. Install Python dependencies.

```bash
pip install -r requirements.txt
```

2. Copy the environment template and add your Gemini API key.

```bash
copy .env.example .env
```

Set:

```text
GEMINI_API_KEY=your_key_here
API_SECRET_KEY=a_long_random_secret_for_api_tokens
FASTAPI_BASE_URL=http://127.0.0.1:8000
```

For SaaS production, also set:

```text
DATABASE_URL=postgresql+psycopg://user:password@host/dbname?sslmode=require
```

If you have a production `DATABASE_URL` in `.env` but want to run locally
without Neon, set:

```text
USE_LOCAL_DATABASE=true
```

3. Install the native Tesseract OCR executable for scanned PDFs.

On Windows, install Tesseract and either add it to `PATH` or set:

```text
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

The Python package `pytesseract` is already listed in `requirements.txt`, but
OCR also needs the native Tesseract program.

## Run

Start the FastAPI backend:

```bash
python run_api.py
```

For hot reload during backend development:

```bash
set API_RELOAD=true
python run_api.py
```

Then open:

```text
http://127.0.0.1:8000/docs
```

Useful API flow:

1. `POST /api/auth/register` or `POST /api/auth/login`
2. Copy the returned bearer token into the `/docs` Authorize button
3. `POST /api/documents/upload`
4. `GET /api/knowledge/summary`
5. `POST /api/conversations`
6. `POST /api/conversations/{conversation_id}/ask`

Start the Streamlit frontend client in a second terminal:

```bash
streamlit run app.py
```

Then open the local Streamlit URL. The Streamlit app does not call Gemini,
FAISS, SQLAlchemy, or indexing code directly; it sends login, upload, indexing,
chat, memory, and document requests to the FastAPI backend.

You can also index files already placed in `data/uploads/` from the terminal:

```bash
python main.py
```

## OCR Modes

| Mode | Use Case |
|---|---|
| `auto` | OCR only pages with little native PDF text. Best default. |
| `always` | OCR every page/photo and combine it with native text. Slower, useful for scanned manuals and handwritten annotations. |
| `off` | Native document text only. Fastest. |

Handwritten text quality depends on scan clarity and Tesseract's ability to
recognize the handwriting. For critical maintenance work, verify OCR-derived
answers against the page image.

## Interview Demo Checklist

- Add a real equipment manual, SOP, DOCX file, or photographed page through the app menu.
- Use `auto` OCR first; use `always` for scanned manuals.
- Ask a practical question such as:
  - "When should the spindle bearings be lubricated?"
  - "What is the troubleshooting procedure for overheating?"
  - "How do I inspect hydraulic leakage safely?"
- Open "Sources used" under the assistant answer to show file, page, OCR/native
  extraction mode, and retrieved snippet.
- Add equipment context in "Case Memory" to demonstrate multi-turn support.

## Configuration

Environment variables are optional unless noted.

| Variable | Default | Purpose |
|---|---:|---|
| `GEMINI_API_KEY` | Required | Gemini answer generation |
| `DATABASE_URL` | Local SQLite fallback | Neon Postgres connection string for SaaS auth/audit/chat data |
| `USE_LOCAL_DATABASE` | `false` | Force local SQLite even when `DATABASE_URL` exists |
| `SQLITE_BUSY_TIMEOUT_SECONDS` | `60` | Local SQLite wait time when another process briefly holds the DB lock |
| `API_SECRET_KEY` | Dev fallback | Signing key for FastAPI bearer tokens |
| `ACCESS_TOKEN_MAX_AGE_SECONDS` | `28800` | Bearer token lifetime |
| `FASTAPI_BASE_URL` | `http://127.0.0.1:8000` | Backend URL used by the Streamlit frontend |
| `MAX_UPLOAD_FILES` | `10` | Maximum files accepted per upload request |
| `MAX_UPLOAD_MB` | `200` | Maximum size per uploaded file |
| `CORS_ALLOW_ORIGINS` | Localhost dev URLs | Comma-separated browser origins allowed to call the API |
| `PASSWORD_HASH_ITERATIONS` | `260000` | PBKDF2 password hashing cost |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Primary generation model |
| `GEMINI_FALLBACK_MODELS` | `gemini-2.0-flash` | Fallback generation models |
| `FAISS_CANDIDATES` | `30` | Initial vector search candidates |
| `HYBRID_CANDIDATES` | `30` | Combined semantic/BM25 candidates before reranking |
| `RERANK_TOP_K` | `8` | Final source chunks after reranking |
| `GENERAL_GUIDANCE_THRESHOLD` | `-0.25` | Low-confidence threshold for refusing unsupported answers |
| `OCR_MODE` | `auto` | Default OCR behavior |
| `OCR_DPI` | `220` | Render quality for OCR pages |
| `OCR_LANG` | `eng` | Tesseract language code |
| `TESSERACT_CMD` | Empty | Optional explicit path to Tesseract executable |

## Notes For Publishing

Do not commit `.env`, uploaded manuals, generated FAISS indexes, pickle caches,
logs, or local chat history. They are intentionally excluded in `.gitignore`.

## SaaS Data Isolation

Every upload, document record, chunk, FAISS index, chat, and audit event is
scoped by `organization_id`. Document deletion is also tenant-scoped: the API
checks that the file path is inside the signed-in user's tenant upload directory
before deleting anything from disk, then rebuilds only that tenant's index.
