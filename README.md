# Maintenance Copilot

Maintenance Copilot is an evidence-grounded RAG assistant for industrial maintenance teams. It lets operators and engineers upload manuals, SOPs, service notes, DOCX files, and page images, then ask practical troubleshooting questions with source-backed answers.

The project combines a FastAPI backend for multi-tenant SaaS workflows with a Streamlit frontend for demos and local pilots.

## Highlights

- Email/password authentication with signed bearer tokens
- Organization-scoped uploads, indexes, conversations, memory, and audit records
- Direct uploads for PDF, DOCX, TXT, Markdown, PNG, JPG, TIFF, BMP, and WebP
- Native PDF/DOCX extraction with table-aware chunking
- OCR fallback for scanned PDFs, embedded document images, and photographed pages
- FAISS semantic search combined with BM25 keyword retrieval and reranking
- Gemini answer generation with visible evidence cards and source references
- Persistent case history, uploaded-file audit logs, and case memory notes
- Tenant-scoped document deletion with index rebuild
- Local SQLite fallback plus optional production Postgres via `DATABASE_URL`

## Architecture

```text
maintenance-copilot/
|-- app.py                  Streamlit frontend client
|-- api/                    FastAPI API, schemas, and auth security
|-- core/                   Tenant resolution and shared exceptions
|-- database/               SQLAlchemy models, schema, seed, and session setup
|-- repositories/           Data access for documents, vectors, and organizations
|-- services/               Auth, audit chat, indexing, and retrieval services
|-- main.py                 Extraction, OCR, chunking, and FAISS indexing
|-- query.py                Retrieval, prompt construction, and Gemini calls
|-- tests/                  Backend, storage, routing, and frontend boundary tests
|-- docs/                   Architecture and data-layout notes
`-- data/                   Local runtime data, ignored except placeholder files
```

## Quick Start

Install Python dependencies:

```bash
pip install -r requirements.txt
```

Create your local environment file:

```bash
copy .env.example .env
```

Set at least:

```text
GEMINI_API_KEY=your_key_here
API_SECRET_KEY=a_long_random_secret_for_api_tokens
FASTAPI_BASE_URL=http://127.0.0.1:8000
```

For production-style SaaS storage, also set:

```text
DATABASE_URL=postgresql+psycopg://user:password@host/dbname?sslmode=require
```

Run the FastAPI backend:

```bash
python run_api.py
```

Open the API docs:

```text
http://127.0.0.1:8000/docs
```

Run the Streamlit frontend in a second terminal:

```bash
streamlit run app.py
```

## OCR Setup

For scanned manuals and photographed pages, install the native Tesseract OCR executable. On Windows, either add it to `PATH` or set:

```text
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
```

The Python package `pytesseract` is included in `requirements.txt`, but the native Tesseract program is still required for OCR.

## Testing

Run the test suite with:

```bash
python -m pytest
```

Fast checks used during development:

```bash
python -m py_compile app.py main.py query.py
python -m pytest tests
```

## Publishing Notes

This repository is configured to exclude secrets, uploaded manuals, generated FAISS indexes, local SQLite databases, logs, caches, and installed dependencies. Keep `.env` local and publish `.env.example` instead.

For a deeper product and architecture walkthrough, see [docs/README.md](docs/README.md).
