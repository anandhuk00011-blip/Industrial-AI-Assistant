from __future__ import annotations

import html
import json
import re
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
HISTORY_PATH = BASE_DIR / "chat_history.json"
SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]

try:
    from main import INPUT_FOLDER, get_ocr_status, index_pdfs

    INDEXER_ERROR = None
except Exception as exc:  # pragma: no cover - shown in UI
    INPUT_FOLDER = BASE_DIR / "data_input"
    INDEXER_ERROR = str(exc)

    def get_ocr_status() -> dict[str, Any]:
        return {"ready": False, "error": INDEXER_ERROR}

    def index_pdfs(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError(INDEXER_ERROR)

try:
    from query import ask_copilot, get_database_summary, reload_vector_database

    QUERY_ERROR = None
except Exception as exc:  # pragma: no cover - shown in UI
    QUERY_ERROR = str(exc)

    def ask_copilot(*_args: Any, **_kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        return f"Query engine failed to start: {QUERY_ERROR}", []

    def reload_vector_database(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        return {
            "available": False,
            "chunks": 0,
            "vectors": 0,
            "files": [],
            "file_count": 0,
            "extraction_counts": {},
            "api_key_configured": False,
            "generation_models": [],
        }

    def get_database_summary() -> dict[str, Any]:
        return reload_vector_database()


st.set_page_config(
    page_title="Maintenance Copilot",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
<style>
:root {
    --mc-bg: #0f1217;
    --mc-panel: #151a21;
    --mc-panel-2: #1b222b;
    --mc-border: rgba(226, 232, 240, 0.10);
    --mc-border-strong: rgba(226, 232, 240, 0.18);
    --mc-text: #e8edf4;
    --mc-muted: #8a95a6;
    --mc-blue: #2f7df6;
    --mc-teal: #18b6a0;
    --mc-amber: #d99a22;
    --mc-red: #e05252;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--mc-bg);
    color: var(--mc-text);
}

[data-testid="stSidebar"] {
    display: none;
}

[data-testid="stToolbar"], #MainMenu, footer {
    display: none;
}

[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="collapsedControl"] {
    display: none !important;
}

.block-container {
    padding-top: 1.25rem;
    padding-bottom: 2rem;
    max-width: 1100px;
}

.mc-header {
    border: 1px solid var(--mc-border);
    background: linear-gradient(135deg, #151a21 0%, #101923 100%);
    border-radius: 8px;
    padding: 18px 20px;
    margin-bottom: 18px;
}

.mc-kicker {
    color: var(--mc-teal);
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}

.mc-title {
    color: var(--mc-text);
    font-size: 28px;
    font-weight: 750;
    line-height: 1.2;
    margin-top: 4px;
}

.mc-subtitle {
    color: var(--mc-muted);
    font-size: 14px;
    line-height: 1.5;
    margin-top: 6px;
}

.mc-status-row {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 10px;
    margin-bottom: 16px;
}

.mc-stat {
    background: var(--mc-panel);
    border: 1px solid var(--mc-border);
    border-radius: 8px;
    padding: 12px 14px;
}

.mc-stat-label {
    color: var(--mc-muted);
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}

.mc-stat-value {
    color: var(--mc-text);
    font-size: 20px;
    font-weight: 700;
    margin-top: 3px;
}

.mc-empty {
    border: 1px dashed var(--mc-border-strong);
    background: rgba(21, 26, 33, 0.55);
    border-radius: 8px;
    padding: 24px;
    margin: 8px 0 16px;
}

.mc-empty h3 {
    margin: 0 0 8px;
    color: var(--mc-text);
    font-size: 18px;
}

.mc-empty p {
    margin: 0;
    color: var(--mc-muted);
    font-size: 14px;
    line-height: 1.5;
}

.source-box {
    border-left: 3px solid var(--mc-blue);
    background: rgba(47, 125, 246, 0.06);
    padding: 10px 12px;
    border-radius: 6px;
    margin: 8px 0;
}

.source-meta {
    color: var(--mc-text);
    font-size: 13px;
    font-weight: 650;
}

.source-text {
    color: var(--mc-muted);
    font-size: 13px;
    line-height: 1.45;
    margin-top: 5px;
}

.small-muted {
    color: var(--mc-muted);
    font-size: 12px;
}

.menu-state {
    color: var(--mc-muted);
    font-size: 12px;
    margin: -4px 0 12px;
}

button[kind="primary"], .stButton > button {
    border-radius: 6px !important;
}

div[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.35rem 0;
}

div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-user"] {
    background: var(--mc-blue);
}

div[data-testid="stChatMessage"] [data-testid="chatAvatarIcon-assistant"] {
    background: var(--mc-teal);
}

@media (max-width: 760px) {
    .mc-status-row {
        grid-template-columns: repeat(2, minmax(0, 1fr));
    }
    .mc-title {
        font-size: 23px;
    }
}
</style>
""",
    unsafe_allow_html=True,
)


def now_label() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M")


def safe_filename(filename: str) -> str:
    filename = Path(filename).name
    filename = re.sub(r"[^A-Za-z0-9._ -]", "_", filename).strip(" .")
    return filename or f"document-{uuid.uuid4().hex[:8]}"


def load_history() -> list[dict[str, Any]]:
    if not HISTORY_PATH.exists():
        return []
    try:
        data = json.loads(HISTORY_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and item.get("id")]


def save_history(conversations: list[dict[str, Any]]) -> None:
    HISTORY_PATH.write_text(
        json.dumps(conversations, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def create_conversation(title: str = "New maintenance case") -> dict[str, Any]:
    timestamp = now_label()
    return {
        "id": uuid.uuid4().hex,
        "title": title,
        "created_at": timestamp,
        "updated_at": timestamp,
        "messages": [],
        "memory": [],
    }


def initialize_state() -> None:
    if "conversations" not in st.session_state:
        st.session_state.conversations = load_history()
        if not st.session_state.conversations:
            st.session_state.conversations = [create_conversation()]
            save_history(st.session_state.conversations)

    ids = [conversation["id"] for conversation in st.session_state.conversations]
    if "current_conversation_id" not in st.session_state or st.session_state.current_conversation_id not in ids:
        st.session_state.current_conversation_id = ids[0]

    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Auto"
    if "use_case_memory" not in st.session_state:
        st.session_state.use_case_memory = True
    if "left_panel_open" not in st.session_state:
        st.session_state.left_panel_open = True


def current_conversation() -> dict[str, Any]:
    for conversation in st.session_state.conversations:
        if conversation["id"] == st.session_state.current_conversation_id:
            return conversation
    conversation = create_conversation()
    st.session_state.conversations.insert(0, conversation)
    st.session_state.current_conversation_id = conversation["id"]
    return conversation


def title_from_question(question: str) -> str:
    cleaned = re.sub(r"\s+", " ", question).strip()
    if len(cleaned) > 58:
        cleaned = cleaned[:55].rstrip() + "..."
    return cleaned or "Maintenance case"


def conversation_label(conversation_id: str) -> str:
    for conversation in st.session_state.conversations:
        if conversation["id"] == conversation_id:
            title = conversation.get("title", "Maintenance case")
            updated = conversation.get("updated_at", "")
            return f"{title}  ({updated})"
    return "Unknown case"


def render_metric_row(summary: dict[str, Any]) -> None:
    api_status = "Ready" if summary.get("api_key_configured") else "Missing key"
    available = "Online" if summary.get("available") else "No index"
    st.markdown(
        f"""
<div class="mc-status-row">
    <div class="mc-stat">
        <div class="mc-stat-label">Manuals</div>
        <div class="mc-stat-value">{summary.get("file_count", 0)}</div>
    </div>
    <div class="mc-stat">
        <div class="mc-stat-label">Search Chunks</div>
        <div class="mc-stat-value">{summary.get("chunks", 0)}</div>
    </div>
    <div class="mc-stat">
        <div class="mc-stat-label">Vector Index</div>
        <div class="mc-stat-value">{html.escape(available)}</div>
    </div>
    <div class="mc-stat">
        <div class="mc-stat-label">Gemini</div>
        <div class="mc-stat-value">{html.escape(api_status)}</div>
    </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_sources(evidence: list[dict[str, Any]]) -> None:
    if not evidence:
        return

    with st.expander("Sources used", expanded=False):
        for item in evidence:
            filename = html.escape(str(item.get("file", "Unknown")))
            page = html.escape(str(item.get("page", "?")))
            extraction = html.escape(str(item.get("extraction", "native")))
            score = html.escape(str(item.get("score", "")))
            snippet = html.escape(str(item.get("snippet", "")))
            st.markdown(
                f"""
<div class="source-box">
    <div class="source-meta">{filename} | page {page} | {extraction} | score {score}</div>
    <div class="source-text">{snippet}</div>
</div>
""",
                unsafe_allow_html=True,
            )


def render_chat(conversation: dict[str, Any]) -> None:
    messages = conversation.get("messages", [])
    if not messages:
        st.markdown(
            """
<div class="mc-empty">
    <h3>Start a maintenance case</h3>
    <p>Ask about a fault, service interval, procedure, component, alarm, or specification from the indexed manuals.</p>
</div>
""",
            unsafe_allow_html=True,
        )
        sample_cols = st.columns(3)
        sample_questions = [
            "When should the spindle bearings be lubricated?",
            "What is the troubleshooting procedure for overheating?",
            "How do I safely inspect the hydraulic leakage point?",
        ]
        for col, question in zip(sample_cols, sample_questions):
            if col.button(question, use_container_width=True):
                st.session_state.pending_question = question
                st.rerun()

    for message in messages:
        role = "user" if message.get("role") == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(message.get("content", ""))
            if role == "assistant":
                render_sources(message.get("evidence", []))


def save_uploaded_files(uploaded_files: list[Any]) -> list[str]:
    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    saved = []
    for uploaded in uploaded_files:
        filename = safe_filename(uploaded.name)
        target = INPUT_FOLDER / filename
        target.write_bytes(uploaded.getbuffer())
        saved.append(filename)
    return saved


def process_indexing(uploaded_files: list[Any], ocr_mode: str, force_rebuild: bool) -> None:
    if INDEXER_ERROR:
        st.error(INDEXER_ERROR)
        return

    saved_files = save_uploaded_files(uploaded_files) if uploaded_files else []
    status_slot = st.empty()
    progress = st.progress(0)
    tick = {"value": 0}

    def progress_callback(event: dict[str, Any]) -> None:
        message = event.get("message", "Working...")
        status_slot.info(message)
        if event.get("event") != "done":
            tick["value"] = min(tick["value"] + 8, 92)
            progress.progress(tick["value"])
        else:
            progress.progress(100)

    try:
        summary = index_pdfs(
            input_folder=INPUT_FOLDER,
            ocr_mode=ocr_mode,
            force_rebuild=force_rebuild,
            progress_callback=progress_callback,
        )
        reload_vector_database(force=True)
    except Exception as exc:
        progress.empty()
        status_slot.error(f"Indexing failed: {exc}")
        return

    progress.empty()
    saved_text = f" Saved: {', '.join(saved_files)}." if saved_files else ""
    status_slot.success(
        f"Index ready with {summary.get('chunks_total', 0)} chunks and "
        f"{summary.get('vectors_total', 0)} vectors.{saved_text}"
    )
    warnings = summary.get("warnings", [])
    if warnings:
        with st.expander("Indexing warnings", expanded=False):
            for warning in warnings[:20]:
                st.warning(warning)


def mode_code(label: str) -> str | None:
    return {
        "Diagnostic": "MODE A",
        "Procedural": "MODE B",
        "Conceptual": "MODE C",
    }.get(label)


def run_question(question: str) -> None:
    conversation = current_conversation()
    question = question.strip()
    if not question:
        return

    history_before = conversation.get("messages", [])[-8:] if st.session_state.use_case_memory else []
    memory = conversation.get("memory", []) if st.session_state.use_case_memory else []

    if not conversation.get("messages"):
        conversation["title"] = title_from_question(question)

    conversation.setdefault("messages", []).append(
        {
            "role": "user",
            "content": question,
            "time": now_label(),
        }
    )
    conversation["updated_at"] = now_label()
    save_history(st.session_state.conversations)

    with st.spinner("Searching manuals and preparing a grounded answer..."):
        answer, evidence = ask_copilot(
            question,
            conversation_history=history_before,
            user_memory=memory,
            force_mode=mode_code(st.session_state.response_mode),
        )

    conversation.setdefault("messages", []).append(
        {
            "role": "assistant",
            "content": answer,
            "evidence": evidence,
            "time": now_label(),
        }
    )
    conversation["updated_at"] = now_label()
    save_history(st.session_state.conversations)
    st.rerun()


def render_control_panel(
    database_summary: dict[str, Any],
    conversation: dict[str, Any],
) -> dict[str, Any]:
    st.markdown("### Maintenance Copilot")
    st.caption("Upload, index, history, and case memory")

    if INDEXER_ERROR:
        st.error(f"Indexer error: {INDEXER_ERROR}")
    if QUERY_ERROR:
        st.error(f"Query error: {QUERY_ERROR}")

    st.divider()
    st.markdown("#### Knowledge Upload")
    uploaded_files = st.file_uploader(
        "Upload manuals, SOPs, DOCX files, or page photos",
        type=SUPPORTED_UPLOAD_TYPES,
        accept_multiple_files=True,
        key="manual_uploads",
    )
    ocr_choice = st.selectbox(
        "OCR mode",
        ["auto", "always", "off"],
        index=0,
        help="Auto OCR scans pages/photos with little native text. Always OCR is slower but stronger for scanned manuals and handwritten notes.",
        key="ocr_mode_choice",
    )
    force_rebuild = st.checkbox(
        "Rebuild the full index",
        value=False,
        key="force_rebuild_index",
    )
    if st.button("Process manuals", type="primary", use_container_width=True, key="process_manuals"):
        process_indexing(uploaded_files or [], ocr_choice, force_rebuild)
        database_summary = reload_vector_database(force=True)

    ocr_status = get_ocr_status()
    if ocr_status.get("ready"):
        st.caption("OCR engine: ready")
    else:
        st.caption("OCR engine: not detected")
    st.caption("Supported: PDF, DOCX, TXT, Markdown, PNG, JPG, TIFF, BMP, WebP")

    st.divider()
    st.markdown("#### Indexed Sources")
    st.metric("Manuals", database_summary.get("file_count", 0))
    st.metric("Chunks", database_summary.get("chunks", 0))
    st.metric("OCR pages/photos", database_summary.get("ocr_pages", 0))
    if database_summary.get("document_types"):
        type_summary = ", ".join(
            f"{ext or 'file'}: {count}"
            for ext, count in sorted(database_summary.get("document_types", {}).items())
        )
        st.caption(type_summary)
    if database_summary.get("files"):
        with st.expander("Manual list", expanded=False):
            for filename in database_summary.get("files", []):
                st.write(filename)

    st.divider()
    st.markdown("#### History")
    if st.button("New case", use_container_width=True, key="new_case"):
        new_case = create_conversation()
        st.session_state.conversations.insert(0, new_case)
        st.session_state.current_conversation_id = new_case["id"]
        save_history(st.session_state.conversations)
        st.rerun()

    conversation_ids = [item["id"] for item in st.session_state.conversations]
    current_index = conversation_ids.index(st.session_state.current_conversation_id)
    selected_id = st.selectbox(
        "Open case",
        conversation_ids,
        index=current_index,
        format_func=conversation_label,
        key="open_case",
    )
    if selected_id != st.session_state.current_conversation_id:
        st.session_state.current_conversation_id = selected_id
        st.rerun()

    if len(st.session_state.conversations) > 1:
        if st.button("Delete current case", use_container_width=True, key="delete_current_case"):
            st.session_state.conversations = [
                item for item in st.session_state.conversations
                if item["id"] != st.session_state.current_conversation_id
            ]
            st.session_state.current_conversation_id = st.session_state.conversations[0]["id"]
            save_history(st.session_state.conversations)
            st.rerun()

    if st.button("Clear current messages", use_container_width=True, key="clear_current_messages"):
        conversation["messages"] = []
        conversation["updated_at"] = now_label()
        save_history(st.session_state.conversations)
        st.rerun()

    st.divider()
    st.markdown("#### Response")
    mode_options = ["Auto", "Diagnostic", "Procedural", "Conceptual"]
    if st.session_state.response_mode not in mode_options:
        st.session_state.response_mode = "Auto"
    st.session_state.response_mode = st.selectbox(
        "Mode",
        mode_options,
        index=mode_options.index(st.session_state.response_mode),
        key="response_mode_select",
    )
    st.session_state.use_case_memory = st.checkbox(
        "Use case memory",
        value=st.session_state.use_case_memory,
        key="use_case_memory_toggle",
    )

    st.markdown("#### Case Memory")
    memory_text = st.text_area(
        "Equipment, symptoms, readings, constraints",
        value="\n".join(conversation.get("memory", [])),
        height=120,
        placeholder="Example: CNC VMC-850, spindle overheating after 35 min, bearing noise at high RPM, coolant pump recently replaced.",
        key=f"case_memory_{conversation['id']}",
    )
    new_memory = [line.strip() for line in memory_text.splitlines() if line.strip()]
    if new_memory != conversation.get("memory", []):
        conversation["memory"] = new_memory
        conversation["updated_at"] = now_label()
        save_history(st.session_state.conversations)

    return database_summary


def render_main_content(database_summary: dict[str, Any], conversation: dict[str, Any]) -> None:
    st.markdown(
        """
<div class="mc-header">
    <div class="mc-kicker">Maintenance Copilot</div>
    <div class="mc-title">Industrial knowledge assistant for faster machine troubleshooting</div>
    <div class="mc-subtitle">Upload manuals, SOPs, Word files, and photos of pages. The system extracts text, OCRs scanned content, remembers the case, and answers like a practical maintenance engineer with source references.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    render_metric_row(database_summary)

    if not database_summary.get("available"):
        st.info("Open the menu, upload manuals/SOPs/DOCX files/page photos, and process them before asking document-grounded questions.")
    elif not database_summary.get("api_key_configured"):
        st.warning("GEMINI_API_KEY is not configured in .env. Retrieval will work after indexing, but answer generation needs the key.")

    render_chat(conversation)

    pending_question = st.session_state.pop("pending_question", None)
    typed_question = st.chat_input("Ask about a fault, procedure, specification, or component")
    active_question = pending_question or typed_question
    if active_question:
        run_question(active_question)


initialize_state()

try:
    database_summary = reload_vector_database()
except Exception as exc:
    database_summary = get_database_summary()
    st.error(f"Could not load vector database: {exc}")

conversation = current_conversation()


toggle_label = "Hide menu" if st.session_state.left_panel_open else "Show menu"
if st.button(toggle_label, key="left_panel_toggle"):
    st.session_state.left_panel_open = not st.session_state.left_panel_open
    st.rerun()

menu_state = "Menu open" if st.session_state.left_panel_open else "Menu hidden"
st.markdown(f'<div class="menu-state">{menu_state}</div>', unsafe_allow_html=True)

if st.session_state.left_panel_open:
    panel_col, main_col = st.columns([0.32, 0.68], gap="large")
    with panel_col:
        with st.container(border=True):
            database_summary = render_control_panel(database_summary, conversation)
    with main_col:
        render_main_content(database_summary, conversation)
else:
    render_main_content(database_summary, conversation)
