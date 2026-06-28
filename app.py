from __future__ import annotations

import html
import os
from typing import Any

import requests
import streamlit as st

SUPPORTED_UPLOAD_TYPES = ["pdf", "docx", "txt", "md", "png", "jpg", "jpeg", "tif", "tiff", "bmp", "webp"]
API_BASE_URL = os.getenv("FASTAPI_BASE_URL", "http://127.0.0.1:8000").rstrip("/")


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
    --mc-border: rgba(226, 232, 240, 0.10);
    --mc-border-strong: rgba(226, 232, 240, 0.18);
    --mc-text: #e8edf4;
    --mc-muted: #8a95a6;
    --mc-blue: #2f7df6;
    --mc-teal: #18b6a0;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--mc-bg);
    color: var(--mc-text);
}

[data-testid="stSidebar"], [data-testid="stToolbar"], #MainMenu, footer {
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

.mc-empty p, .small-muted, .menu-state {
    color: var(--mc-muted);
    font-size: 12px;
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

.confidence-pill {
    display: inline-flex;
    align-items: center;
    border: 1px solid var(--mc-border-strong);
    border-radius: 999px;
    padding: 2px 8px;
    margin-left: 6px;
    color: var(--mc-teal);
    background: rgba(24, 182, 160, 0.08);
    font-size: 11px;
    font-weight: 700;
}

.source-text {
    color: var(--mc-muted);
    font-size: 13px;
    line-height: 1.45;
    margin-top: 5px;
}

button[kind="primary"], .stButton > button {
    border-radius: 6px !important;
}

div[data-testid="stChatMessage"] {
    background: transparent;
    padding: 0.35rem 0;
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


class APIClientError(RuntimeError):
    pass


def api_url(path: str) -> str:
    return f"{API_BASE_URL}{path}"


def auth_headers() -> dict[str, str]:
    token = st.session_state.get("access_token")
    return {"Authorization": f"Bearer {token}"} if token else {}


def api_request(
    method: str,
    path: str,
    *,
    auth: bool = True,
    timeout: int = 60,
    **kwargs: Any,
) -> Any:
    headers = kwargs.pop("headers", {})
    if auth:
        headers = {**headers, **auth_headers()}
    try:
        response = requests.request(
            method,
            api_url(path),
            headers=headers,
            timeout=timeout,
            **kwargs,
        )
    except requests.RequestException as exc:
        raise APIClientError(
            f"FastAPI backend is not reachable at {API_BASE_URL}. Start it with `python run_api.py`."
        ) from exc

    if response.status_code == 401:
        st.session_state.pop("access_token", None)
        st.session_state.pop("auth_user", None)
        raise APIClientError("Session expired. Please sign in again.")
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        raise APIClientError(str(detail))
    if response.status_code == 204 or not response.content:
        return None
    return response.json()


def login(email: str, password: str) -> None:
    data = api_request(
        "POST",
        "/api/auth/login",
        auth=False,
        json={"email": email, "password": password},
    )
    st.session_state.access_token = data["access_token"]
    st.session_state.auth_user = data["user"]
    reset_workspace_state()


def register(organization_name: str, full_name: str, email: str, password: str) -> None:
    data = api_request(
        "POST",
        "/api/auth/register",
        auth=False,
        json={
            "organization_name": organization_name,
            "full_name": full_name,
            "email": email,
            "password": password,
        },
    )
    st.session_state.access_token = data["access_token"]
    st.session_state.auth_user = data["user"]
    reset_workspace_state()


def reset_workspace_state() -> None:
    for key in ("conversations", "current_conversation_id", "database_summary"):
        st.session_state.pop(key, None)


def active_user() -> dict[str, str]:
    return st.session_state.auth_user


def refresh_summary() -> dict[str, Any]:
    summary = api_request("GET", "/api/knowledge/summary", timeout=120)
    st.session_state.database_summary = summary
    return summary


def refresh_conversations() -> list[dict[str, Any]]:
    conversations = api_request("GET", "/api/conversations")
    st.session_state.conversations = conversations
    return conversations


def create_conversation(title: str = "New maintenance case") -> dict[str, Any]:
    conversation = api_request("POST", "/api/conversations", json={"title": title})
    refresh_conversations()
    return conversation


def initialize_state() -> None:
    if "response_mode" not in st.session_state:
        st.session_state.response_mode = "Auto"
    if "use_case_memory" not in st.session_state:
        st.session_state.use_case_memory = True
    if "left_panel_open" not in st.session_state:
        st.session_state.left_panel_open = True

    if "conversations" not in st.session_state:
        conversations = refresh_conversations()
        if not conversations:
            conversation = create_conversation()
            conversations = refresh_conversations()
            st.session_state.current_conversation_id = conversation["id"]
        else:
            st.session_state.current_conversation_id = conversations[0]["id"]

    ids = [conversation["id"] for conversation in st.session_state.conversations]
    if "current_conversation_id" not in st.session_state or st.session_state.current_conversation_id not in ids:
        if ids:
            st.session_state.current_conversation_id = ids[0]
        else:
            conversation = create_conversation()
            st.session_state.current_conversation_id = conversation["id"]


def current_conversation() -> dict[str, Any]:
    for conversation in st.session_state.conversations:
        if conversation["id"] == st.session_state.current_conversation_id:
            return conversation
    conversation = create_conversation()
    st.session_state.current_conversation_id = conversation["id"]
    return conversation


def conversation_label(conversation_id: str) -> str:
    for conversation in st.session_state.conversations:
        if conversation["id"] == conversation_id:
            return f"{conversation.get('title', 'Maintenance case')}  ({conversation.get('updated_at', '')})"
    return "Unknown case"


def render_login_screen() -> None:
    st.markdown(
        f"""
<div class="mc-header">
    <div class="mc-kicker">Secure API Client</div>
    <div class="mc-title">MaintenanceCopilot AI</div>
    <div class="mc-subtitle">This Streamlit UI talks only to the FastAPI backend at {html.escape(API_BASE_URL)}.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    try:
        health = api_request("GET", "/api/health", auth=False, timeout=8)
        st.caption(f"Backend: {health.get('status', 'unknown')} | database: {'enabled' if health.get('database_enabled') else 'offline'}")
    except APIClientError as exc:
        st.error(str(exc))

    login_tab, signup_tab = st.tabs(["Sign in", "Create workspace"])
    with login_tab:
        with st.form("login_form"):
            email = st.text_input("Work email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            try:
                login(email, password)
            except APIClientError as exc:
                st.error(str(exc))
            else:
                st.rerun()

    with signup_tab:
        with st.form("signup_form"):
            organization_name = st.text_input("Factory / company name", key="signup_org")
            full_name = st.text_input("Your name", key="signup_name")
            email = st.text_input("Work email", key="signup_email")
            password = st.text_input("Password", type="password", key="signup_password")
            submitted = st.form_submit_button("Create secure workspace", type="primary", use_container_width=True)
        if submitted:
            try:
                register(organization_name, full_name, email, password)
            except APIClientError as exc:
                st.error(str(exc))
            else:
                st.rerun()


def require_login() -> bool:
    if st.session_state.get("access_token") and st.session_state.get("auth_user"):
        return True
    render_login_screen()
    return False


def render_metric_row(summary: dict[str, Any]) -> None:
    api_status = "Ready" if summary.get("api_key_configured") else "Missing key"
    available = "Online" if summary.get("available") else "No index"
    hybrid_status = "Ready" if summary.get("bm25_ready") and summary.get("available") else available
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
        <div class="mc-stat-label">Hybrid Search</div>
        <div class="mc-stat-value">{html.escape(hybrid_status)}</div>
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
            section = html.escape(str(item.get("section", "General")))
            extraction = html.escape(str(item.get("extraction", "native")))
            score = html.escape(str(item.get("score", "")))
            semantic_score = html.escape(str(item.get("semantic_score", "")))
            bm25_score = html.escape(str(item.get("bm25_score", "")))
            confidence = html.escape(str(item.get("confidence", "")))
            snippet = html.escape(str(item.get("snippet", "")))
            st.markdown(
                f"""
<div class="source-box">
    <div class="source-meta">{filename} | page {page} | {section} <span class="confidence-pill">{confidence}%</span></div>
    <div class="small-muted">{extraction} | rerank {score} | semantic {semantic_score} | BM25 {bm25_score}</div>
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


def upload_and_index(uploaded_files: list[Any], ocr_mode: str, force_rebuild: bool) -> None:
    if uploaded_files:
        files = [
            ("files", (uploaded.name, uploaded.getvalue(), uploaded.type or "application/octet-stream"))
            for uploaded in uploaded_files
        ]
        response = api_request(
            "POST",
            "/api/documents/upload",
            files=files,
            params={
                "ocr_mode": ocr_mode,
                "force_rebuild": force_rebuild,
                "index_after_upload": False,
            },
            timeout=300,
        )
        saved_files = ", ".join(response.get("saved_files", []))
        if saved_files:
            st.info(f"Uploaded: {saved_files}. Building searchable index now...")

    response = api_request(
        "POST",
        "/api/knowledge/index",
        json={"ocr_mode": ocr_mode, "force_rebuild": force_rebuild},
        timeout=1800,
    )
    summary = response.get("summary", {})
    st.success(
        f"Index ready with {summary.get('chunks_total', 0)} chunks and "
        f"{summary.get('vectors_total', 0)} vectors."
    )
    refresh_summary()


def run_question(question: str) -> None:
    question = question.strip()
    if not question:
        return
    conversation = current_conversation()
    with st.spinner("FastAPI is searching manuals and preparing a grounded answer..."):
        response = api_request(
            "POST",
            f"/api/conversations/{conversation['id']}/ask",
            json={
                "question": question,
                "response_mode": st.session_state.response_mode,
                "use_case_memory": st.session_state.use_case_memory,
            },
            timeout=300,
        )
    returned = response["conversation"]
    st.session_state.conversations = [
        returned if item["id"] == returned["id"] else item
        for item in st.session_state.conversations
    ]
    refresh_conversations()
    st.rerun()


def render_control_panel(database_summary: dict[str, Any], conversation: dict[str, Any]) -> dict[str, Any]:
    user = active_user()
    st.markdown("### Maintenance Copilot")
    st.caption("Streamlit frontend client")
    st.caption(f"{user['organization_name']} | {user['email']}")
    st.caption(f"API: {API_BASE_URL}")
    if st.button("Sign out", use_container_width=True, key="sign_out"):
        st.session_state.clear()
        st.rerun()

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
    force_rebuild = st.checkbox("Rebuild the full index", value=False, key="force_rebuild_index")
    if st.button("Process manuals", type="primary", use_container_width=True, key="process_manuals"):
        try:
            upload_and_index(uploaded_files or [], ocr_choice, force_rebuild)
        except APIClientError as exc:
            st.error(str(exc))
        else:
            database_summary = st.session_state.database_summary

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

    try:
        document_records = api_request("GET", "/api/documents")
    except APIClientError:
        document_records = []
    if document_records:
        with st.expander("Database document audit", expanded=False):
            for record in document_records[:30]:
                size = record.get("size_bytes") or 0
                name_col, action_col = st.columns([0.72, 0.28], vertical_alignment="center")
                with name_col:
                    st.caption(
                        f"{record['file_name']} | {record['status']} | "
                        f"{size:,} bytes | {record['created_at']}"
                    )
                with action_col:
                    if st.button("Delete", key=f"delete_doc_{record['id']}", use_container_width=True):
                        try:
                            api_request("DELETE", f"/api/documents/{record['id']}", timeout=900)
                            refresh_summary()
                        except APIClientError as exc:
                            st.error(str(exc))
                        else:
                            st.success(f"Deleted {record['file_name']} and rebuilt this tenant's index.")
                            st.rerun()

    st.divider()
    st.markdown("#### History")
    if st.button("New case", use_container_width=True, key="new_case"):
        try:
            new_case = create_conversation()
        except APIClientError as exc:
            st.error(str(exc))
        else:
            st.session_state.current_conversation_id = new_case["id"]
            st.rerun()

    conversation_ids = [item["id"] for item in st.session_state.conversations]
    if conversation_ids:
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
            try:
                api_request("DELETE", f"/api/conversations/{st.session_state.current_conversation_id}")
            except APIClientError as exc:
                st.error(str(exc))
            else:
                refresh_conversations()
                st.session_state.current_conversation_id = st.session_state.conversations[0]["id"]
                st.rerun()

    if st.button("Clear current messages", use_container_width=True, key="clear_current_messages"):
        try:
            api_request("DELETE", f"/api/conversations/{conversation['id']}/messages")
        except APIClientError as exc:
            st.error(str(exc))
        else:
            refresh_conversations()
            st.rerun()

    st.divider()
    st.markdown("#### Response")
    mode_options = ["Auto", "Diagnostic", "Procedural", "Conceptual"]
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
        try:
            updated = api_request(
                "PATCH",
                f"/api/conversations/{conversation['id']}/memory",
                json={"memory_lines": new_memory},
            )
        except APIClientError as exc:
            st.error(str(exc))
        else:
            st.session_state.conversations = [
                updated if item["id"] == updated["id"] else item
                for item in st.session_state.conversations
            ]

    return database_summary


def render_main_content(database_summary: dict[str, Any], conversation: dict[str, Any]) -> None:
    st.markdown(
        """
<div class="mc-header">
    <div class="mc-kicker">Maintenance Copilot</div>
    <div class="mc-title">Industrial knowledge assistant for faster machine troubleshooting</div>
    <div class="mc-subtitle">Streamlit is now a frontend client. Login, uploads, indexing, chat, memory, audit history, retrieval, and AI answers are served by the FastAPI backend.</div>
</div>
""",
        unsafe_allow_html=True,
    )

    render_metric_row(database_summary)

    if not database_summary.get("available"):
        st.info("Open the menu, upload manuals/SOPs/DOCX files/page photos, and process them before asking document-grounded questions.")
    elif not database_summary.get("api_key_configured"):
        st.warning("GEMINI_API_KEY is not configured in the backend environment.")

    render_chat(conversation)

    pending_question = st.session_state.pop("pending_question", None)
    typed_question = st.chat_input("Ask about a fault, procedure, specification, or component")
    active_question = pending_question or typed_question
    if active_question:
        try:
            run_question(active_question)
        except APIClientError as exc:
            st.error(str(exc))


if not require_login():
    st.stop()

try:
    initialize_state()
    database_summary = st.session_state.get("database_summary") or refresh_summary()
except APIClientError as exc:
    st.error(str(exc))
    st.stop()

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
        render_main_content(database_summary, current_conversation())
else:
    render_main_content(database_summary, conversation)
