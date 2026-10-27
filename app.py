"""Psychometric Domain Mapping — Streamlit application.

Dataset lookup and ranking work without a Groq API key. Explanations and
job-description analysis are optional and never overwrite stored scores.
"""

from __future__ import annotations

import streamlit as st

from src.chat_service import reply_to_chat
from src.config import get_config
from src.data_loader import DataLoadError, DatasetLoader
from src.groq_router import GroqRouter

st.set_page_config(
    page_title="Psychometric Domain Mapping",
    page_icon="PM",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-family: "Source Sans 3", "Segoe UI", sans-serif; }
    .block-title { font-size: 1.7rem; font-weight: 700; color: #1e293b; margin-bottom: 0.15rem; }
    .block-sub { color: #475569; margin-bottom: 0.8rem; }
    .domain-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1d4ed8;
        border-radius: 10px;
        padding: 0.9rem 1rem;
        margin-bottom: 0.65rem;
        color: #0f172a;
    }
    .score-pill {
        display: inline-block;
        min-width: 2.1rem;
        text-align: center;
        color: white;
        font-weight: 700;
        border-radius: 8px;
        padding: 0.15rem 0.45rem;
    }
    .tag {
        display: inline-block;
        margin: 0.15rem 0.25rem 0.15rem 0;
        padding: 0.12rem 0.5rem;
        border-radius: 999px;
        font-size: 0.78rem;
        background: #e2e8f0;
        color: #1e293b;
    }
    .note-amber {
        background: #fffbeb;
        border: 1px solid #fcd34d;
        color: #78350f;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .note-blue {
        background: #eff6ff;
        border: 1px solid #bfdbfe;
        color: #1e3a8a;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    .note-red {
        background: #fef2f2;
        border: 1px solid #fecaca;
        color: #7f1d1d;
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
        margin: 0.5rem 0 0.8rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_loader() -> DatasetLoader:
    loader = DatasetLoader(config=get_config())
    loader.load()
    return loader


@st.cache_resource
def get_router() -> GroqRouter:
    return GroqRouter(config=get_config().groq)


def _opening_message(roles: list[str]) -> str:
    """First chat message: the roles a person can enter."""
    listed = "\n".join(f"{index}. {role}" for index, role in enumerate(roles, start=1))
    return (
        f"You can enter these {len(roles)} roles:\n\n"
        f"{listed}\n\n"
        "Type one of them. I will list the stored domains that fit, "
        "what each domain means, and why it fits that role."
    )


def _render_chat() -> None:
    """Type a message. The reply reads like a chat and keeps stored scores fixed."""
    st.markdown(
        """
        <style>
        .block-container { max-width: 760px; padding-top: 1.2rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="block-title">Chat</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="block-sub">Type a role. Each domain includes what it means, why it fits, and its score.</div>',
        unsafe_allow_html=True,
    )
    roles = get_loader().get_roles()
    opening = _opening_message(roles)
    if "chat_messages" not in st.session_state or not st.session_state.chat_messages:
        st.session_state.chat_messages = [{"role": "assistant", "content": opening}]
    elif st.session_state.chat_messages[0].get("role") == "assistant":
        st.session_state.chat_messages[0]["content"] = opening
    if "chat_role" not in st.session_state:
        st.session_state.chat_role = None

    for message in st.session_state.chat_messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for warning in message.get("warnings", []):
                st.caption(warning)

    text = st.chat_input("Type a role")
    if not text:
        return

    st.session_state.chat_messages.append({"role": "user", "content": text})
    with st.chat_message("user"):
        st.markdown(text)
    history = st.session_state.chat_messages[:-1]
    with st.chat_message("assistant"):
        with st.spinner("Thinking…"):
            try:
                answer, context, warnings = reply_to_chat(
                    text,
                    get_loader(),
                    get_router(),
                    history,
                    previous_role=st.session_state.chat_role,
                )
            except Exception:
                answer = "That role could not be answered just now. Type it again."
                context = None
                warnings = []
        st.markdown(answer)
        for warning in warnings:
            st.caption(warning)
    if context is not None and context.matched_role:
        st.session_state.chat_role = context.matched_role
    st.session_state.chat_messages.append(
        {"role": "assistant", "content": answer, "warnings": warnings}
    )


def main() -> None:
    with st.sidebar:
        st.markdown("**Psychometric domain mapping**")
        router = get_router()
        if router.config.is_configured:
            st.success("Groq key is set")
        else:
            st.info("Groq key not set. The chat still lists stored domains.")
        try:
            info = get_loader().get_dataset_info()
            st.metric("Roles", info.num_roles)
            st.metric("Domains", info.num_domains)
            st.metric("Mappings", info.num_mappings)
        except DataLoadError as exc:
            st.error(str(exc))
        st.caption("Scores are provisional role-level estimates.")
        if st.button("Clear chat"):
            st.session_state.pop("chat_messages", None)
            st.session_state.chat_role = None
            st.rerun()

    _render_chat()


main()
