import ssl
import urllib3
import requests as _req

ssl._create_default_https_context = ssl._create_unverified_context
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

_orig = _req.Session.request
def _no_ssl(self, *args, **kw):
    kw.setdefault("verify", False)
    return _orig(self, *args, **kw)
_req.Session.request = _no_ssl
del _no_ssl, _req

import os
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

from rag_pipeline import (
    load_and_split_pdfs, build_vectorstore, build_rag_chain,
    save_vectorstore, load_vectorstore, vectorstore_exists, delete_vectorstore,
)

load_dotenv()

st.set_page_config(page_title="PDF RAG Chat", page_icon="📄", layout="wide")
st.title("📄 Chat with your PDFs")
st.caption("Upload PDFs, then ask questions about their content.")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "processed_files" not in st.session_state:
    st.session_state.processed_files = {}  # {filename: chunk_count}

# ── Auto-load persisted vectorstore on startup ────────────────────────────────
if st.session_state.rag_chain is None and vectorstore_exists():
    try:
        vs = load_vectorstore()
        st.session_state.rag_chain = build_rag_chain(vs)
    except Exception:
        delete_vectorstore()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("1. Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button("Process PDFs", type="primary", use_container_width=True):
            if not os.getenv("OPENAI_API_KEY"):
                st.error("OPENAI_API_KEY is not set. Add it to a .env file.")
            else:
                with st.spinner("Loading and embedding PDFs..."):
                    chunks, stats = load_and_split_pdfs(uploaded_files)
                    vectorstore = build_vectorstore(chunks)
                    save_vectorstore(vectorstore)
                    st.session_state.rag_chain = build_rag_chain(vectorstore)
                    st.session_state.chat_history = []
                    st.session_state.processed_files = stats
                total = sum(stats.values())
                st.success(f"Ready! Indexed {total} chunks from {len(stats)} file(s).")

    if st.session_state.processed_files:
        st.divider()
        st.markdown("**Indexed files:**")
        for name, count in st.session_state.processed_files.items():
            st.markdown(f"- **{name}** — {count} chunks")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    with col2:
        if st.button("Clear PDFs", use_container_width=True):
            delete_vectorstore()
            st.session_state.rag_chain = None
            st.session_state.chat_history = []
            st.session_state.processed_files = {}
            st.rerun()

# ── Chat area ─────────────────────────────────────────────────────────────────
if not st.session_state.rag_chain:
    st.info("Upload and process one or more PDFs using the sidebar to get started.")
else:
    for msg in st.session_state.chat_history:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    if question := st.chat_input("Ask a question about your documents..."):
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history,
                })
            answer = result["answer"]
            st.markdown(answer)

            source_docs = result.get("context", [])
            if source_docs:
                with st.expander("📚 Sources used"):
                    seen = set()
                    for doc in source_docs:
                        src = doc.metadata.get("source", "unknown")
                        page = doc.metadata.get("page", "?")
                        key = (src, page)
                        if key not in seen:
                            seen.add(key)
                            st.markdown(f"**{src}** — page {page}")
                            st.caption(doc.page_content[:300] + "...")

        st.session_state.chat_history.append(HumanMessage(content=question))
        st.session_state.chat_history.append(AIMessage(content=answer))