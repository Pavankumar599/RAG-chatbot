import os
import shutil
import tempfile
from pathlib import Path

import httpx
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

load_dotenv()

VECTORSTORE_PATH = "vectorstore_cache"


def _embeddings():
    return OpenAIEmbeddings(model="text-embedding-3-small", http_client=httpx.Client(verify=False))


def load_and_split_pdfs(pdf_files: list) -> tuple[list, dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "],
    )
    all_chunks = []
    stats = {}
    for pdf_file in pdf_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf_file.read())
            tmp_path = tmp.name
        try:
            loader = PyPDFLoader(tmp_path)
            pages = loader.load()
            for page in pages:
                page.metadata["source"] = pdf_file.name
            chunks = splitter.split_documents(pages)
            all_chunks.extend(chunks)
            stats[pdf_file.name] = len(chunks)
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    return all_chunks, stats


def build_vectorstore(chunks: list) -> FAISS:
    return FAISS.from_documents(chunks, _embeddings())


def save_vectorstore(vectorstore: FAISS) -> None:
    vectorstore.save_local(VECTORSTORE_PATH)


def load_vectorstore() -> FAISS:
    return FAISS.load_local(VECTORSTORE_PATH, _embeddings(), allow_dangerous_deserialization=True)


def vectorstore_exists() -> bool:
    return os.path.isdir(VECTORSTORE_PATH)


def delete_vectorstore() -> None:
    if os.path.isdir(VECTORSTORE_PATH):
        shutil.rmtree(VECTORSTORE_PATH)


def build_rag_chain(vectorstore: FAISS):
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        http_client=httpx.Client(verify=False),
        async_client=httpx.AsyncClient(verify=False),
    )
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    condense_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the conversation history and the latest user question, "
         "rewrite the question to be self-contained. Do NOT answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, condense_prompt)

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant that answers questions strictly based on "
         "the provided document context. If the answer is not in the context, "
         "say so clearly.\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    answer_chain = create_stuff_documents_chain(llm, answer_prompt)
    return create_retrieval_chain(history_aware_retriever, answer_chain)