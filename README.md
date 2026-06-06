# RAG-chatbot

A small Streamlit app for Retrieval-Augmented Generation (RAG) over PDFs.

Features
- Upload one or more PDF files and build an embeddings-backed vectorstore.
- Ask questions in a chat UI — answers are generated grounded on the indexed PDFs.
- Persistent local vectorstore cache (`vectorstore_cache`) to avoid re-indexing.

Prerequisites
- Python 3.10+ (3.11 recommended)
- An OpenAI API key with access to the models used by the app.

Install
1. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

Configuration
- Create a `.env` file in the project root with your OpenAI key:

```
OPENAI_API_KEY=sk-...
```

Running the app

Start the Streamlit UI:

```bash
streamlit run app.py
```

Then open the URL shown by Streamlit (usually http://localhost:8501).

How it works
- Upload PDFs via the sidebar. The app:
	- extracts pages with `PyPDFLoader`,
	- splits text into overlapping chunks,
	- embeds chunks with OpenAI embeddings, and
	- stores them in a FAISS vectorstore saved to `vectorstore_cache`.
- The RAG chain rewrites the user's question to be self-contained and queries the vectorstore to produce grounded answers.

Notes & troubleshooting
- Make sure `OPENAI_API_KEY` is set; the app will show an error otherwise.
- The project uses `faiss-cpu`; installing that on some platforms can be tricky. If you hit installation issues, see faiss installation docs or use a compatible Python wheel.
- This repo disables SSL verification for HTTP clients used by the OpenAI/embedding clients (see `app.py` / `rag_pipeline.py`). Only run this code in trusted, local development environments.

Files
- `app.py` — Streamlit application and UI.
- `rag_pipeline.py` — PDF loading, splitting, embeddings, and RAG chain construction.
- `requirements.txt` — Python dependencies.

License
- MIT 

Enjoy — upload some PDFs and start asking questions!
