# Hackathon Teammate Recommendation System

FastAPI backend plus Streamlit demo that stores and retrieves hackathon resumes with ChromaDB vectors and Sentence Transformers embeddings.

## Features
- FastAPI endpoints for health check, resume ingest, listing, and teammate recommendations.
- ChromaDB persistent vector store (`chroma_storage/`) that keeps embeddings + metadata locally.
- Sentence Transformers (`all-MiniLM-L6-v2`) for compact semantic embeddings.
- Streamlit UI for quickly uploading resumes and testing recommendations without the future React app.
- LangGraph pipelines orchestrate both resume ingestion (document→embedding→Chroma) and recommendations (query build→embed→vector search→rerank) for clearer, testable flow.
- Flan-T5–powered helper infers missing preferred role and key skills directly from the uploaded resume, so contributors can skip those fields.
- Cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) rescoring the top Chroma hits for better semantic accuracy.

## Setup
1. Create and activate a Python 3.10+ virtual environment.
2. Install dependencies:
	```bash
	pip install -r requirements.txt
	```

## Running the FastAPI server
```bash
uvicorn main:app --reload
```
- The API listens on `http://127.0.0.1:8000` by default.
- `POST /add_resume` accepts the full resume JSON payload.
- `GET /resumes` lists everything currently in the `resumes` collection.
- `DELETE /resumes/{resume_id}` removes a stored resume (Streamlit exposes a delete button per card).
- `POST /recommend` runs semantic search + reranking and returns the best candidate, score, matching skills, and explanation.

## Streamlit demo app
```bash
streamlit run streamlit_app.py
```
- Uses the same FastAPI URL (override with `API_BASE_URL`).
- Upload a PDF resume; the app auto-extracts text for embedding and will infer preferred role/skills if you leave them blank.
- Provides forms for uploading resumes and requesting recommendations.

## How ChromaDB fits in
1. The API initializes a persistent ChromaDB client pointing at `./chroma_storage` and a collection named `resumes`.
2. When `/add_resume` is called, we combine role, skills, experience, and summary into one document, embed it, and `upsert` it alongside resume metadata.
3. `/recommend` embeds the requester’s role/skills/summary, runs a similarity query inside Chroma (cosine distance), and receives the top-K nearest vectors.
4. We rerank those matches with additional skill/role heuristics plus a cross-encoder semantic score before responding with the single best teammate suggestion and rationale.

By persisting embeddings + metadata locally, restarting the server or Streamlit app preserves the stored resumes without any external databases.
# mcp-resume-engine
