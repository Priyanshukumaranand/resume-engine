from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from embedder import ResumeEmbedder
from graph import build_add_resume_graph, build_recommendation_graph
from models import (
    RecommendationRequest,
    RecommendationResponse,
    Resume,
    ResumeListResponse,
)
from llm import ResumeLLM
from reranker import RecommendationRanker
from vectorstore import ResumeVectorStore


APP_NAME = "Hackathon Teammate Recommendation API"
PERSIST_DIRECTORY = os.path.join(os.path.dirname(__file__), "chroma_storage")

app = FastAPI(title=APP_NAME, version="1.0.0")
embedder = ResumeEmbedder()
vector_store = ResumeVectorStore(persist_directory=PERSIST_DIRECTORY)
resume_llm = ResumeLLM()
ranker = RecommendationRanker()
add_resume_graph = build_add_resume_graph(embedder, vector_store)
recommendation_graph = build_recommendation_graph(embedder, vector_store, ranker)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/add_resume")
async def add_resume(resume: Resume) -> dict:
    enriched = _ensure_resume_fields(resume)
    add_resume_graph.invoke({"resume": enriched})
    return {"status": "created", "id": enriched.id}


@app.get("/resumes", response_model=ResumeListResponse)
async def list_resumes() -> ResumeListResponse:
    resumes = vector_store.get_all_resumes()
    return ResumeListResponse(resumes=resumes)


@app.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: str) -> dict:
    deleted = vector_store.delete(resume_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Resume not found")
    return {"status": "deleted", "id": resume_id}


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend_teammate(request: RecommendationRequest) -> RecommendationResponse:
    if not vector_store.has_resumes():
        raise HTTPException(status_code=404, detail="No resumes stored")

    state = recommendation_graph.invoke({"request": request})
    ranked = state.get("ranked", [])
    if not ranked:
        raise HTTPException(status_code=404, detail="No matching candidates")
    best = ranked[0]
    response = RecommendationResponse(
        best_match=best,
        considered=len(ranked),
    )
    return response


def _ensure_resume_fields(resume: Resume) -> Resume:
    needs_role = not (resume.role and resume.role.strip())
    needs_skills = len(resume.skills) == 0

    if not needs_role and not needs_skills:
        return resume

    inference_text = "\n".join(part for part in [resume.experience, resume.summary] if part)
    inferred_role, inferred_skills = resume_llm.infer_role_and_skills(inference_text)

    update: dict = {}
    if needs_role:
        update["role"] = inferred_role or "Hackathon Contributor"
    if needs_skills:
        update["skills"] = _deduplicate_skills(inferred_skills) or ["teamwork"]

    enriched = resume.model_copy(update=update)
    if not enriched.role:
        enriched = enriched.model_copy(update={"role": "Hackathon Contributor"})
    if not enriched.skills:
        enriched = enriched.model_copy(update={"skills": ["teamwork"]})
    return enriched


def _deduplicate_skills(skills: list[str]) -> list[str]:
    seen = set()
    cleaned = []
    for skill in skills or []:
        text = skill.strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(text)
    return cleaned[:10]