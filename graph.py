from __future__ import annotations

from typing import List, Sequence, Tuple, TypedDict

from langgraph.graph import END, StateGraph

from embedder import ResumeEmbedder
from models import RecommendationRequest, RecommendationResult, Resume
from reranker import RecommendationRanker
from vectorstore import ResumeVectorStore


class AddResumeState(TypedDict, total=False):
    resume: Resume
    document: str
    embedding: List[float]


class RecommendState(TypedDict, total=False):
    request: RecommendationRequest
    query_text: str
    query_embedding: List[float]
    candidates: List[Tuple[Resume, float]]
    ranked: List[RecommendationResult]


def build_add_resume_graph(embedder: ResumeEmbedder, store: ResumeVectorStore):
    graph = StateGraph(AddResumeState)

    def build_document(state: AddResumeState):
        document = store.build_document(state["resume"])
        return {"document": document}

    def embed(state: AddResumeState):
        embedding = embedder.embed_text(state["document"])
        return {"embedding": embedding}

    def persist(state: AddResumeState):
        store.add_resume(state["resume"], state["embedding"], state["document"])
        return {}

    graph.add_node("build_document", build_document)
    graph.add_node("embed", embed)
    graph.add_node("persist", persist)

    graph.set_entry_point("build_document")
    graph.add_edge("build_document", "embed")
    graph.add_edge("embed", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


def build_recommendation_graph(
    embedder: ResumeEmbedder,
    store: ResumeVectorStore,
    ranker: RecommendationRanker | None = None,
):
    graph = StateGraph(RecommendState)

    def build_query(state: RecommendState):
        query_text = store.build_query_text(state["request"])
        return {"query_text": query_text}

    def embed_query(state: RecommendState):
        embedding = embedder.embed_text(state["query_text"])
        return {"query_embedding": embedding}

    def search(state: RecommendState):
        candidates = store.query(state["query_embedding"], top_k=state["request"].top_k)
        return {"candidates": candidates}

    def rerank(state: RecommendState):
        ranked = _rerank_candidates(
            candidates=state["candidates"],
            request=state["request"],
            query_text=state.get("query_text", ""),
            store=store,
            ranker=ranker,
        )
        return {"ranked": ranked}

    graph.add_node("build_query", build_query)
    graph.add_node("embed_query", embed_query)
    graph.add_node("search", search)
    graph.add_node("rerank", rerank)

    graph.set_entry_point("build_query")
    graph.add_edge("build_query", "embed_query")
    graph.add_edge("embed_query", "search")
    graph.add_edge("search", "rerank")
    graph.add_edge("rerank", END)

    return graph.compile()


def _rerank_candidates(
    candidates: Sequence[Tuple[Resume, float]],
    request: RecommendationRequest,
    query_text: str,
    store: ResumeVectorStore,
    ranker: RecommendationRanker | None = None,
) -> List[RecommendationResult]:
    request_skill_set = {skill.lower().strip() for skill in request.skills if skill.strip()}
    ranked: List[RecommendationResult] = []

    for resume, similarity in candidates:
        candidate_skills = {skill.lower().strip(): skill for skill in resume.skills}
        matches = [cand for key, cand in candidate_skills.items() if key in request_skill_set]
        skill_score = (len(matches) / max(len(request_skill_set), 1)) if request_skill_set else 0
        role_score = 0.0
        if resume.role and resume.role.strip():
            role_score = 1.0 if resume.role.lower() == request.role.lower() else 0.5 if request.role.lower() in resume.role.lower() else 0.0

        cross_score = 0.0
        if ranker is not None:
            candidate_text = store.build_document(resume)
            cross_score = ranker.score(query_text, candidate_text)

        final_score = (
            (similarity * 0.4)
            + (skill_score * 0.25)
            + (role_score * 0.1)
            + (cross_score * 0.25)
        )
        explanation = _build_explanation(resume, final_score, matches, role_score)
        ranked.append(
            RecommendationResult(
                candidate=resume,
                match_score=round(final_score, 4),
                matching_skills=matches,
                explanation=explanation,
            )
        )

    ranked.sort(key=lambda result: result.match_score, reverse=True)
    return ranked


def _build_explanation(
    resume: Resume, score: float, matches: List[str], role_score: float
) -> str:
    parts = [
        f"Score {score:.2f}",
        f"role match: {'exact' if role_score == 1 else 'different'}",
        f"skill overlap: {', '.join(matches) if matches else 'none'}",
    ]
    return "; ".join(parts)
