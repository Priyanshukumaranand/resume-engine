from __future__ import annotations

import json
import os
from typing import List, Sequence

import chromadb

from models import RecommendationRequest, Resume


class ResumeVectorStore:
    """Manages all access to the ChromaDB resume collection."""

    def __init__(self, persist_directory: str, collection_name: str = "resumes") -> None:
        os.makedirs(persist_directory, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _metadata_from_resume(self, resume: Resume) -> dict:
        metadata = resume.model_dump()
        metadata["skills"] = json.dumps(resume.skills)
        metadata["role"] = resume.role or ""
        return metadata

    def _resume_from_metadata(self, metadata: dict) -> Resume:
        skills_raw = metadata.get("skills", "[]")
        skills = json.loads(skills_raw) if isinstance(skills_raw, str) else skills_raw
        return Resume(
            id=str(metadata["id"]),
            name=metadata["name"],
            email=metadata["email"],
            role=metadata.get("role") or None,
            skills=skills,
            experience=metadata["experience"],
            summary=metadata["summary"],
        )

    def add_resume(self, resume: Resume, embedding: Sequence[float], document: str) -> None:
        self._collection.upsert(
            ids=[resume.id],
            embeddings=[list(embedding)],
            documents=[document],
            metadatas=[self._metadata_from_resume(resume)],
        )

    def get_all_resumes(self) -> List[Resume]:
        result = self._collection.get(include=["metadatas"])
        if not result or not result.get("metadatas"):
            return []
        return [self._resume_from_metadata(metadata) for metadata in result["metadatas"]]

    def has_resumes(self) -> bool:
        try:
            return self._collection.count() > 0
        except AttributeError:
            result = self._collection.get(limit=1)
            return bool(result.get("ids")) if result else False

    def query(self, embedding: Sequence[float], top_k: int) -> List[tuple[Resume, float]]:
        query_result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=top_k,
            include=["metadatas", "distances"],
        )
        if not query_result.get("metadatas"):
            return []
        candidates: List[tuple[Resume, float]] = []
        metadatas = query_result["metadatas"][0]
        distances = query_result["distances"][0]
        for metadata, distance in zip(metadatas, distances):
            resume = self._resume_from_metadata(metadata)
            similarity = 1 - distance  # cosine metric returns smaller distance for closer vectors
            candidates.append((resume, similarity))
        return candidates

    def delete(self, resume_id: str) -> bool:
        if not resume_id:
            return False
        existing = self._collection.get(ids=[resume_id])
        ids = existing.get("ids") if existing else None
        if not ids:
            return False
        self._collection.delete(ids=[resume_id])
        return True

    @staticmethod
    def build_document(resume: Resume) -> str:
        parts = [
            resume.role or "",
            ", ".join(resume.skills) if resume.skills else "",
            resume.experience,
            resume.summary,
        ]
        return "\n".join(part for part in parts if part).strip()

    @staticmethod
    def build_query_text(request: RecommendationRequest) -> str:
        parts = [
            request.role,
            ", ".join(request.skills) if request.skills else "",
            request.summary,
        ]
        return "\n".join(part for part in parts if part).strip()
