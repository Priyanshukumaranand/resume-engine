from __future__ import annotations

import json
import os
from typing import List, Sequence, Tuple, Optional

import chromadb

try:
    from .models import Resume
    from .graph import get_section_weight, SECTION_WEIGHTS
except ImportError:
    from backend.models import Resume  # type: ignore
    try:
        from backend.graph import get_section_weight, SECTION_WEIGHTS  # type: ignore
    except ImportError:
        SECTION_WEIGHTS = {"skills": 1.3, "projects": 1.2, "experience": 1.1, "education": 1.0, "header": 0.9, "unknown": 1.0}
        def get_section_weight(section_type: str) -> float:
            return SECTION_WEIGHTS.get(section_type, 1.0)


# Confidence thresholds for retrieval
MIN_SIMILARITY_THRESHOLD = 0.4
CONFIDENT_SIMILARITY = 0.7


class ResumeVectorStore:
    """
    Manages ChromaDB resume collection with section-aware storage and weighted retrieval.
    
    Features:
    - Section metadata stored per chunk
    - Weighted retrieval based on section type
    - Privacy: email stored internally but not exposed
    """

    def __init__(self, persist_directory: str, collection_name: str = "resumes") -> None:
        os.makedirs(persist_directory, exist_ok=True)
        self._client = chromadb.PersistentClient(path=persist_directory)
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def _metadata_from_resume(
        self,
        resume: Resume,
        chunk_index: int,
        section_type: str = "unknown",
    ) -> dict:
        """Create metadata dict for a chunk with enhanced filtering support."""
        # Create searchable skills text for metadata filtering
        skills_text = ", ".join(resume.skills).lower() if resume.skills else ""
        
        metadata = {
            "id": resume.id,
            "anon_id": resume.anon_id,
            "name": resume.name,
            # email is stored for internal use but should not be returned in queries
            "email": resume.email,
            "role": resume.role or "",
            "skills": json.dumps(resume.skills),
            "skills_text": skills_text,  # For text-based filtering
            "projects": json.dumps(resume.projects),
            "education": resume.education or "",
            "experience": resume.experience[:800] if resume.experience else "",
            "summary": resume.summary[:800] if resume.summary else "",
            "chunk_index": chunk_index,
            "section_type": section_type,
        }
        return metadata

    def _resume_from_metadata(self, metadata: dict) -> Resume:
        """Reconstruct Resume from chunk metadata."""
        skills_raw = metadata.get("skills", "[]")
        skills = json.loads(skills_raw) if isinstance(skills_raw, str) else skills_raw
        
        projects_raw = metadata.get("projects", "[]")
        projects = json.loads(projects_raw) if isinstance(projects_raw, str) else projects_raw
        
        # Generate anon_id from email if not present (backward compatibility)
        anon_id = metadata.get("anon_id", "")
        if not anon_id:
            email = metadata.get("email", "")
            if email:
                import hashlib
                anon_id = hashlib.sha256(email.lower().encode()).hexdigest()[:16]
            else:
                # Fallback: generate from resume id
                resume_id = str(metadata.get("id", "unknown"))
                anon_id = hashlib.sha256(resume_id.encode()).hexdigest()[:16]
        
        return Resume(
            id=str(metadata["id"]),
            anon_id=anon_id,
            name=metadata.get("name") or "Unknown",
            email=metadata.get("email", ""),
            role=metadata.get("role") or None,
            skills=skills,
            projects=projects,
            education=metadata.get("education") or None,
            experience=metadata.get("experience", ""),
            summary=metadata.get("summary", ""),
        )

    def add_resume_chunks(
        self,
        resume: Resume,
        embeddings: Sequence[Sequence[float]],
        chunks: Sequence[str],
        section_types: Optional[Sequence[str]] = None,
    ) -> None:
        """
        Add resume chunks with section metadata.
        
        Args:
            resume: Resume model
            embeddings: Embeddings for each chunk
            chunks: Text chunks
            section_types: Section type for each chunk (skills, experience, etc.)
        """
        ids: List[str] = []
        metadatas: List[dict] = []
        documents: List[str] = []
        
        # Default section types if not provided
        if section_types is None:
            section_types = ["unknown"] * len(chunks)
        
        for index, (embedding, chunk) in enumerate(zip(embeddings, chunks)):
            section = section_types[index] if index < len(section_types) else "unknown"
            ids.append(f"{resume.id}::chunk-{index}")
            metadatas.append(self._metadata_from_resume(resume, index, section))
            documents.append(chunk)

        if not ids:
            return

        self._collection.upsert(
            ids=ids,
            embeddings=[list(e) for e in embeddings],
            documents=documents,
            metadatas=metadatas,
        )

    def get_all_resumes(self) -> List[Resume]:
        """Get all unique resumes (deduplicated by ID)."""
        result = self._collection.get(include=["metadatas"], limit=1000)
        if not result or not result.get("metadatas"):
            return []
        seen = {}
        for metadata in result["metadatas"]:
            resume_id = metadata.get("id")
            if resume_id in seen:
                continue
            seen[resume_id] = self._resume_from_metadata(metadata)
        return list(seen.values())

    def has_resumes(self) -> bool:
        try:
            return self._collection.count() > 0
        except AttributeError:
            result = self._collection.get(limit=1)
            return bool(result.get("ids")) if result else False

    def query(
        self,
        embedding: Sequence[float],
        top_k: int,
        apply_section_weights: bool = True,
    ) -> List[Tuple[Resume, str, float, str]]:
        """
        Query for similar resumes with parent document context.
        
        Improvements:
        - Fetches 50 candidates for better coverage
        - Prepends candidate name/role to chunks for context
        - Deduplicates by resume ID to avoid single-candidate dominance
        
        Returns:
            List of (Resume, chunk_with_context, weighted_similarity, section_type) tuples
        """
        # Retrieve many more candidates for better coverage
        fetch_k = min(top_k * 10, 50)
        
        query_result = self._collection.query(
            query_embeddings=[list(embedding)],
            n_results=fetch_k,
            include=["metadatas", "distances", "documents"],
        )
        if not query_result.get("metadatas"):
            return []
        
        # Group candidates by resume ID for deduplication
        resume_best_chunks: dict = {}  # resume_id -> (resume, chunk, similarity, section, skills_match)
        
        metadatas = query_result["metadatas"][0]
        distances = query_result["distances"][0]
        documents = query_result.get("documents", [[]])[0]
        
        for metadata, distance, document in zip(metadatas, distances, documents):
            resume = self._resume_from_metadata(metadata)
            raw_similarity = 1 - distance
            section_type = metadata.get("section_type", "unknown")
            
            # Apply section weight for technical queries
            if apply_section_weights:
                weight = get_section_weight(section_type)
                weighted_similarity = raw_similarity * weight
            else:
                weighted_similarity = raw_similarity
            
            # Prepend candidate context to chunk (Parent Document Retrieval)
            role_part = f" ({resume.role})" if resume.role else ""
            skills_part = f" - Skills: {', '.join(resume.skills[:5])}" if resume.skills else ""
            chunk_with_context = f"Candidate: {resume.name}{role_part}{skills_part}\n{document}"
            
            # Keep best chunk per resume (deduplication)
            resume_id = resume.id
            if resume_id not in resume_best_chunks:
                resume_best_chunks[resume_id] = (resume, chunk_with_context, weighted_similarity, section_type)
            else:
                # Keep the chunk with higher similarity
                if weighted_similarity > resume_best_chunks[resume_id][2]:
                    resume_best_chunks[resume_id] = (resume, chunk_with_context, weighted_similarity, section_type)
        
        # Convert to list and sort by similarity
        candidates = list(resume_best_chunks.values())
        candidates.sort(key=lambda x: x[2], reverse=True)
        return candidates[:top_k]

    def query_by_skills(
        self,
        skills: List[str],
        embedding: Sequence[float],
        top_k: int,
    ) -> List[Tuple[Resume, List[str], float]]:
        """
        Query for resumes matching specific skills.
        
        Returns candidates with their explicitly matched skills.
        
        Returns:
            List of (Resume, matched_skills, similarity) tuples
        """
        candidates = self.query(embedding, top_k * 2)
        
        skills_lower = {s.lower() for s in skills}
        results: List[Tuple[Resume, List[str], float]] = []
        seen_ids = set()
        
        for resume, chunk, similarity, section in candidates:
            if resume.id in seen_ids:
                continue
            seen_ids.add(resume.id)
            
            # Find explicitly matched skills
            resume_skills_lower = {s.lower() for s in resume.skills}
            matched = [s for s in resume.skills if s.lower() in skills_lower]
            
            if matched:
                results.append((resume, matched, similarity))
        
        # Sort by number of matched skills, then similarity
        results.sort(key=lambda x: (len(x[1]), x[2]), reverse=True)
        return results[:top_k]

    def delete(self, resume_id: str) -> bool:
        if not resume_id:
            return False
        existing = self._collection.get(where={"id": resume_id})
        ids = existing.get("ids") if existing else None
        if not ids:
            return False
        self._collection.delete(where={"id": resume_id})
        return True

    @staticmethod
    def build_document(resume: Resume) -> str:
        parts = [
            resume.role or "",
            ", ".join(resume.skills) if resume.skills else "",
            ", ".join(resume.projects) if resume.projects else "",
            resume.experience,
            resume.summary,
        ]
        return "\n".join(part for part in parts if part).strip()

    def get_chunk_similarities(
        self,
        embedding: Sequence[float],
        top_k: int,
    ) -> Tuple[List[float], List[str]]:
        """
        Get raw similarity scores and section types for confidence calculation.
        
        Returns:
            Tuple of (similarities, section_types)
        """
        candidates = self.query(embedding, top_k, apply_section_weights=False)
        similarities = [c[2] for c in candidates]
        sections = [c[3] for c in candidates]
        return similarities, sections
