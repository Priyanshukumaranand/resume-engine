from __future__ import annotations

import re
from typing import List, Tuple, TypedDict, Optional

from langgraph.graph import END, StateGraph

try:
    from .embedder import ResumeEmbedder
    from .llm import ResumeLLM
    from .models import Resume, SectionChunk
    from .vectorstore import ResumeVectorStore
    from .anonymizer import generate_anon_id, strip_pii, extract_email
except ImportError:
    from backend.embedder import ResumeEmbedder  # type: ignore
    from backend.llm import ResumeLLM  # type: ignore
    from backend.models import Resume, SectionChunk  # type: ignore
    from backend.vectorstore import ResumeVectorStore  # type: ignore
    from backend.anonymizer import generate_anon_id, strip_pii, extract_email  # type: ignore


# Section detection patterns
SECTION_PATTERNS = {
    "skills": [
        r"(?i)^[\s]*(?:technical\s+)?skills?\s*[:|\-|–]?",
        r"(?i)^[\s]*(?:core\s+)?competenc(?:ies|e)\s*[:|\-|–]?",
        r"(?i)^[\s]*technologies?\s*[:|\-|–]?",
        r"(?i)^[\s]*tech\s+stack\s*[:|\-|–]?",
        r"(?i)^[\s]*proficienc(?:ies|y)\s*[:|\-|–]?",
        r"(?i)^[\s]*tools?\s*(?:&|and)?\s*technologies?\s*[:|\-|–]?",
    ],
    "experience": [
        r"(?i)^[\s]*(?:work\s+)?experience\s*[:|\-|–]?",
        r"(?i)^[\s]*employment\s*(?:history)?\s*[:|\-|–]?",
        r"(?i)^[\s]*work\s+history\s*[:|\-|–]?",
        r"(?i)^[\s]*professional\s+experience\s*[:|\-|–]?",
    ],
    "projects": [
        r"(?i)^[\s]*projects?\s*[:|\-|–]?",
        r"(?i)^[\s]*(?:personal|academic|side)\s+projects?\s*[:|\-|–]?",
        r"(?i)^[\s]*portfolio\s*[:|\-|–]?",
        r"(?i)^[\s]*work\s+samples?\s*[:|\-|–]?",
    ],
    "education": [
        r"(?i)^[\s]*education\s*[:|\-|–]?",
        r"(?i)^[\s]*academic\s*(?:background|qualifications?)?\s*[:|\-|–]?",
        r"(?i)^[\s]*degrees?\s*[:|\-|–]?",
        r"(?i)^[\s]*qualifications?\s*[:|\-|–]?",
    ],
}

# Weights for different section types in retrieval
SECTION_WEIGHTS = {
    "skills": 1.3,      # Boost skills matches
    "projects": 1.2,    # Boost project matches
    "experience": 1.1,  # Slight boost for experience
    "education": 1.0,   # Normal weight
    "header": 0.9,      # Slightly lower for header info
    "unknown": 1.0,     # Normal weight for unknown
}


class AddResumeState(TypedDict, total=False):
    resume: Resume
    raw_text: str
    chunks: List[str]
    section_chunks: List[SectionChunk]
    embeddings: List[List[float]]


def detect_sections(text: str) -> List[Tuple[str, int, int]]:
    """
    Detect section boundaries in resume text.
    
    Returns:
        List of (section_type, start_idx, end_idx) tuples
    """
    if not text:
        return []
    
    lines = text.split("\n")
    sections: List[Tuple[str, int, int]] = []
    current_section = "header"
    current_start = 0
    current_pos = 0
    
    for line in lines:
        line_start = current_pos
        current_pos += len(line) + 1  # +1 for newline
        
        # Check if line matches any section header
        for section_type, patterns in SECTION_PATTERNS.items():
            for pattern in patterns:
                if re.match(pattern, line.strip()):
                    # Save previous section
                    if current_start < line_start:
                        sections.append((current_section, current_start, line_start))
                    current_section = section_type
                    current_start = line_start
                    break
            else:
                continue
            break
    
    # Add final section
    if current_start < len(text):
        sections.append((current_section, current_start, len(text)))
    
    return sections


def chunk_by_section(
    text: str,
    sections: List[Tuple[str, int, int]],
    chunk_size: int = 800,
    overlap: int = 200,
) -> List[SectionChunk]:
    """
    Create chunks preserving section boundaries.
    
    Each chunk is tagged with its section type.
    Chunks do NOT cross section boundaries.
    
    Args:
        text: Full resume text
        sections: Section boundaries from detect_sections()
        chunk_size: Maximum chunk size in characters
        overlap: Overlap between chunks within same section
        
    Returns:
        List of SectionChunk with section metadata
    """
    if not text or not sections:
        # Fallback: single chunk with unknown section
        if text:
            return [SectionChunk(text=text, section_type="unknown", chunk_index=0)]
        return []
    
    chunks: List[SectionChunk] = []
    global_index = 0
    
    for section_type, start, end in sections:
        section_text = text[start:end].strip()
        if not section_text:
            continue
        
        # Chunk within section
        section_start = 0
        while section_start < len(section_text):
            chunk_end = section_start + chunk_size
            chunk_text = section_text[section_start:chunk_end].strip()
            
            if chunk_text:
                chunks.append(SectionChunk(
                    text=chunk_text,
                    section_type=section_type,
                    chunk_index=global_index,
                ))
                global_index += 1
            
            if chunk_end >= len(section_text):
                break
            
            # Move forward with overlap
            section_start = chunk_end - overlap
            if section_start < 0:
                section_start = 0
    
    return chunks


def _chunk_text_legacy(text: str, chunk_size: int = 800, overlap: int = 200) -> List[str]:
    """Legacy chunking function for backward compatibility."""
    clean = (text or "").strip()
    if not clean:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(clean):
        end = start + chunk_size
        chunk = clean[start:end]
        chunks.append(chunk)
        if end >= len(clean):
            break
        start = end - overlap
        if start < 0:
            start = 0
    return chunks


def build_add_resume_graph(embedder: ResumeEmbedder, store: ResumeVectorStore):
    """Build LangGraph for adding a new resume with section-aware chunking."""
    graph = StateGraph(AddResumeState)

    def detect_and_chunk(state: AddResumeState):
        """Detect sections and create section-aware chunks with candidate context."""
        resume = state["resume"]
        raw_text = resume.raw_text or _combine_fields(resume)
        
        # Detect sections
        sections = detect_sections(raw_text)
        
        # Create section-aware chunks
        section_chunks = chunk_by_section(raw_text, sections)
        
        # Prepend candidate context to each chunk (Parent Document Retrieval)
        # This helps the LLM identify which candidate each chunk belongs to
        role_part = f" ({resume.role})" if resume.role else ""
        skills_part = f" | Skills: {', '.join(resume.skills[:5])}" if resume.skills else ""
        context_prefix = f"[Candidate: {resume.name}{role_part}{skills_part}]\n"
        
        # Create chunks with context
        chunks = [context_prefix + sc.text for sc in section_chunks]
        
        return {
            "chunks": chunks,
            "section_chunks": section_chunks,
        }

    def embed(state: AddResumeState):
        embeddings = embedder.embed_many(state["chunks"])
        return {"embeddings": embeddings}

    def persist(state: AddResumeState):
        section_chunks = state.get("section_chunks", [])
        section_types = [sc.section_type for sc in section_chunks]
        store.add_resume_chunks(
            state["resume"],
            state["embeddings"],
            state["chunks"],
            section_types=section_types,
        )
        return {}

    graph.add_node("chunk", detect_and_chunk)
    graph.add_node("embed", embed)
    graph.add_node("persist", persist)

    graph.set_entry_point("chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


class RefreshResumeState(TypedDict, total=False):
    resume: Resume
    extracted: dict
    chunks: List[str]
    section_chunks: List[SectionChunk]
    embeddings: List[List[float]]


def build_refresh_resume_graph(
    resume_llm: ResumeLLM, embedder: ResumeEmbedder, store: ResumeVectorStore
):
    """Build LangGraph for refreshing/re-extracting a resume."""
    graph = StateGraph(RefreshResumeState)

    def extract(state: RefreshResumeState):
        resume = state.get("resume")
        text = resume.raw_text or resume.experience or resume.summary or ""
        extracted = resume_llm.extract_resume_fields(text) if text else {}
        return {"extracted": extracted}

    def merge(state: RefreshResumeState):
        resume = state["resume"]
        data = state.get("extracted", {}) or {}
        
        # Preserve existing anon_id or generate new one
        email = data.get("email") or resume.email or ""
        anon_id = resume.anon_id if resume.anon_id else generate_anon_id(email)
        
        updated = Resume(
            id=resume.id,
            anon_id=anon_id,
            name=(data.get("name") or resume.name or "Unknown").strip(),
            email=email,
            role=(data.get("role") or resume.role),
            skills=_deduplicate_skills(data.get("skills") or resume.skills),
            projects=data.get("projects") or resume.projects,
            education=data.get("education") or resume.education,
            experience=(data.get("experience") or resume.experience),
            summary=(data.get("summary") or resume.summary),
            raw_text=resume.raw_text,
        )
        return {"resume": updated}

    def detect_and_chunk(state: RefreshResumeState):
        resume = state["resume"]
        raw_text = resume.raw_text or _combine_fields(resume)
        
        sections = detect_sections(raw_text)
        section_chunks = chunk_by_section(raw_text, sections)
        
        # Prepend candidate context to each chunk
        role_part = f" ({resume.role})" if resume.role else ""
        skills_part = f" | Skills: {', '.join(resume.skills[:5])}" if resume.skills else ""
        context_prefix = f"[Candidate: {resume.name}{role_part}{skills_part}]\n"
        chunks = [context_prefix + sc.text for sc in section_chunks]
        
        return {
            "chunks": chunks,
            "section_chunks": section_chunks,
        }

    def embed(state: RefreshResumeState):
        embeddings = embedder.embed_many(state["chunks"])
        return {"embeddings": embeddings}

    def persist(state: RefreshResumeState):
        store.delete(state["resume"].id)
        section_chunks = state.get("section_chunks", [])
        section_types = [sc.section_type for sc in section_chunks]
        store.add_resume_chunks(
            state["resume"],
            state["embeddings"],
            state["chunks"],
            section_types=section_types,
        )
        return {}

    graph.add_node("extract", extract)
    graph.add_node("merge", merge)
    graph.add_node("chunk", detect_and_chunk)
    graph.add_node("embed", embed)
    graph.add_node("persist", persist)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "merge")
    graph.add_edge("merge", "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "persist")
    graph.add_edge("persist", END)

    return graph.compile()


def _combine_fields(resume: Resume) -> str:
    """Combine resume fields into searchable text."""
    parts = [
        resume.name,
        resume.role or "",
        ", ".join(resume.skills) if resume.skills else "",
        ", ".join(resume.projects) if resume.projects else "",
        resume.education or "",
        resume.experience,
        resume.summary,
    ]
    return "\n".join(part for part in parts if part).strip()


def _deduplicate_skills(skills: List[str]) -> List[str]:
    seen = set()
    cleaned: List[str] = []
    for skill in skills or []:
        text = skill.strip()
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        cleaned.append(text)
    return cleaned[:15]


def get_section_weight(section_type: str) -> float:
    """Get retrieval weight for a section type."""
    return SECTION_WEIGHTS.get(section_type, 1.0)
