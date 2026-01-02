from __future__ import annotations

import os
from uuid import uuid4
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from io import BytesIO

try:  # Support running as package or as a plain script
    from .embedder import ResumeEmbedder
    from .graph import build_add_resume_graph, build_refresh_resume_graph
    from .models import (
        QARequest,
        QAResponse,
        QASource,
        Resume,
        ResumeListResponse,
        ResumeResponse,
    )
    from .llm import ResumeLLM
    from .vectorstore import ResumeVectorStore
    from .anonymizer import generate_anon_id, strip_pii, extract_email, UNKNOWN_VALUE
    from .validator import ResponseValidator, FALLBACK_INSUFFICIENT_INFO
except ImportError:
    from backend.embedder import ResumeEmbedder  # type: ignore
    from backend.graph import build_add_resume_graph, build_refresh_resume_graph  # type: ignore
    from backend.models import (  # type: ignore
        QARequest,
        QAResponse,
        QASource,
        Resume,
        ResumeListResponse,
        ResumeResponse,
    )
    from backend.llm import ResumeLLM  # type: ignore
    from backend.vectorstore import ResumeVectorStore  # type: ignore
    from backend.anonymizer import generate_anon_id, strip_pii, extract_email, UNKNOWN_VALUE  # type: ignore
    from backend.validator import ResponseValidator, FALLBACK_INSUFFICIENT_INFO  # type: ignore


APP_NAME = "Privacy-First Resume Discovery API"
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PERSIST_DIRECTORY = PROJECT_ROOT / "chroma_storage"

# Lowered confidence threshold for better response rate
CONFIDENCE_THRESHOLD = 0.35

app = FastAPI(
    title=APP_NAME,
    version="2.0.0",
    description="Privacy-preserving student discovery API",
)

default_origins = (
    "https://ce-bootcamp.vercel.app,"
    "https://branchbase-backend.azurewebsites.net,"
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://localhost:5173,"
    "http://127.0.0.1:5173"
)
allowed_origins = [
    origin.strip()
    for origin in os.getenv("ALLOWED_ORIGINS", default_origins).split(",")
    if origin.strip()
]
if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
)

# Initialize components
embedder = ResumeEmbedder()
vector_store = ResumeVectorStore(persist_directory=str(PERSIST_DIRECTORY))
resume_llm = ResumeLLM()
response_validator = ResponseValidator()
add_resume_graph = build_add_resume_graph(embedder, vector_store)
refresh_resume_graph = build_refresh_resume_graph(resume_llm, embedder, vector_store)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/ingest_pdf")
async def ingest_pdf(file: UploadFile = File(...)) -> dict:
    """
    Ingest a PDF resume with full extraction pipeline.
    
    This is the single endpoint for processing resumes:
    1. Extracts text from PDF
    2. Strips PII (phone, URLs, addresses) - keeps name
    3. Runs strict field extraction with source attribution
    4. Generates stable anon_id from email
    5. Chunks by section and stores with embeddings
    
    Returns complete extracted data for verification.
    """
    if file.content_type not in {"application/pdf", "application/octet-stream"}:
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported")

    pdf_text = _extract_pdf_text(file)
    if not pdf_text:
        raise HTTPException(status_code=400, detail="Could not read any text from the PDF")

    # Extract email for anon_id before stripping PII
    email = extract_email(pdf_text) or UNKNOWN_VALUE
    anon_id = generate_anon_id(email)
    
    # Strip phone, URLs, addresses from text (keep name)
    sanitized_text = strip_pii(pdf_text)

    # Run strict extraction
    try:
        extracted = resume_llm.extract_resume_fields(pdf_text) or {}
    except RuntimeError:
        # Fallback: still ingest with basic data
        extracted = {
            "name": "Unknown",
            "role": None,
            "skills": [],
            "projects": [],
            "education": None,
            "experience": sanitized_text[:1500],
            "summary": sanitized_text[:500],
        }

    resume = Resume(
        id=str(uuid4()),
        anon_id=anon_id,
        name=extracted.get("name") or "Unknown",
        email=email,
        role=extracted.get("role"),
        skills=_deduplicate_skills(extracted.get("skills") or []),
        projects=extracted.get("projects") or [],
        education=extracted.get("education"),
        experience=extracted.get("experience") or sanitized_text[:1500],
        summary=extracted.get("summary") or sanitized_text[:500],
        raw_text=sanitized_text,
    )
    
    try:
        add_resume_graph.invoke({"resume": resume})
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    
    # Return complete extracted data for verification
    return {
        "status": "processed",
        "id": resume.id,
        "anon_id": resume.anon_id,
        "name": resume.name,
        "role": resume.role,
        "skills": resume.skills,
        "projects": resume.projects,
        "education": resume.education,
        "summary": resume.summary[:300] if resume.summary else None,
    }


@app.get("/resumes", response_model=ResumeListResponse)
async def list_resumes() -> ResumeListResponse:
    """
    List all resumes.
    
    Privacy: Email hidden, name visible.
    """
    resumes = vector_store.get_all_resumes()
    response_resumes = [ResumeResponse.from_resume(r) for r in resumes]
    return ResumeListResponse(resumes=response_resumes)


@app.delete("/resumes/{resume_id}")
async def delete_resume(resume_id: str) -> dict:
    deleted = vector_store.delete(resume_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Resume not found")
    return {"status": "deleted", "id": resume_id}


@app.post("/qa", response_model=QAResponse)
async def answer_question(request: QARequest) -> QAResponse:
    """
    Evidence-based QA with improved search.
    
    Features:
    - Query expansion for better section matching
    - Answers based on retrieved context
    - Falls back to rule-based answer when LLM fails
    - Source sections and evidence snippets included
    """
    if not vector_store.has_resumes():
        raise HTTPException(status_code=404, detail="No resumes stored")

    question = request.question.strip()
    question_lower = question.lower()
    
    # Query expansion - add relevant section keywords based on question type
    expanded_query = question
    if any(word in question_lower for word in ["skill", "technology", "tech", "know", "language", "framework"]):
        expanded_query = f"{question} Technical Skills Programming Languages ML AI"
    elif any(word in question_lower for word in ["project", "built", "created", "developed", "made"]):
        expanded_query = f"{question} Projects Key Projects developed built"
    elif any(word in question_lower for word in ["experience", "work", "job", "role", "intern"]):
        expanded_query = f"{question} Experience Work History internship"
    elif any(word in question_lower for word in ["education", "degree", "university", "college"]):
        expanded_query = f"{question} Education University B.Tech degree"
    
    # Increase top_k for better coverage
    effective_top_k = max(request.top_k, 5)
    query_embedding = embedder.embed_text(expanded_query)
    
    # Get candidates with section metadata
    candidates = vector_store.query(query_embedding, top_k=effective_top_k)
    if not candidates:
        raise HTTPException(status_code=404, detail="No relevant resumes found")

    # Extract data for confidence calculation
    similarities = [c[2] for c in candidates]
    section_types = [c[3] for c in candidates]
    chunks = [c[1] for c in candidates]
    
    # Calculate confidence - use max similarity as primary indicator
    max_similarity = max(similarities) if similarities else 0
    avg_similarity = sum(similarities) / len(similarities) if similarities else 0
    confidence = (max_similarity * 0.7) + (avg_similarity * 0.3)
    
    # Build sources (name visible, email hidden)
    matches = _prepare_matches(candidates)
    sources = [_build_qa_source(match) for match in matches]
    
    # Get unique section types
    unique_sections = list(set(section_types))
    
    # Evidence snippets from top matches
    evidence_snippets = [m.chunk[:200] for m in matches[:3] if m.chunk]
    
    # Try to generate answer - be more permissive
    is_fallback = False
    answer = ""
    
    if confidence >= CONFIDENCE_THRESHOLD and matches:
        # Build context and get LLM answer
        context = _build_qa_context(matches)
        try:
            answer = resume_llm.answer_question(question, context).strip()
        except RuntimeError:
            answer = ""

        # Check if answer is useful
        if not answer or _is_unhelpful_answer(answer):
            # Use rule-based answer instead of fallback
            answer = _build_rule_based_answer(question, matches)
            if answer:
                is_fallback = False  # Rule-based is still useful
            else:
                answer = FALLBACK_INSUFFICIENT_INFO
                is_fallback = True
    else:
        # Low confidence - try rule-based first
        answer = _build_rule_based_answer(question, matches)
        if answer:
            is_fallback = False
        else:
            answer = FALLBACK_INSUFFICIENT_INFO
            is_fallback = True
    
    return QAResponse(
        answer=answer,
        confidence_score=round(confidence, 3),
        source_sections=unique_sections,
        evidence_snippets=evidence_snippets,
        sources=sources,
        is_fallback=is_fallback,
    )


def _is_unhelpful_answer(answer: str) -> bool:
    """Check if the LLM answer is unhelpful."""
    unhelpful_phrases = {
        "i do not know",
        "i don't know",
        "i can't find",
        "i cannot find",
        "insufficient information",
        "no information",
        "not found",
        "not available",
    }
    answer_lower = answer.lower().strip()
    return any(phrase in answer_lower for phrase in unhelpful_phrases)


def _extract_pdf_text(upload: UploadFile) -> str:
    try:
        raw = upload.file.read()
    finally:
        upload.file.close()

    if not raw:
        return ""

    reader = PdfReader(BytesIO(raw))
    texts: list[str] = []
    for page in reader.pages:
        try:
            extracted = page.extract_text() or ""
        except Exception:
            extracted = ""
        if extracted:
            texts.append(extracted.strip())

    combined = "\n\n".join(texts)
    # Trim overly large payloads
    return combined[:20000]


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
    return cleaned[:15]


@dataclass
class QAMatch:
    resume: Resume
    chunk: str
    similarity: float
    section_type: str


def _prepare_matches(candidates: List[tuple]) -> List[QAMatch]:
    matches: List[QAMatch] = []
    for item in candidates:
        resume, chunk, similarity = item[0], item[1], item[2]
        section_type = item[3] if len(item) > 3 else "unknown"
        matches.append(QAMatch(
            resume=resume,
            chunk=chunk,
            similarity=similarity,
            section_type=section_type,
        ))
    matches.sort(key=lambda match: match.similarity, reverse=True)
    return matches


def _build_qa_source(match: QAMatch) -> QASource:
    resume = match.resume
    excerpt = (match.chunk or resume.summary or resume.experience)[:400]
    return QASource(
        id=resume.id,
        anon_id=resume.anon_id,
        name=resume.name,  # Name visible
        role=resume.role,
        skills=resume.skills,
        summary=excerpt,
    )


def _build_qa_context(matches: List[QAMatch], limit: int = 5) -> str:
    chunks: List[str] = []
    for match in matches[:limit]:
        resume = match.resume
        parts = [
            f"Candidate: {resume.name}",
            f"Role: {resume.role or 'Not specified'}",
            f"Skills: {', '.join(resume.skills) if resume.skills else 'Not listed'}",
            f"Projects: {', '.join(resume.projects) if resume.projects else 'Not listed'}",
            f"Education: {resume.education or 'Not provided'}",
            f"Experience: {resume.experience[:500] if resume.experience else 'Not provided'}",
            f"Summary: {resume.summary[:300] if resume.summary else 'Not provided'}",
        ]
        chunks.append("\n".join(parts))
    return "\n\n---\n\n".join(chunks)


def _build_rule_based_answer(question: str, matches: List[QAMatch]) -> str:
    """Build a rule-based answer from matched resumes and chunks."""
    if not matches:
        return ""

    question_lower = question.lower()
    
    # Deduplicate by resume ID
    seen_ids = set()
    unique_matches = []
    for m in matches:
        if m.resume.id not in seen_ids:
            seen_ids.add(m.resume.id)
            unique_matches.append(m)
    
    if not unique_matches:
        return ""
    
    top = unique_matches[0]
    resume = top.resume
    chunk = top.chunk or ""
    
    # Helper to extract section from chunk text
    def _extract_section(chunk: str, section_name: str) -> str:
        """Extract content after a section header from chunk."""
        import re
        patterns = [
            rf'{section_name}[:\s]*\n([^\n]+(?:\n(?![A-Z][a-z]+:)[^\n]+)*)',
            rf'{section_name}\s*([^\n]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, chunk, re.IGNORECASE)
            if match:
                return match.group(1).strip()[:400]
        return ""
    
    # Build answer based on question type
    if any(word in question_lower for word in ["experience", "work", "job", "role", "intern"]):
        # Look for experience section in chunk
        exp_text = _extract_section(chunk, "Experience") or _extract_section(chunk, "Work")
        if exp_text:
            return f"{resume.name}'s experience: {exp_text}"
        exp = resume.experience[:400] if resume.experience else resume.summary[:400]
        return f"{resume.name} has the following experience: {exp}"
    
    elif any(word in question_lower for word in ["skill", "technology", "tech", "know", "language"]):
        # Look for skills section in chunk
        skills_text = _extract_section(chunk, "Skills") or _extract_section(chunk, "Technical Skills")
        if skills_text:
            return f"{resume.name}'s skills: {skills_text}"
        if resume.skills:
            return f"{resume.name} has these skills: {', '.join(resume.skills)}"
        # Fall back to chunk which likely contains skills
        return f"From {resume.name}'s resume: {chunk[:400]}"
    
    elif any(word in question_lower for word in ["project", "built", "created", "developed", "made"]):
        # Look for projects section in chunk
        projects_text = _extract_section(chunk, "Projects") or _extract_section(chunk, "Key Projects")
        if projects_text:
            return f"{resume.name}'s projects: {projects_text}"
        if resume.projects:
            return f"{resume.name} has worked on these projects: {', '.join(resume.projects)}"
        # Search all chunks for project mentions
        for m in unique_matches:
            if "project" in m.chunk.lower():
                proj_text = _extract_section(m.chunk, "Projects")
                if proj_text:
                    return f"{m.resume.name}'s projects: {proj_text}"
                # Return project-related portion of chunk
                return f"{m.resume.name}'s projects: {m.chunk[:400]}"
        return f"From {resume.name}'s experience: {chunk[:400]}"
    
    elif any(word in question_lower for word in ["education", "degree", "university", "college", "study"]):
        edu_text = _extract_section(chunk, "Education")
        if edu_text:
            return f"{resume.name}'s education: {edu_text}"
        if resume.education:
            return f"{resume.name}'s education: {resume.education}"
        return f"Education information for {resume.name}: {chunk[:300]}"
    
    elif any(word in question_lower for word in ["who", "find", "candidate", "top", "best"]):
        # General candidate search - provide comprehensive info
        parts = [f"{resume.name}"]
        if resume.role:
            parts.append(f"({resume.role})")
        # Extract skills from chunk if not in resume
        skills_text = _extract_section(chunk, "Skills")
        if skills_text:
            parts.append(f"with {skills_text[:100]}")
        elif resume.skills:
            parts.append(f"with skills in {', '.join(resume.skills[:5])}")
        return f"Best match: {' '.join(parts)}. {chunk[:200]}"
    
    else:
        # Default: return chunk content which has full context
        role = resume.role or "professional"
        return f"{resume.name} ({role}): {chunk[:400]}"