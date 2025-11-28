from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


class Resume(BaseModel):
    id: str = Field(..., description="Unique identifier for the resume")
    name: str
    email: str
    role: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: str
    summary: str

    @field_validator("id", "name", "email", mode="before")
    @classmethod
    def _strip_and_validate(cls, value: str) -> str:
        if value is None:
            raise ValueError("Value cannot be null")
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("Value cannot be empty")
        return trimmed

    @field_validator("role", mode="before")
    @classmethod
    def _normalize_role(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        trimmed = value.strip()
        return trimmed or None

    @field_validator("skills", mode="before")
    @classmethod
    def _normalize_skills(cls, value):
        if not value:
            return []
        if isinstance(value, str):
            candidates = value.split(",")
        else:
            candidates = value
        cleaned = []
        for item in candidates:
            text = item.strip()
            if text:
                cleaned.append(text)
        return cleaned


class ResumeListResponse(BaseModel):
    resumes: List[Resume]


class RecommendationRequest(BaseModel):
    role: str
    skills: List[str] = Field(default_factory=list)
    summary: str = ""
    top_k: int = Field(default=5, ge=1, le=20, description="Number of candidates to evaluate before reranking")

    @field_validator("role", mode="before")
    @classmethod
    def _validate_role(cls, value: str) -> str:
        if value is None:
            raise ValueError("role is required")
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("role cannot be empty")
        return trimmed

    @field_validator("skills", mode="before")
    @classmethod
    def _normalize_req_skills(cls, value):
        if not value:
            return []
        if isinstance(value, str):
            candidates = value.split(",")
        else:
            candidates = value
        cleaned = []
        for item in candidates:
            text = item.strip()
            if text:
                cleaned.append(text)
        return cleaned


class RecommendationResult(BaseModel):
    candidate: Resume
    match_score: float
    matching_skills: List[str] = Field(default_factory=list)
    explanation: str


class RecommendationResponse(BaseModel):
    best_match: RecommendationResult
    considered: int
