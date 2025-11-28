from __future__ import annotations

import json
from typing import List, Tuple

from transformers import pipeline


class ResumeLLM:
    """Lightweight helper that infers role and skills using a local seq2seq model."""

    def __init__(self, model_name: str = "google/flan-t5-base") -> None:
        self.model_name = model_name
        self._pipeline = None

    def _pipeline_instance(self):
        if self._pipeline is None:
            self._pipeline = pipeline(
                "text2text-generation",
                model=self.model_name,
                tokenizer=self.model_name,
            )
        return self._pipeline

    def infer_role_and_skills(self, resume_text: str) -> Tuple[str | None, List[str]]:
        text = (resume_text or "").strip()
        if not text:
            return None, []
        generator = self._pipeline_instance()
        prompt = (
            "Extract the primary hackathon role and up to eight key technical skills from the resume text. "
            "Respond with compact JSON using keys 'role' and 'skills'. Resume:\n"
            f"{text}"
        )
        output = generator(prompt, max_new_tokens=160, num_return_sequences=1)[0]["generated_text"]
        role, skills = self._parse_response(output)
        return role, skills

    def _parse_response(self, response: str) -> Tuple[str | None, List[str]]:
        response = response.strip()
        if not response:
            return None, []
        role = None
        skills: List[str] = []
        try:
            data = json.loads(response)
            role = self._clean_role(data.get("role")) if isinstance(data, dict) else None
            raw_skills = data.get("skills") if isinstance(data, dict) else []
            skills = self._clean_skills(raw_skills)
            if role or skills:
                return role, skills
        except Exception:
            pass

        lowered = response.lower()
        if "skills" in lowered and "role" in lowered:
            role_hint = response.split("role")[-1].split("skills")[0]
            role = self._clean_role(role_hint.split(":")[-1])
            skills_part = response.split("skills")[-1]
            skills = self._clean_skills(skills_part.replace(":", "").replace("-", ","))
        else:
            role = self._clean_role(response.split("skills")[0])

        return role, skills

    @staticmethod
    def _clean_role(value) -> str | None:
        if not value:
            return None
        text = str(value).strip().strip("\"'")
        return text or None

    @staticmethod
    def _clean_skills(value) -> List[str]:
        if not value:
            return []
        if isinstance(value, str):
            candidates = value.split(",")
        else:
            candidates = value
        cleaned = []
        for item in candidates:
            text = str(item).strip()
            if text and text not in cleaned:
                cleaned.append(text)
        return cleaned[:8]
