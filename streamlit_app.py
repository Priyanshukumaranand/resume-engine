from __future__ import annotations

import os
from uuid import uuid4

import requests
import streamlit as st
from pypdf import PdfReader

API_BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Hackathon Teammate Finder", layout="wide")
st.title("Hackathon Teammate Finder Demo")


def _post(path: str, payload: dict):
    response = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def _get(path: str):
    response = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _delete(path: str):
    response = requests.delete(f"{API_BASE_URL}{path}", timeout=30)
    response.raise_for_status()
    return response.json()


def _extract_text_from_pdf(uploaded_file) -> str:
    if uploaded_file is None:
        raise ValueError("No PDF supplied")
    uploaded_file.seek(0)
    reader = PdfReader(uploaded_file)
    pages = [page.extract_text() or "" for page in reader.pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError("Unable to extract text from PDF")
    return text


def _summarize_text(text: str, max_chars: int = 600) -> str:
    compact = " ".join(text.split())
    return compact[:max_chars]


with st.expander("Add resume", expanded=True):
    with st.form("resume_form"):
        resume_id_input = st.text_input("ID", placeholder="student-001")
        name = st.text_input("Name")
        email = st.text_input("Email")
        role = st.text_input("Preferred Role (optional)")
        skills = st.text_input("Skills (comma separated, optional)")
        pdf_file = st.file_uploader("Resume PDF", type=["pdf"], accept_multiple_files=False)
        extracted_text_preview = ""
        if pdf_file is not None:
            try:
                extracted_text_preview = _extract_text_from_pdf(pdf_file)
                st.text_area(
                    "Extracted text (read-only)",
                    value=extracted_text_preview[:2000],
                    height=200,
                    disabled=True,
                )
            except ValueError as exc:
                st.error(str(exc))

        submitted = st.form_submit_button("Store resume")

        if submitted:
            if pdf_file is None:
                st.error("Please upload a resume PDF before submitting.")
                st.stop()
            try:
                full_text = extracted_text_preview or _extract_text_from_pdf(pdf_file)
            except ValueError as exc:
                st.error(str(exc))
                st.stop()

            resume_id = resume_id_input.strip() or f"resume-{uuid4().hex[:8]}"
            name_clean = name.strip()
            email_clean = email.strip()
            role_clean = role.strip()
            if not name_clean or not email_clean:
                st.error("Name and email are required.")
                st.stop()

            experience = full_text
            summary = _summarize_text(full_text)
            payload = {
                "id": resume_id,
                "name": name_clean,
                "email": email_clean,
                "role": role_clean,
                "skills": [skill.strip() for skill in skills.split(",") if skill.strip()],
                "experience": experience,
                "summary": summary,
            }
            try:
                data = _post("/add_resume", payload)
                st.success(f"Stored resume {data['id']}")
            except requests.HTTPError as exc:
                st.error(f"Failed: {exc.response.text}")

with st.expander("Find a teammate", expanded=True):
    with st.form("recommend_form"):
        role = st.text_input("Desired Role", placeholder="Backend Engineer")
        skills = st.text_input("Important Skills (comma separated)")
        summary = st.text_area("Problem / Project Context")
        top_k = st.slider("Candidates to consider", 1, 20, 5)
        submitted = st.form_submit_button("Recommend teammate")

        if submitted:
            payload = {
                "role": role,
                "skills": [skill.strip() for skill in skills.split(",") if skill.strip()],
                "summary": summary,
                "top_k": top_k,
            }
            try:
                result = _post("/recommend", payload)
                best = result["best_match"]
                st.subheader("Best candidate")
                st.write(best["candidate"])
                st.write({
                    "match_score": best["match_score"],
                    "matching_skills": best["matching_skills"],
                    "explanation": best["explanation"],
                })
            except requests.HTTPError as exc:
                st.error(f"Failed: {exc.response.text}")

with st.expander("Stored resumes"):
    try:
        resumes = _get("/resumes").get("resumes", [])
        if not resumes:
            st.write("No resumes yet")
        else:
            for resume in resumes:
                cols = st.columns([4, 1])
                with cols[0]:
                    st.write(resume)
                with cols[1]:
                    if st.button(
                        "Delete",
                        key=f"delete-{resume['id']}",
                    ):
                        try:
                            _delete(f"/resumes/{resume['id']}")
                            st.success(f"Deleted {resume['id']}")
                            st.experimental_rerun()
                        except requests.HTTPError as exc:
                            st.error(f"Failed to delete: {exc.response.text}")
    except requests.HTTPError as exc:
        st.error(f"Could not load resumes: {exc.response.text}")
