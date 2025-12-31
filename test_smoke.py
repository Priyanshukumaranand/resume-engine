import os
import pytest
from unittest.mock import MagicMock, patch

# Set dummy env vars before importing main to bypass dependency init checks
os.environ["GEMINI_API_KEY"] = "dummy_key"
os.environ["HUGGINGFACE_API_TOKEN"] = "dummy_token"

# Now import app
from backend.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.fixture(autouse=True)
def mock_dependencies():
    """Mock out the heavy/external dependencies in backend.main"""
    with patch("backend.main.resume_llm") as mock_llm, \
         patch("backend.main.embedder") as mock_embedder, \
         patch("backend.main.vector_store") as mock_vs:
        
        # Mock LLM extraction
        mock_llm.extract_resume_fields.return_value = {
            "name": "Test User",
            "email": "test@example.com",
            "role": "Engineer",
            "skills": ["python", "testing"],
            "summary": "Summary",
            "experience": "Experience"
        }
        mock_llm.answer_question.return_value = "This is a mocked answer."

        # Mock Embedder
        mock_embedder.embed_text.return_value = [0.1, 0.2, 0.3]
        
        # Mock Vector Store
        mock_vs.has_resumes.return_value = True
        mock_vs.get_all_resumes.return_value = []
        mock_vs.query.return_value = [] # Return empty list or mocked matches
        
        yield

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_ingest_pdf_mock():
    # Create a dummy PDF file content
    # In a real scenario we might want a valid minimal PDF binary, 
    # but since main.py calls _extract_pdf_text which calls PyPDF, 
    # we might also want to mock _extract_pdf_text to avoid needing a real PDF.
    
    with patch("backend.main._extract_pdf_text", return_value="Mock PDF content"):
        files = {"file": ("resume.pdf", b"dummy content", "application/pdf")}
        response = client.post("/ingest_pdf", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "created"
        assert "id" in data

def test_list_resumes():
    response = client.get("/resumes")
    assert response.status_code == 200
    assert "resumes" in response.json()

def test_qa_flow():
    # minimal Setup for success
    # We need vector_store.query to return something so we don't get 404 "No relevant resumes found"
    # wrapper around the auto-fixture if we need specific return values
    from backend.main import vector_store
    
    # Mock return so it looks like we found a match
    mock_match = (MagicMock(id="1", name="Alice", role="Dev", skills=[], summary="Sum", experience="Exp"), "chunk text", 0.9)
    vector_store.query.return_value = [mock_match]
    
    payload = {"question": "Who knows Python?", "top_k": 3}
    response = client.post("/qa", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["sources"]) > 0
