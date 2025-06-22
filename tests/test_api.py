"""
Tests for FastAPI endpoints.
"""
import pytest
from fastapi.testclient import TestClient
import json
from pathlib import Path

from backend.main import app, vector_store

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint returns correct info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "AI-Klipperen API"
    assert data["version"] == "1.0.0"
    assert data["status"] == "running"


def test_ingest_endpoint():
    """Test video ingest endpoint."""
    request_data = {
        "project_name": "test_project",
        "video_paths": ["/fake/path/video.mp4"]
    }
    
    response = client.post("/ingest", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "job_id" in data
    assert "message" in data
    assert "1 videos" in data["message"]


def test_search_endpoint():
    """Test search endpoint with different search types."""
    # Test hybrid search
    request_data = {
        "query": "sunset beach",
        "n_results": 10,
        "search_type": "hybrid"
    }
    
    response = client.post("/search", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "frames" in data
    assert "asr_segments" in data


def test_chat_endpoint():
    """Test chat interaction endpoint."""
    request_data = {
        "message": "Generate a 30 second energetic montage",
        "project_name": "test_project"
    }
    
    response = client.post("/chat", json=request_data)
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "conversation_id" in data


def test_projects_endpoint():
    """Test projects listing endpoint."""
    response = client.get("/projects")
    assert response.status_code == 200
    data = response.json()
    assert "projects" in data
    assert isinstance(data["projects"], list)


def test_job_status_not_found():
    """Test job status for non-existent job."""
    response = client.get("/ingest/status/fake-job-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_render_status_not_found():
    """Test render status for non-existent job."""
    response = client.get("/render/status/fake-job-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"


def test_download_not_found():
    """Test download for non-existent job."""
    response = client.get("/download/fake-job-id")
    assert response.status_code == 404
    assert response.json()["detail"] == "Job not found"