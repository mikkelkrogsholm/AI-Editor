"""
Basic tests for AI-Klipperen.
"""
import pytest
import tempfile
from pathlib import Path
import numpy as np
from moviepy.editor import ColorClip

# Import our modules
from backend.vector import VectorStore
from backend.chat_tools import Storyboard, ClipSegment


def test_vector_store_init():
    """Test vector store initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_directory=tmpdir)
        assert store is not None
        assert store.frame_collection is not None
        assert store.asr_collection is not None


def test_add_and_search_frame():
    """Test adding and searching frames."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = VectorStore(persist_directory=tmpdir)
        
        # Add a test frame
        frame_id = "test_frame_001"
        embedding = np.random.rand(1024).tolist()
        
        store.add_frame(
            frame_id=frame_id,
            video_path="/test/video.mp4",
            timestamp="00:01:23.45",
            caption="A beautiful sunset over the ocean",
            tags=["sunset", "ocean", "nature"],
            mood="calm",
            quality=8.5,
            embedding=embedding
        )
        
        # Search for the frame
        results = store.search_frames(
            query_embedding=embedding,
            n_results=5
        )
        
        assert len(results) > 0
        assert results[0]["id"] == frame_id
        assert results[0]["caption"] == "A beautiful sunset over the ocean"


def test_storyboard_creation():
    """Test storyboard data model."""
    # Create clip segments
    clips = [
        ClipSegment(clip_id="clip1", in_point=0.0, out_point=3.0, transition="cut"),
        ClipSegment(clip_id="clip2", in_point=0.0, out_point=5.0, transition="fade"),
        ClipSegment(clip_id="clip3", in_point=2.0, out_point=6.0, transition="cut"),
    ]
    
    # Create storyboard
    storyboard = Storyboard(
        project_name="test_project",
        duration=12.0,
        timeline=clips,
        notes="Test storyboard"
    )
    
    assert storyboard.project_name == "test_project"
    assert storyboard.duration == 12.0
    assert len(storyboard.timeline) == 3
    assert storyboard.timeline[0].clip_id == "clip1"


def test_create_dummy_video():
    """Test creating a dummy video file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test_video.mp4"
        
        # Create a 3-second color clip
        clip = ColorClip(size=(640, 480), color=(255, 0, 0), duration=3)
        clip.write_videofile(str(output_path), fps=24, logger=None)
        
        assert output_path.exists()
        assert output_path.stat().st_size > 0


@pytest.mark.asyncio
async def test_api_root():
    """Test API root endpoint."""
    from backend.main import app
    from httpx import AsyncClient
    from fastapi.testclient import TestClient
    
    # Use TestClient for testing FastAPI
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["name"] == "AI-Klipperen API"


def test_cli_import():
    """Test CLI can be imported."""
    from cli import app as cli_app
    assert cli_app is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])