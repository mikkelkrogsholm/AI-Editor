"""
Tests for video processing pipeline.
"""
import pytest
from pathlib import Path
import tempfile
import numpy as np

from backend.pipeline import OllamaClient, VideoPipeline
from backend.vector import VectorStore


class MockOllamaClient:
    """Mock Ollama client for testing."""
    
    def generate_caption(self, image_base64: str, model: str = "llava:latest") -> str:
        return "Test caption for video frame"
    
    def generate_embedding(self, text: str, model: str = "nomic-embed-text:latest") -> list[float]:
        return np.random.rand(768).tolist()
    
    def transcribe_audio(self, audio_path: str, model: str = "whisper:base") -> list[dict]:
        return [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Test transcription",
                "speaker": "Speaker1"
            }
        ]


def test_ollama_client_init():
    """Test Ollama client initialization."""
    client = OllamaClient()
    assert client.base_url == "http://localhost:11434"
    assert client.client is not None


def test_video_pipeline_init():
    """Test video pipeline initialization."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = VectorStore(persist_directory=tmpdir)
        ollama_client = MockOllamaClient()
        
        pipeline = VideoPipeline(vector_store, ollama_client)
        assert pipeline.vector_store is not None
        assert pipeline.ollama is not None
        assert pipeline.temp_dir.exists()


def test_analyze_frame():
    """Test frame analysis."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = VectorStore(persist_directory=tmpdir)
        ollama_client = MockOllamaClient()
        pipeline = VideoPipeline(vector_store, ollama_client)
        
        # Create a dummy image
        from PIL import Image
        img = Image.new('RGB', (100, 100), color='red')
        img_path = Path(tmpdir) / "test.jpg"
        img.save(img_path)
        
        # Analyze frame
        result = pipeline.analyze_frame(img_path)
        
        assert "caption" in result
        assert "tags" in result
        assert "mood" in result
        assert "quality" in result
        assert isinstance(result["tags"], list)
        assert isinstance(result["quality"], float)


def test_extract_tags():
    """Test tag extraction from caption."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = VectorStore(persist_directory=tmpdir)
        ollama_client = MockOllamaClient()
        pipeline = VideoPipeline(vector_store, ollama_client)
        
        caption = "A crowd of people dancing at sunset during a concert"
        tags = pipeline._extract_tags(caption)
        
        assert "crowd" in tags
        assert "sunset" in tags
        assert "concert" in tags
        assert len(tags) <= 5


def test_detect_mood():
    """Test mood detection from caption."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = VectorStore(persist_directory=tmpdir)
        ollama_client = MockOllamaClient()
        pipeline = VideoPipeline(vector_store, ollama_client)
        
        # Test energetic mood
        caption = "Energetic crowd jumping and dancing"
        mood = pipeline._detect_mood(caption)
        assert mood == "energetic"
        
        # Test calm mood
        caption = "Peaceful sunset over calm waters"
        mood = pipeline._detect_mood(caption)
        assert mood == "calm"
        
        # Test neutral mood
        caption = "A person walking down the street"
        mood = pipeline._detect_mood(caption)
        assert mood == "neutral"


def test_assess_quality():
    """Test quality assessment from caption."""
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = VectorStore(persist_directory=tmpdir)
        ollama_client = MockOllamaClient()
        pipeline = VideoPipeline(vector_store, ollama_client)
        
        # Test high quality
        caption = "Clear, sharp, and well-composed shot"
        quality = pipeline._assess_quality(caption)
        assert quality > 6.0
        
        # Test low quality
        caption = "Blurry and dark footage"
        quality = pipeline._assess_quality(caption)
        assert quality < 5.0