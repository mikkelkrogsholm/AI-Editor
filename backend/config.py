"""
Configuration settings for AI-Klipperen backend.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import os


class ModelConfig(BaseSettings):
    """Model configuration for different AI tasks."""
    
    # Vision models
    vision_caption_model: str = Field(
        default="minicpm-v:8b-2.6-q4_0",
        description="Model for generating image captions from video frames"
    )
    vision_caption_fallback: str = Field(
        default="llava:latest",
        description="Fallback model if primary vision model is not available"
    )
    
    # Embedding models
    text_embedding_model: str = Field(
        default="snowflake-arctic-embed2:latest",
        description="Model for generating text embeddings (1024-dim)"
    )
    image_embedding_model: str = Field(
        default="snowflake-arctic-embed2:latest",
        description="Model for generating image embeddings from text (1024-dim)"
    )
    embedding_dimension: int = Field(
        default=1024,
        description="Default dimension of embedding vectors (snowflake-arctic-embed2 uses 1024)"
    )
    
    # Language models
    chat_model: str = Field(
        default="deepseek-r1:32b",
        description="Model for chat interactions and storyboard generation"
    )
    chat_model_fallback: str = Field(
        default="mistral:latest",
        description="Fallback chat model"
    )
    
    # Audio models (Note: Ollama doesn't support audio models yet)
    asr_model: str = Field(
        default="whisper.cpp",
        description="ASR implementation to use (whisper.cpp recommended for local processing)"
    )
    asr_model_size: str = Field(
        default="base",
        description="Whisper model size: tiny, base, small, medium, large"
    )
    
    model_config = {
        "env_prefix": "AI_CLIP_MODEL_",
        "env_file": ".env"
    }


class ProcessingConfig(BaseSettings):
    """Processing configuration settings."""
    
    # Frame extraction
    frame_extraction_fps: float = Field(
        default=1.0,
        description="Frames per second to extract from videos"
    )
    frame_quality: int = Field(
        default=2,
        ge=1,
        le=31,
        description="JPEG quality for extracted frames (1-31, lower is better)"
    )
    
    # Audio processing
    audio_sample_rate: int = Field(
        default=16000,
        description="Sample rate for audio extraction"
    )
    audio_channels: int = Field(
        default=1,
        description="Number of audio channels (1=mono, 2=stereo)"
    )
    
    # Quality thresholds
    default_quality_threshold: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="Default minimum quality score for search results"
    )
    
    # Batch processing
    batch_size: int = Field(
        default=10,
        description="Number of frames to process in parallel"
    )
    max_workers: int = Field(
        default=4,
        description="Maximum number of worker threads"
    )
    
    model_config = {
        "env_prefix": "AI_CLIP_PROCESS_",
        "env_file": ".env"
    }


class StorageConfig(BaseSettings):
    """Storage configuration settings."""
    
    # Directories
    chroma_persist_dir: str = Field(
        default="./chroma_db",
        description="Directory for ChromaDB persistence"
    )
    temp_dir: Optional[str] = Field(
        default=None,
        description="Temporary directory for processing (uses system temp if not set)"
    )
    upload_dir: str = Field(
        default="./uploads",
        description="Directory for uploaded files"
    )
    output_dir: str = Field(
        default="./outputs",
        description="Directory for rendered videos"
    )
    
    # File limits
    max_upload_size: int = Field(
        default=5 * 1024 * 1024 * 1024,  # 5GB
        description="Maximum upload file size in bytes"
    )
    
    model_config = {
        "env_prefix": "AI_CLIP_STORAGE_",
        "env_file": ".env"
    }


class OllamaConfig(BaseSettings):
    """Ollama server configuration."""
    
    host: str = Field(
        default="http://localhost:11434",
        description="Ollama server URL"
    )
    timeout: float = Field(
        default=300.0,
        description="Request timeout in seconds"
    )
    max_retries: int = Field(
        default=3,
        description="Maximum number of retries for failed requests"
    )
    
    model_config = {
        "env_prefix": "OLLAMA_",
        "env_file": ".env"
    }


class RenderConfig(BaseSettings):
    """Video rendering configuration."""
    
    # Preview settings
    preview_resolution: Tuple[int, int] = Field(
        default=(640, 360),
        description="Resolution for preview renders"
    )
    preview_fps: int = Field(
        default=24,
        description="FPS for preview renders"
    )
    preview_watermark: bool = Field(
        default=True,
        description="Add watermark to preview renders"
    )
    
    # Final render settings
    final_resolution: Tuple[int, int] = Field(
        default=(1920, 1080),
        description="Resolution for final renders"
    )
    final_fps: int = Field(
        default=30,
        description="FPS for final renders"
    )
    final_bitrate: str = Field(
        default="10M",
        description="Bitrate for final renders"
    )
    final_preset: str = Field(
        default="slow",
        description="FFmpeg preset for final renders (ultrafast, fast, medium, slow)"
    )
    
    # Codec settings
    video_codec: str = Field(
        default="libx264",
        description="Video codec to use"
    )
    audio_codec: str = Field(
        default="aac",
        description="Audio codec to use"
    )
    audio_bitrate: str = Field(
        default="192k",
        description="Audio bitrate"
    )
    
    # Transition settings
    default_transition_duration: float = Field(
        default=1.0,
        description="Default transition duration in seconds"
    )
    
    model_config = {
        "env_prefix": "AI_CLIP_RENDER_",
        "env_file": ".env"
    }


class Settings(BaseSettings):
    """Main settings class that combines all configurations."""
    
    # API settings
    api_title: str = Field(
        default="AI-Klipperen API",
        description="API title"
    )
    api_version: str = Field(
        default="1.0.0",
        description="API version"
    )
    
    # Sub-configurations
    models: ModelConfig = Field(default_factory=ModelConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    render: RenderConfig = Field(default_factory=RenderConfig)
    
    # Feature flags
    enable_asr: bool = Field(
        default=False,
        description="Enable audio transcription (requires Whisper model)"
    )
    enable_gpu: bool = Field(
        default=True,
        description="Enable GPU acceleration if available"
    )
    
    model_config = {
        "env_prefix": "AI_CLIP_",
        "env_file": ".env"
    }
        
    def save_to_file(self, path: str = "config.json"):
        """Save current configuration to JSON file."""
        import json
        with open(path, 'w') as f:
            json.dump(self.model_dump(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, path: str = "config.json") -> "Settings":
        """Load configuration from JSON file."""
        import json
        if Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
                return cls(**data)
        return cls()


# Global settings instance
settings = Settings()


# Helper function to get model with fallback
def get_model_with_fallback(primary: str, fallback: str, available_models: list[str]) -> str:
    """Get model name with fallback if primary is not available."""
    if primary in available_models:
        return primary
    elif fallback in available_models:
        return fallback
    elif available_models:
        return available_models[0]
    else:
        raise ValueError(f"No models available. Tried {primary} and {fallback}")


# Example usage:
if __name__ == "__main__":
    # Print current configuration
    print("Current Configuration:")
    print(f"Vision Model: {settings.models.vision_caption_model}")
    print(f"Chat Model: {settings.models.chat_model}")
    print(f"Frame Extraction FPS: {settings.processing.frame_extraction_fps}")
    print(f"Preview Resolution: {settings.render.preview_resolution}")
    
    # Save to file
    settings.save_to_file("ai_clip_config.json")
    
    # Load from file
    loaded_settings = Settings.load_from_file("ai_clip_config.json")
    print("\nLoaded settings match:", settings == loaded_settings)