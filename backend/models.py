"""
Data models for AI-Klipperen.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import datetime


class MediaMetadata(BaseModel):
    """Technical metadata for media files."""
    # Common fields
    width: int
    height: int
    aspect_ratio: str
    aspect_ratio_decimal: float
    orientation: str
    resolution_name: str
    file_size: int
    
    # Video-specific fields
    fps: Optional[float] = None
    duration: Optional[float] = None
    codec: Optional[str] = None
    pixel_format: Optional[str] = None
    bitrate: Optional[str] = None
    has_audio: Optional[bool] = None
    
    # Audio fields
    audio_codec: Optional[str] = None
    audio_channels: Optional[int] = None
    audio_sample_rate: Optional[int] = None
    audio_bitrate: Optional[str] = None
    
    # Image-specific fields
    format: Optional[str] = None
    mode: Optional[str] = None
    color_mode: Optional[str] = None
    channels: Optional[int] = None
    has_transparency: Optional[bool] = None
    dpi: Optional[int] = None


class FrameMetadata(BaseModel):
    """Metadata for a video frame."""
    frame_id: str
    video_path: str
    project: str
    timestamp: str
    timestamp_seconds: float
    caption: str
    tags: List[str]
    mood: str
    quality: float
    clip_path: Optional[str] = None
    media_metadata: Optional[MediaMetadata] = None


class ClipMetadata(BaseModel):
    """Metadata for a video clip."""
    clip_id: str
    video_path: str
    project: str
    clip_path: str
    start_time: float
    end_time: float
    duration: float
    frame_count: int
    media_metadata: MediaMetadata


class ASRSegment(BaseModel):
    """ASR transcription segment."""
    segment_id: str
    video_path: str
    project: str
    start_time: float
    end_time: float
    text: str
    speaker: str = "Unknown"
    confidence: Optional[float] = None


class ProjectInfo(BaseModel):
    """Project information and statistics."""
    name: str
    created: datetime
    updated: datetime
    stats: Dict[str, Any] = Field(default_factory=dict)
    assets: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    """Request to ingest video files."""
    path: str
    project: str
    recursive: bool = False
    extensions: List[str] = Field(default_factory=lambda: ['.mp4', '.mov', '.avi', '.mkv', '.webm'])


class SearchRequest(BaseModel):
    """Request to search for clips."""
    query: str
    project: Optional[str] = None
    limit: int = 10
    quality_threshold: Optional[float] = None
    aspect_ratio: Optional[str] = None
    orientation: Optional[str] = None
    min_resolution: Optional[str] = None


class SearchResult(BaseModel):
    """Search result item."""
    frame_id: str
    video_path: str
    project: str
    timestamp: str
    timestamp_seconds: float
    caption: str
    tags: List[str]
    mood: str
    quality: float
    similarity: float
    clip_path: Optional[str] = None
    media_metadata: Optional[MediaMetadata] = None


class ImportAssetRequest(BaseModel):
    """Request to import an asset into a project."""
    file_path: str
    project: str
    asset_type: str = Field(..., regex="^(video|audio_music|audio_voiceover|still_logo|still_graphic|still_photo)$")


class ProjectProcessRequest(BaseModel):
    """Request to process all unprocessed videos in a project."""
    project: str
    force: bool = False  # Re-process even if already processed