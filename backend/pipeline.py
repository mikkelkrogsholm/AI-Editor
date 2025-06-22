"""
Video processing pipeline for frame extraction, captioning, ASR, and embeddings.
"""
import ffmpeg
import numpy as np
from pathlib import Path
import tempfile
import shutil
import json
import httpx
from typing import List, Dict, Any, Optional, Tuple, Generator
from PIL import Image
import io
import base64
from tqdm import tqdm
import asyncio
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import logging

from .vector import VectorStore
from .config import settings, get_model_with_fallback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama API."""
    
    def __init__(self, base_url: str = None):
        self.base_url = base_url or settings.ollama.host
        self.client = httpx.Client(timeout=settings.ollama.timeout)
        self._available_models = None
        self._embedding_dimensions = {}
    
    def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama."""
        if self._available_models is None:
            try:
                response = self.client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    self._available_models = [m["name"] for m in models]
                else:
                    self._available_models = []
            except Exception as e:
                logger.error(f"Failed to get available models: {e}")
                self._available_models = []
        return self._available_models
    
    def generate_caption(self, image_base64: str, model: str = None) -> str:
        """Generate caption for an image using vision model."""
        if model is None:
            available = self.get_available_models()
            model = get_model_with_fallback(
                settings.models.vision_caption_model,
                settings.models.vision_caption_fallback,
                available
            )
        
        prompt = "Describe this video frame in detail. Focus on: people, actions, objects, setting, mood, and visual quality. Be concise but comprehensive."
        
        response = self.client.post(
            f"{self.base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "images": [image_base64],
                "stream": False
            }
        )
        
        if response.status_code == 200:
            return response.json()["response"]
        else:
            logger.error(f"Caption generation failed: {response.text}")
            return "Failed to generate caption"
    
    def generate_embedding(self, text: str, model: str = None) -> List[float]:
        """Generate text embedding using Ollama."""
        if model is None:
            available = self.get_available_models()
            model = get_model_with_fallback(
                settings.models.text_embedding_model,
                settings.models.text_embedding_model,  # Use same as fallback
                available
            )
        
        response = self.client.post(
            f"{self.base_url}/api/embeddings",
            json={
                "model": model,
                "prompt": text
            }
        )
        
        if response.status_code == 200:
            embedding = response.json()["embedding"]
            # Cache the dimension for this model
            if model not in self._embedding_dimensions:
                self._embedding_dimensions[model] = len(embedding)
                logger.info(f"Model {model} has embedding dimension: {len(embedding)}")
            return embedding
        else:
            logger.error(f"Embedding generation failed: {response.text}")
            # Try to use cached dimension or default
            dim = self._embedding_dimensions.get(model, settings.models.embedding_dimension)
            return [0.0] * dim  # Return zero vector as fallback
    
    def get_embedding_dimension(self, model: str = None) -> int:
        """Get the embedding dimension for a model."""
        if model is None:
            model = settings.models.text_embedding_model
        
        # Check cache first
        if model in self._embedding_dimensions:
            return self._embedding_dimensions[model]
        
        # Generate a test embedding to get dimension
        try:
            embedding = self.generate_embedding("test", model)
            return len(embedding)
        except:
            return settings.models.embedding_dimension
    
    def transcribe_audio(self, audio_path: str, model: str = "whisper:base") -> List[Dict[str, Any]]:
        """Transcribe audio using Whisper model (placeholder for actual implementation)."""
        # Note: Ollama doesn't directly support audio models yet
        # This is a placeholder that would need integration with whisper.cpp or similar
        return [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Placeholder transcription",
                "speaker": "Unknown"
            }
        ]


class VideoPipeline:
    """Main pipeline for processing videos."""
    
    def __init__(self, vector_store: VectorStore, ollama_client: OllamaClient):
        self.vector_store = vector_store
        self.ollama = ollama_client
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ai_clip_"))
    
    def __del__(self):
        """Cleanup temporary directory."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def extract_frames(self, video_path: str, fps: float = None) -> Generator[Tuple[float, Path], None, None]:
        """Extract frames from video at specified FPS."""
        if fps is None:
            fps = settings.processing.frame_extraction_fps
            
        video_path = Path(video_path)
        output_pattern = str(self.temp_dir / f"{video_path.stem}_frame_%06d.jpg")
        
        try:
            # Get video info
            probe = ffmpeg.probe(str(video_path))
            video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
            duration = float(probe['format']['duration'])
            
            # Extract frames
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.filter(stream, 'fps', fps=fps)
            stream = ffmpeg.output(stream, output_pattern, **{'qscale:v': settings.processing.frame_quality})
            ffmpeg.run(stream, quiet=True, overwrite_output=True)
            
            # Yield frames with timestamps
            frame_files = sorted(self.temp_dir.glob(f"{video_path.stem}_frame_*.jpg"))
            for i, frame_path in enumerate(frame_files):
                timestamp = i / fps
                yield timestamp, frame_path
                
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            raise
    
    def extract_audio(self, video_path: str) -> Path:
        """Extract audio from video as 16kHz WAV."""
        video_path = Path(video_path)
        audio_path = self.temp_dir / f"{video_path.stem}_audio.wav"
        
        try:
            stream = ffmpeg.input(str(video_path))
            stream = ffmpeg.output(
                stream,
                str(audio_path),
                acodec='pcm_s16le',
                ar=16000,
                ac=1
            )
            ffmpeg.run(stream, quiet=True, overwrite_output=True)
            return audio_path
            
        except Exception as e:
            logger.error(f"Audio extraction failed: {e}")
            raise
    
    def analyze_frame(self, frame_path: Path) -> Dict[str, Any]:
        """Analyze a single frame: caption, tags, mood, quality."""
        # Load and encode image
        with Image.open(frame_path) as img:
            # Resize for faster processing
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=85)
            image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        # Generate caption
        caption = self.ollama.generate_caption(image_base64)
        
        # Extract tags and mood from caption (simplified)
        tags = self._extract_tags(caption)
        mood = self._detect_mood(caption)
        quality = self._assess_quality(caption)
        
        return {
            "caption": caption,
            "tags": tags,
            "mood": mood,
            "quality": quality
        }
    
    def _extract_tags(self, caption: str) -> List[str]:
        """Extract tags from caption (simplified implementation)."""
        # Common objects and concepts
        tag_words = [
            "person", "people", "crowd", "face", "hand", "body",
            "sunset", "sunrise", "night", "day", "indoor", "outdoor",
            "stage", "concert", "performance", "music", "dance",
            "happy", "sad", "excited", "calm", "energetic"
        ]
        
        caption_lower = caption.lower()
        tags = []
        
        for word in tag_words:
            if word in caption_lower:
                tags.append(word)
        
        return tags[:5]  # Limit to 5 tags
    
    def _detect_mood(self, caption: str) -> str:
        """Detect mood from caption."""
        caption_lower = caption.lower()
        
        mood_indicators = {
            "energetic": ["energetic", "excited", "dynamic", "active", "vibrant"],
            "calm": ["calm", "peaceful", "serene", "quiet", "still"],
            "happy": ["happy", "joyful", "cheerful", "smiling", "celebration"],
            "dramatic": ["dramatic", "intense", "powerful", "striking"],
            "neutral": ["neutral", "normal", "regular"]
        }
        
        for mood, indicators in mood_indicators.items():
            if any(indicator in caption_lower for indicator in indicators):
                return mood
        
        return "neutral"
    
    def _assess_quality(self, caption: str) -> float:
        """Assess visual quality from caption (0-10 scale)."""
        caption_lower = caption.lower()
        
        # Quality indicators
        quality_score = 5.0  # Base score
        
        # Positive indicators
        if any(word in caption_lower for word in ["clear", "sharp", "detailed", "vibrant"]):
            quality_score += 2.0
        
        # Negative indicators
        if any(word in caption_lower for word in ["blurry", "dark", "unclear", "grainy"]):
            quality_score -= 2.0
        
        # Composition indicators
        if any(word in caption_lower for word in ["well-composed", "centered", "balanced"]):
            quality_score += 1.0
        
        return max(0.0, min(10.0, quality_score))
    
    async def process_video(self, video_path: str, project_name: str) -> Dict[str, Any]:
        """Process entire video through the pipeline."""
        video_path = Path(video_path)
        stats = {
            "video": str(video_path),
            "project": project_name,
            "frames_processed": 0,
            "asr_segments": 0,
            "errors": []
        }
        
        try:
            logger.info(f"Processing video: {video_path}")
            
            # Extract frames
            logger.info("Extracting frames...")
            frames = list(self.extract_frames(str(video_path), fps=1.0))
            
            # Extract audio
            logger.info("Extracting audio...")
            audio_path = self.extract_audio(str(video_path))
            
            # Process frames
            logger.info(f"Processing {len(frames)} frames...")
            for timestamp, frame_path in tqdm(frames, desc="Analyzing frames"):
                try:
                    # Analyze frame
                    analysis = self.analyze_frame(frame_path)
                    
                    # Generate embedding
                    embedding = self.ollama.generate_embedding(analysis["caption"])
                    
                    # Store in vector DB
                    frame_id = f"{project_name}_{video_path.stem}_f{int(timestamp*100):06d}"
                    self.vector_store.add_frame(
                        frame_id=frame_id,
                        video_path=str(video_path),
                        timestamp=f"{int(timestamp//60):02d}:{int(timestamp%60):02d}.{int((timestamp%1)*100):02d}",
                        caption=analysis["caption"],
                        tags=analysis["tags"],
                        mood=analysis["mood"],
                        quality=analysis["quality"],
                        embedding=embedding
                    )
                    
                    stats["frames_processed"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process frame at {timestamp}: {e}")
                    stats["errors"].append(f"Frame {timestamp}: {str(e)}")
            
            # Process audio (placeholder)
            logger.info("Processing audio...")
            asr_segments = self.ollama.transcribe_audio(str(audio_path))
            
            for i, segment in enumerate(asr_segments):
                try:
                    # Generate embedding for text
                    embedding = self.ollama.generate_embedding(segment["text"])
                    
                    # Store in vector DB
                    segment_id = f"{project_name}_{video_path.stem}_a{i:04d}"
                    self.vector_store.add_asr_segment(
                        segment_id=segment_id,
                        video_path=str(video_path),
                        start_time=segment["start"],
                        end_time=segment["end"],
                        text=segment["text"],
                        speaker=segment["speaker"],
                        embedding=embedding
                    )
                    
                    stats["asr_segments"] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process ASR segment {i}: {e}")
                    stats["errors"].append(f"ASR segment {i}: {str(e)}")
            
            logger.info(f"Processing complete: {stats['frames_processed']} frames, {stats['asr_segments']} ASR segments")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            stats["errors"].append(f"Pipeline error: {str(e)}")
        
        return stats
    
    def process_directory(self, directory: str, project_name: str, extensions: List[str] = None) -> List[Dict[str, Any]]:
        """Process all videos in a directory."""
        if extensions is None:
            extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
        
        directory = Path(directory)
        video_files = []
        
        for ext in extensions:
            video_files.extend(directory.glob(f"*{ext}"))
            video_files.extend(directory.glob(f"*{ext.upper()}"))
        
        results = []
        
        for video_file in video_files:
            logger.info(f"Processing: {video_file}")
            result = asyncio.run(self.process_video(str(video_file), project_name))
            results.append(result)
        
        return results