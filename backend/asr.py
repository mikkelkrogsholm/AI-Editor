"""
Audio Speech Recognition (ASR) module for transcribing audio.

Since Ollama doesn't support audio models, this module provides integration
options for various ASR solutions.
"""
import subprocess
import json
import tempfile
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from .config import settings

logger = logging.getLogger(__name__)


class ASRTranscriber:
    """Base class for ASR transcription."""
    
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe audio file and return segments with timestamps."""
        raise NotImplementedError


class WhisperCppTranscriber(ASRTranscriber):
    """Transcriber using whisper.cpp for local, fast processing."""
    
    def __init__(self, model_path: str = None, model_size: str = None):
        self.model_size = model_size or settings.models.asr_model_size
        self.model_path = model_path or f"models/ggml-{self.model_size}.bin"
        self.whisper_cpp_path = "whisper.cpp/main"  # Path to whisper.cpp executable
        
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe using whisper.cpp."""
        try:
            # Create temporary output file
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
                output_path = tmp.name
            
            # Run whisper.cpp
            cmd = [
                self.whisper_cpp_path,
                "-m", self.model_path,
                "-f", audio_path,
                "--output-json",
                "--output-file", output_path.replace(".json", ""),
                "--language", "auto",
                "--max-len", "50",  # Max segment length
                "--split-on-word",  # Split on word boundaries
            ]
            
            logger.info(f"Running whisper.cpp: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.error(f"whisper.cpp failed: {result.stderr}")
                return []
            
            # Parse JSON output
            with open(output_path, 'r') as f:
                data = json.load(f)
            
            # Convert to our format
            segments = []
            for segment in data.get("transcription", []):
                segments.append({
                    "start": segment["timestamps"]["from"] / 1000.0,  # Convert ms to seconds
                    "end": segment["timestamps"]["to"] / 1000.0,
                    "text": segment["text"].strip(),
                    "speaker": "Unknown"  # whisper.cpp doesn't do speaker diarization
                })
            
            # Clean up
            Path(output_path).unlink(missing_ok=True)
            
            return segments
            
        except FileNotFoundError:
            logger.error("whisper.cpp not found. Please install from: https://github.com/ggerganov/whisper.cpp")
            return []
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return []


class FasterWhisperTranscriber(ASRTranscriber):
    """Transcriber using faster-whisper Python package."""
    
    def __init__(self, model_size: str = None, device: str = "auto"):
        self.model_size = model_size or settings.models.asr_model_size
        self.device = device
        self._model = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                from faster_whisper import WhisperModel
                self._model = WhisperModel(
                    self.model_size,
                    device=self.device,
                    compute_type="int8" if self.device == "cpu" else "float16"
                )
            except ImportError:
                logger.error("faster-whisper not installed. Run: pip install faster-whisper")
                raise
    
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe using faster-whisper."""
        try:
            self._load_model()
            
            segments_list = []
            segments, info = self._model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True  # Voice activity detection
            )
            
            logger.info(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")
            
            for segment in segments:
                segments_list.append({
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                    "speaker": "Unknown"
                })
            
            return segments_list
            
        except Exception as e:
            logger.error(f"faster-whisper transcription failed: {e}")
            return []


class OpenAIWhisperTranscriber(ASRTranscriber):
    """Transcriber using OpenAI's whisper Python package."""
    
    def __init__(self, model_size: str = None):
        self.model_size = model_size or settings.models.asr_model_size
        self._model = None
    
    def _load_model(self):
        """Lazy load the model."""
        if self._model is None:
            try:
                import whisper
                self._model = whisper.load_model(self.model_size)
            except ImportError:
                logger.error("openai-whisper not installed. Run: pip install openai-whisper")
                raise
    
    def transcribe(self, audio_path: str) -> List[Dict[str, Any]]:
        """Transcribe using openai-whisper."""
        try:
            self._load_model()
            
            result = self._model.transcribe(
                audio_path,
                word_timestamps=True,
                verbose=False
            )
            
            segments = []
            for segment in result["segments"]:
                segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "speaker": "Unknown"
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"openai-whisper transcription failed: {e}")
            return []


def get_transcriber(implementation: str = None) -> ASRTranscriber:
    """Factory function to get the appropriate transcriber."""
    if implementation is None:
        implementation = settings.models.asr_model
    
    if implementation == "whisper.cpp":
        return WhisperCppTranscriber()
    elif implementation == "faster-whisper":
        return FasterWhisperTranscriber()
    elif implementation == "openai-whisper":
        return OpenAIWhisperTranscriber()
    else:
        logger.warning(f"Unknown ASR implementation: {implementation}. Using whisper.cpp")
        return WhisperCppTranscriber()


# Example usage for development/testing
if __name__ == "__main__":
    # Test with a dummy audio file
    transcriber = get_transcriber("whisper.cpp")
    segments = transcriber.transcribe("test_audio.wav")
    
    for segment in segments:
        print(f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}")