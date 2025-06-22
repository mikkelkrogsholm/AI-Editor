"""
Media analysis module for extracting technical metadata from video and image files.
"""
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class MediaAnalyzer:
    """Analyzes media files to extract technical metadata."""
    
    # Common aspect ratios
    ASPECT_RATIOS = {
        (16, 9): "16:9",
        (9, 16): "9:16",
        (4, 3): "4:3",
        (3, 4): "3:4",
        (1, 1): "1:1",
        (21, 9): "21:9",
        (4, 5): "4:5",
        (5, 4): "5:4",
        (2.39, 1): "2.39:1"
    }
    
    # Resolution names
    RESOLUTIONS = {
        (640, 480): "480p",
        (1280, 720): "720p",
        (1920, 1080): "1080p",
        (2560, 1440): "1440p",
        (3840, 2160): "4K",
        (7680, 4320): "8K"
    }
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract comprehensive video metadata using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(video_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            # Extract video stream info
            video_stream = None
            audio_stream = None
            
            for stream in data.get('streams', []):
                if stream['codec_type'] == 'video' and video_stream is None:
                    video_stream = stream
                elif stream['codec_type'] == 'audio' and audio_stream is None:
                    audio_stream = stream
            
            if not video_stream:
                raise ValueError("No video stream found")
            
            # Extract metadata
            width = int(video_stream['width'])
            height = int(video_stream['height'])
            aspect_ratio, orientation = self._calculate_aspect_ratio(width, height)
            
            # Parse frame rate
            fps = None
            if 'r_frame_rate' in video_stream:
                fps_parts = video_stream['r_frame_rate'].split('/')
                if len(fps_parts) == 2 and int(fps_parts[1]) > 0:
                    fps = int(fps_parts[0]) / int(fps_parts[1])
            
            # Parse duration
            duration = float(data['format'].get('duration', 0))
            
            metadata = {
                "width": width,
                "height": height,
                "aspect_ratio": aspect_ratio,
                "aspect_ratio_decimal": round(width / height, 3),
                "orientation": orientation,
                "resolution_name": self._get_resolution_name(width, height),
                "fps": round(fps, 2) if fps else None,
                "duration": round(duration, 2),
                "codec": video_stream.get('codec_name'),
                "pixel_format": video_stream.get('pix_fmt'),
                "bitrate": data['format'].get('bit_rate'),
                "size": int(data['format'].get('size', 0)),
                "has_audio": audio_stream is not None
            }
            
            # Add audio metadata if present
            if audio_stream:
                metadata.update({
                    "audio_codec": audio_stream.get('codec_name'),
                    "audio_channels": audio_stream.get('channels'),
                    "audio_sample_rate": int(audio_stream.get('sample_rate', 0)),
                    "audio_bitrate": audio_stream.get('bit_rate')
                })
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to analyze video {video_path}: {e}")
            raise
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Extract comprehensive image metadata using PIL."""
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                aspect_ratio, orientation = self._calculate_aspect_ratio(width, height)
                
                # Get DPI (default to 72 if not present)
                dpi = img.info.get('dpi', (72, 72))
                if isinstance(dpi, tuple):
                    dpi = dpi[0]  # Use horizontal DPI
                
                metadata = {
                    "width": width,
                    "height": height,
                    "aspect_ratio": aspect_ratio,
                    "aspect_ratio_decimal": round(width / height, 3),
                    "orientation": orientation,
                    "resolution_name": self._get_resolution_name(width, height),
                    "format": img.format,
                    "mode": img.mode,
                    "has_transparency": img.mode in ('RGBA', 'LA', 'P') and 'transparency' in img.info,
                    "dpi": dpi,
                    "file_size": Path(image_path).stat().st_size
                }
                
                # Add color information
                if img.mode == 'RGB':
                    metadata["color_mode"] = "RGB"
                    metadata["channels"] = 3
                elif img.mode == 'RGBA':
                    metadata["color_mode"] = "RGBA"
                    metadata["channels"] = 4
                elif img.mode == 'L':
                    metadata["color_mode"] = "Grayscale"
                    metadata["channels"] = 1
                else:
                    metadata["color_mode"] = img.mode
                    metadata["channels"] = len(img.mode)
                
                return metadata
                
        except Exception as e:
            logger.error(f"Failed to analyze image {image_path}: {e}")
            raise
    
    def get_audio_info(self, audio_path: str) -> Dict[str, Any]:
        """Extract audio metadata using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(audio_path)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            # Find audio stream
            audio_stream = None
            for stream in data.get('streams', []):
                if stream['codec_type'] == 'audio':
                    audio_stream = stream
                    break
            
            if not audio_stream:
                raise ValueError("No audio stream found")
            
            metadata = {
                "codec": audio_stream.get('codec_name'),
                "channels": audio_stream.get('channels'),
                "sample_rate": int(audio_stream.get('sample_rate', 0)),
                "bitrate": audio_stream.get('bit_rate') or data['format'].get('bit_rate'),
                "duration": float(data['format'].get('duration', 0)),
                "size": int(data['format'].get('size', 0)),
                "format": data['format'].get('format_name')
            }
            
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to analyze audio {audio_path}: {e}")
            raise
    
    def _calculate_aspect_ratio(self, width: int, height: int) -> Tuple[str, str]:
        """Calculate aspect ratio and orientation from dimensions."""
        # Calculate greatest common divisor
        def gcd(a, b):
            while b:
                a, b = b, a % b
            return a
        
        # Simplify ratio
        divisor = gcd(width, height)
        simplified_width = width // divisor
        simplified_height = height // divisor
        
        # Check common ratios
        for ratio, name in self.ASPECT_RATIOS.items():
            if isinstance(ratio[0], float):
                # Handle decimal ratios like 2.39:1
                if abs(width / height - ratio[0] / ratio[1]) < 0.01:
                    aspect_ratio = name
                    break
            else:
                # Handle integer ratios
                if (simplified_width, simplified_height) == ratio:
                    aspect_ratio = name
                    break
        else:
            # Use simplified ratio if no match found
            aspect_ratio = f"{simplified_width}:{simplified_height}"
        
        # Determine orientation
        if width > height:
            orientation = "landscape"
        elif height > width:
            orientation = "portrait"
        else:
            orientation = "square"
        
        return aspect_ratio, orientation
    
    def _get_resolution_name(self, width: int, height: int) -> str:
        """Get common resolution name from dimensions."""
        # Check exact matches
        for dims, name in self.RESOLUTIONS.items():
            if (width, height) == dims or (height, width) == dims:
                return name
        
        # Check by height for standard resolutions
        if height == 720 or width == 720:
            return "720p"
        elif height == 1080 or width == 1080:
            return "1080p"
        elif height == 1440 or width == 1440:
            return "1440p"
        elif height == 2160 or width == 2160:
            return "4K"
        elif height == 4320 or width == 4320:
            return "8K"
        
        # Return dimensions if no standard name
        return f"{width}x{height}"
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze any media file and return metadata."""
        path = Path(file_path)
        
        # Video extensions
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm', '.flv', '.wmv', '.m4v'}
        # Image extensions
        image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.webp'}
        # Audio extensions
        audio_extensions = {'.mp3', '.wav', '.flac', '.aac', '.ogg', '.m4a', '.wma'}
        
        ext = path.suffix.lower()
        
        if ext in video_extensions:
            return {
                "type": "video",
                "metadata": self.get_video_info(str(path))
            }
        elif ext in image_extensions:
            return {
                "type": "image",
                "metadata": self.get_image_info(str(path))
            }
        elif ext in audio_extensions:
            return {
                "type": "audio",
                "metadata": self.get_audio_info(str(path))
            }
        else:
            raise ValueError(f"Unsupported file type: {ext}")


# Singleton instance
media_analyzer = MediaAnalyzer()