"""
Video rendering module using MoviePy and FFmpeg.
"""
import ffmpeg
from moviepy.editor import *
from pathlib import Path
import json
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import timedelta

from .chat_tools import Storyboard, ClipSegment
from .vector import VectorStore

logger = logging.getLogger(__name__)


class VideoRenderer:
    """Render videos from storyboards using MoviePy and FFmpeg."""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ai_clip_render_"))
    
    def __del__(self):
        """Cleanup temporary directory."""
        if hasattr(self, 'temp_dir') and self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def _parse_timestamp(self, timestamp: str) -> float:
        """Convert timestamp string (MM:SS.MS) to seconds."""
        parts = timestamp.split(':')
        minutes = int(parts[0])
        seconds_parts = parts[1].split('.')
        seconds = int(seconds_parts[0])
        milliseconds = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
        
        return minutes * 60 + seconds + milliseconds / 100
    
    def _get_clip_info(self, clip_id: str) -> Dict[str, Any]:
        """Get clip information from vector store."""
        # Search for the specific frame
        results = self.vector_store.frame_collection.get(
            ids=[clip_id],
            include=["metadatas", "documents"]
        )
        
        if not results['ids']:
            raise ValueError(f"Clip not found: {clip_id}")
        
        metadata = results['metadatas'][0]
        return {
            "video_path": metadata["video"],
            "timestamp": self._parse_timestamp(metadata["timestamp"]),
            "caption": results['documents'][0] if results['documents'] else ""
        }
    
    def _create_clip(self, segment: ClipSegment) -> VideoFileClip:
        """Create a MoviePy clip from a segment definition."""
        try:
            # Get clip information
            clip_info = self._get_clip_info(segment.clip_id)
            
            # Load the video
            video = VideoFileClip(clip_info["video_path"])
            
            # Calculate actual timestamps
            source_start = clip_info["timestamp"] + segment.in_point
            source_end = clip_info["timestamp"] + segment.out_point
            
            # Extract subclip
            clip = video.subclip(source_start, min(source_end, video.duration))
            
            # Clean up the full video reference
            video.close()
            
            return clip
            
        except Exception as e:
            logger.error(f"Failed to create clip {segment.clip_id}: {e}")
            # Return a blank clip as fallback
            duration = segment.out_point - segment.in_point
            return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    
    def _apply_transition(self, clip1: VideoClip, clip2: VideoClip, transition_type: str, duration: float = 1.0) -> List[VideoClip]:
        """Apply transition between two clips."""
        if transition_type == "cut":
            return [clip1, clip2]
        
        elif transition_type == "fade":
            # Fade out clip1, fade in clip2
            clip1_fade = clip1.fx(vfx.fadeout, duration)
            clip2_fade = clip2.fx(vfx.fadein, duration)
            
            # Overlap the clips
            clip2_fade = clip2_fade.set_start(clip1.duration - duration)
            
            return [clip1_fade, clip2_fade]
        
        elif transition_type == "dissolve":
            # Cross-dissolve between clips
            clip1_fade = clip1.fx(vfx.fadeout, duration)
            clip2_fade = clip2.fx(vfx.fadein, duration)
            
            # Set timing for overlap
            clip2_fade = clip2_fade.set_start(clip1.duration - duration)
            
            return [clip1_fade, clip2_fade]
        
        else:
            # Default to cut
            return [clip1, clip2]
    
    def render_preview(
        self,
        storyboard: Storyboard,
        output_path: str,
        resolution: Tuple[int, int] = (640, 360),
        fps: int = 24,
        watermark: bool = True
    ) -> str:
        """Render a low-resolution preview of the storyboard."""
        try:
            logger.info(f"Rendering preview for {storyboard.project_name}")
            
            # Create clips from timeline
            clips = []
            previous_clip = None
            
            for i, segment in enumerate(storyboard.timeline):
                # Create the clip
                clip = self._create_clip(segment)
                
                # Resize for preview
                clip = clip.resize(resolution)
                
                # Apply transition if not the first clip
                if previous_clip and i > 0:
                    transition_type = storyboard.timeline[i-1].transition
                    transition_clips = self._apply_transition(
                        previous_clip, clip, transition_type
                    )
                    
                    # Replace the last clip with transition result
                    if len(clips) > 0:
                        clips.pop()
                    clips.extend(transition_clips[:-1])  # Add all but the last
                    previous_clip = transition_clips[-1]
                else:
                    clips.append(clip)
                    previous_clip = clip
            
            # Add the last clip
            if previous_clip and previous_clip not in clips:
                clips.append(previous_clip)
            
            # Composite all clips
            if not clips:
                logger.warning("No clips to render")
                # Create a blank video
                final_video = ColorClip(size=resolution, color=(0, 0, 0), duration=1)
            else:
                final_video = CompositeVideoClip(clips)
            
            # Add watermark if requested
            if watermark:
                watermark_text = TextClip(
                    "AI-KLIPPEREN PREVIEW",
                    fontsize=24,
                    color='white',
                    stroke_color='black',
                    stroke_width=2,
                    font='Arial'
                ).set_duration(final_video.duration)
                
                # Position in bottom right
                watermark_text = watermark_text.set_position(('right', 'bottom')).set_margin(10)
                
                final_video = CompositeVideoClip([final_video, watermark_text])
            
            # Add audio if music cue is specified
            if storyboard.music_cue and Path(storyboard.music_cue).exists():
                try:
                    audio = AudioFileClip(storyboard.music_cue)
                    audio = audio.subclip(0, min(audio.duration, final_video.duration))
                    final_video = final_video.set_audio(audio)
                except Exception as e:
                    logger.warning(f"Failed to add audio: {e}")
            
            # Write the preview
            final_video.write_videofile(
                output_path,
                fps=fps,
                codec='libx264',
                preset='ultrafast',
                threads=4
            )
            
            # Cleanup
            final_video.close()
            
            logger.info(f"Preview rendered: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Preview rendering failed: {e}")
            raise
    
    def render_final(
        self,
        storyboard: Storyboard,
        output_path: str,
        resolution: Tuple[int, int] = (1920, 1080),
        fps: int = 30,
        bitrate: str = "10M",
        preset: str = "slow"
    ) -> str:
        """Render the final high-quality video."""
        try:
            logger.info(f"Rendering final video for {storyboard.project_name}")
            
            # Create clips from timeline
            clips = []
            previous_clip = None
            
            for i, segment in enumerate(storyboard.timeline):
                # Create the clip
                clip = self._create_clip(segment)
                
                # Ensure proper resolution
                if clip.size != resolution:
                    clip = clip.resize(resolution)
                
                # Apply transition if not the first clip
                if previous_clip and i > 0:
                    transition_type = storyboard.timeline[i-1].transition
                    transition_clips = self._apply_transition(
                        previous_clip, clip, transition_type, duration=1.5
                    )
                    
                    # Replace the last clip with transition result
                    if len(clips) > 0:
                        clips.pop()
                    clips.extend(transition_clips[:-1])
                    previous_clip = transition_clips[-1]
                else:
                    clips.append(clip)
                    previous_clip = clip
            
            # Add the last clip
            if previous_clip and previous_clip not in clips:
                clips.append(previous_clip)
            
            # Composite all clips
            if not clips:
                logger.warning("No clips to render")
                final_video = ColorClip(size=resolution, color=(0, 0, 0), duration=1)
            else:
                final_video = CompositeVideoClip(clips)
            
            # Add voiceover if specified
            audio_clips = []
            
            if storyboard.voiceover:
                # This would integrate with TTS in a real implementation
                logger.info("Voiceover text provided, but TTS not implemented")
            
            # Add background music
            if storyboard.music_cue and Path(storyboard.music_cue).exists():
                try:
                    music = AudioFileClip(storyboard.music_cue)
                    music = music.subclip(0, min(music.duration, final_video.duration))
                    music = music.volumex(0.3)  # Reduce volume for background
                    audio_clips.append(music)
                except Exception as e:
                    logger.warning(f"Failed to add music: {e}")
            
            # Composite audio if we have any
            if audio_clips:
                final_audio = CompositeAudioClip(audio_clips)
                final_video = final_video.set_audio(final_audio)
            
            # Write the final video with high quality settings
            final_video.write_videofile(
                output_path,
                fps=fps,
                codec='libx264',
                bitrate=bitrate,
                preset=preset,
                audio_codec='aac',
                audio_bitrate='192k',
                threads=8
            )
            
            # Cleanup
            final_video.close()
            
            logger.info(f"Final video rendered: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Final rendering failed: {e}")
            raise
    
    def render_from_edl(self, edl_path: str, output_path: str, preview: bool = False) -> str:
        """Render video from an EDL JSON file."""
        # Load the storyboard
        with open(edl_path, 'r') as f:
            storyboard_data = json.load(f)
        
        storyboard = Storyboard(**storyboard_data)
        
        # Render based on mode
        if preview:
            return self.render_preview(storyboard, output_path)
        else:
            return self.render_final(storyboard, output_path)
    
    def generate_thumbnails(self, storyboard: Storyboard, output_dir: str) -> List[str]:
        """Generate thumbnail images for each clip in the storyboard."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        thumbnails = []
        
        for i, segment in enumerate(storyboard.timeline):
            try:
                # Get clip info
                clip_info = self._get_clip_info(segment.clip_id)
                
                # Extract frame at the midpoint of the segment
                midpoint = clip_info["timestamp"] + (segment.in_point + segment.out_point) / 2
                
                # Use ffmpeg to extract frame
                output_path = output_dir / f"thumb_{i:03d}.jpg"
                
                stream = ffmpeg.input(clip_info["video_path"], ss=midpoint)
                stream = ffmpeg.output(stream, str(output_path), vframes=1, **{'qscale:v': 2})
                ffmpeg.run(stream, quiet=True, overwrite_output=True)
                
                thumbnails.append(str(output_path))
                
            except Exception as e:
                logger.error(f"Failed to generate thumbnail for segment {i}: {e}")
        
        return thumbnails