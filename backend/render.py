"""
Video rendering module using MoviePy and FFmpeg.
"""
import ffmpeg
from pathlib import Path

# Configure ImageMagick path for MoviePy BEFORE importing moviepy
import os
os.environ["IMAGEMAGICK_BINARY"] = "/opt/homebrew/bin/magick"

# Now import moviepy
from moviepy.editor import *
from moviepy.config import change_settings
change_settings({"IMAGEMAGICK_BINARY": "/opt/homebrew/bin/magick"})
import json
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
import logging
from datetime import timedelta

from .chat_tools import Storyboard, ClipSegment
from .vector import VectorStore
from .config import settings

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
        try:
            # Search for the specific frame
            results = self.vector_store.frame_collection.get(
                ids=[clip_id],
                include=["metadatas", "documents"]
            )
            
            if not results['ids'] or not results['ids'][0]:
                logger.error(f"Clip not found in database: {clip_id}")
                raise ValueError(f"Clip not found: {clip_id}")
            
            metadata = results['metadatas'][0]
            logger.info(f"Found clip metadata: {metadata}")
            
            # Get video path - it's already stored with the full relative path
            video_path = metadata.get("video", "")
            
            return {
                "video_path": video_path,
                "timestamp": self._parse_timestamp(metadata.get("timestamp", "00:00.00")),
                "caption": results['documents'][0] if results['documents'] else "",
                "clip_path": metadata.get("clip_path")  # Get pre-extracted clip path
            }
        except Exception as e:
            logger.error(f"Error getting clip info for {clip_id}: {e}")
            raise
    
    def _create_clip(self, segment: ClipSegment) -> VideoFileClip:
        """Create a MoviePy clip from a segment definition."""
        try:
            # Get clip information
            clip_info = self._get_clip_info(segment.clip_id)
            logger.info(f"Creating clip from: {clip_info}")
            
            # Use pre-extracted clip if available
            if clip_info.get("clip_path") and Path(clip_info["clip_path"]).exists():
                logger.info(f"Using pre-extracted clip: {clip_info['clip_path']}")
                # Load pre-extracted clip
                video = VideoFileClip(str(clip_info["clip_path"]))
                
                if video is None:
                    raise ValueError(f"Failed to load video clip: {clip_info['clip_path']}")
                
                # Apply in/out points relative to the clip
                clip_duration = video.duration if video.duration else 10.0
                end_point = min(segment.out_point, clip_duration)
                
                logger.info(f"Creating subclip: {segment.in_point:.2f}s - {end_point:.2f}s (duration: {clip_duration:.2f}s)")
                
                clip = video.subclip(segment.in_point, end_point)
                
                # Don't close the video here as it would close the clip too
            else:
                # Check if video path exists
                video_path = clip_info["video_path"]
                if not Path(video_path).exists():
                    logger.error(f"Video file not found: {video_path}")
                    # Try with absolute path
                    from pathlib import Path as P
                    abs_path = P.cwd() / video_path
                    if abs_path.exists():
                        video_path = str(abs_path)
                    else:
                        raise FileNotFoundError(f"Video file not found: {video_path}")
                
                logger.info(f"Loading source video: {video_path}")
                # Fallback to extracting from source video
                video = VideoFileClip(video_path)
                
                # Calculate actual timestamps
                source_start = clip_info["timestamp"] + segment.in_point
                source_end = clip_info["timestamp"] + segment.out_point
                
                logger.info(f"Extracting subclip: {source_start:.2f}s - {source_end:.2f}s")
                
                # Extract subclip
                clip = video.subclip(source_start, min(source_end, video.duration))
                
                # Don't close the video here as it would close the clip too
            
            return clip
            
        except Exception as e:
            logger.error(f"Failed to create clip {segment.clip_id}: {e}")
            # Return a blank clip as fallback
            duration = segment.out_point - segment.in_point
            logger.warning(f"Using black placeholder clip for {duration:.2f}s")
            return ColorClip(size=(1920, 1080), color=(0, 0, 0), duration=duration)
    
    def _apply_transition(self, clip1: VideoClip, clip2: VideoClip, transition_type: str, duration: float = 1.0) -> List[VideoClip]:
        """Apply transition between two clips."""
        # Temporarily disable all transitions to debug the issue
        logger.info(f"Applying {transition_type} transition (disabled for debugging)")
        return [clip1, clip2]
        
        # Original code commented out for debugging
        # if transition_type == "cut":
        #     return [clip1, clip2]
        # 
        # elif transition_type == "fade":
        #     # Fade out clip1, fade in clip2
        #     clip1_fade = clip1.fx(vfx.fadeout, duration)
        #     clip2_fade = clip2.fx(vfx.fadein, duration)
        #     
        #     # Overlap the clips
        #     clip2_fade = clip2_fade.set_start(clip1.duration - duration)
        #     
        #     return [clip1_fade, clip2_fade]
        # 
        # elif transition_type == "dissolve":
        #     # Cross-dissolve between clips
        #     clip1_fade = clip1.fx(vfx.fadeout, duration)
        #     clip2_fade = clip2.fx(vfx.fadein, duration)
        #     
        #     # Set timing for overlap
        #     clip2_fade = clip2_fade.set_start(clip1.duration - duration)
        #     
        #     return [clip1_fade, clip2_fade]
        # 
        # else:
        #     # Default to cut
        #     return [clip1, clip2]
    
    def render_preview(
        self,
        storyboard: Storyboard,
        output_path: str,
        resolution: Tuple[int, int] = None,
        fps: int = None,
        watermark: bool = None
    ) -> str:
        """Render a low-resolution preview of the storyboard."""
        # Ensure ImageMagick is configured
        os.environ["IMAGEMAGICK_BINARY"] = "/opt/homebrew/bin/magick"
        
        # Use config defaults if not specified
        if resolution is None:
            resolution = settings.render.preview_resolution
        if fps is None:
            fps = settings.render.preview_fps
        if watermark is None:
            watermark = settings.render.preview_watermark
            
        try:
            logger.info(f"Rendering preview for {storyboard.project_name}")
            
            # Create clips from timeline
            clips = []
            previous_clip = None
            
            for i, segment in enumerate(storyboard.timeline):
                # Create the clip
                clip = self._create_clip(segment)
                
                if clip is None:
                    logger.error(f"Failed to create clip for segment {i}: {segment.clip_id}")
                    continue
                
                # Skip resize for now to avoid ImageMagick issues
                logger.info(f"Skipping resize to avoid ImageMagick issues")
                
                # Simply add clips without transitions for now
                clips.append(clip)
                previous_clip = clip
            
            logger.info(f"Collected {len(clips)} clips for rendering")
            
            # Composite all clips
            if not clips:
                logger.warning("No clips to render")
                # Create a blank video
                final_video = ColorClip(size=resolution, color=(0, 0, 0), duration=1)
            else:
                # Use concatenate_videoclips instead of CompositeVideoClip
                from moviepy.video.compositing.concatenate import concatenate_videoclips
                logger.info(f"Concatenating {len(clips)} clips")
                # Use 'chain' method instead of 'compose' to avoid ImageMagick issues
                final_video = concatenate_videoclips(clips, method="chain")
            
            # Add watermark if requested
            if watermark:
                try:
                    watermark_text = TextClip(
                        "AI-KLIPPEREN PREVIEW",
                        fontsize=24,
                        color='white',
                        stroke_color='black',
                        stroke_width=2,
                        font='Arial',
                        method='caption'  # Use caption method which is more reliable
                    ).set_duration(final_video.duration)
                except Exception as e:
                    logger.warning(f"Failed to create watermark text: {e}")
                    # Create a simple colored rectangle as fallback
                    from moviepy.video.fx.all import resize
                    watermark_text = ColorClip(size=(300, 40), color=(255, 255, 255))
                    watermark_text = watermark_text.set_opacity(0.7).set_duration(final_video.duration)
                
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
            logger.info(f"Writing preview to {output_path} with fps={fps}")
            try:
                final_video.write_videofile(
                    output_path,
                    fps=fps,
                    codec='libx264',
                    preset='ultrafast',
                    threads=4,
                    logger=None  # Disable MoviePy's progress bar
                )
            except Exception as e:
                logger.error(f"Failed to write video file: {e}")
                # Try a simpler approach
                logger.info("Trying alternative write method...")
                final_video.write_videofile(
                    output_path,
                    fps=fps,
                    codec='libx264'
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
        resolution: Tuple[int, int] = None,
        fps: int = None,
        bitrate: str = None,
        preset: str = None
    ) -> str:
        """Render the final high-quality video."""
        # Use config defaults if not specified
        if resolution is None:
            resolution = settings.render.final_resolution
        if fps is None:
            fps = settings.render.final_fps
        if bitrate is None:
            bitrate = settings.render.final_bitrate
        if preset is None:
            preset = settings.render.final_preset
            
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