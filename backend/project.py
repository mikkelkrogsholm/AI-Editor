"""
Project management module for organizing media files and metadata.
"""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging
import re

from .config import settings

logger = logging.getLogger(__name__)


class ProjectManager:
    """Manages project folders and assets."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(settings.storage.projects_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def _is_valid_name(self, name: str) -> bool:
        """Check if project name is valid."""
        return bool(re.match(r'^[a-zA-Z0-9_-]+$', name))
    
    def create_project(self, name: str) -> Dict[str, Any]:
        """Create a new project with folder structure."""
        if not self._is_valid_name(name):
            raise ValueError("Project name can only contain letters, numbers, hyphens, and underscores")
        
        project_path = self.base_dir / name
        
        if project_path.exists():
            raise ValueError(f"Project '{name}' already exists")
        
        # Create folder structure
        folders = [
            "sources",
            "clips",
            "frames",
            "audio/extracted",
            "audio/voiceover",
            "audio/music",
            "stills/logos",
            "stills/graphics",
            "stills/photos",
            "renders",
            "storyboards"
        ]
        
        for folder in folders:
            (project_path / folder).mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata = {
            "name": name,
            "created": datetime.now().isoformat(),
            "updated": datetime.now().isoformat(),
            "assets": {
                "videos": [],
                "audio": {
                    "extracted": [],
                    "voiceover": [],
                    "music": []
                },
                "stills": {
                    "logos": [],
                    "graphics": [],
                    "photos": []
                },
                "storyboards": []
            },
            "stats": {
                "total_videos": 0,
                "total_clips": 0,
                "total_frames": 0,
                "total_duration": 0.0
            }
        }
        
        metadata_path = project_path / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Created project: {name}")
        return metadata
    
    def get_project_path(self, name: str) -> Path:
        """Get path to project directory."""
        project_path = self.base_dir / name
        if not project_path.exists():
            raise ValueError(f"Project '{name}' not found")
        return project_path
    
    def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects with basic info."""
        projects = []
        
        for project_dir in self.base_dir.iterdir():
            if project_dir.is_dir() and (project_dir / "metadata.json").exists():
                with open(project_dir / "metadata.json", 'r') as f:
                    metadata = json.load(f)
                
                projects.append({
                    "name": metadata["name"],
                    "created": metadata["created"],
                    "updated": metadata["updated"],
                    "stats": metadata.get("stats", {})
                })
        
        return sorted(projects, key=lambda x: x["updated"], reverse=True)
    
    def delete_project(self, name: str) -> None:
        """Delete a project and all its files."""
        project_path = self.get_project_path(name)
        
        # Safety confirmation
        metadata_path = project_path / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            logger.warning(f"Deleting project '{name}' with {metadata['stats']['total_videos']} videos")
        
        shutil.rmtree(project_path)
        logger.info(f"Deleted project: {name}")
    
    def import_asset(self, project_name: str, file_path: str, asset_type: str) -> Dict[str, Any]:
        """Import an asset into the project."""
        project_path = self.get_project_path(project_name)
        source_path = Path(file_path)
        
        if not source_path.exists():
            raise ValueError(f"File not found: {file_path}")
        
        # Determine destination based on asset type
        type_map = {
            "video": "sources",
            "audio_music": "audio/music",
            "audio_voiceover": "audio/voiceover",
            "still_logo": "stills/logos",
            "still_graphic": "stills/graphics",
            "still_photo": "stills/photos"
        }
        
        if asset_type not in type_map:
            raise ValueError(f"Invalid asset type: {asset_type}")
        
        dest_dir = project_path / type_map[asset_type]
        dest_path = dest_dir / source_path.name
        
        # Copy file
        shutil.copy2(source_path, dest_path)
        
        # Update metadata
        self._update_project_metadata(project_name, asset_type, source_path.name)
        
        logger.info(f"Imported {asset_type}: {source_path.name} to project {project_name}")
        
        return {
            "project": project_name,
            "asset_type": asset_type,
            "filename": source_path.name,
            "path": str(dest_path)
        }
    
    def get_project_assets(self, project_name: str, asset_type: Optional[str] = None) -> Dict[str, List[str]]:
        """Get list of assets in a project."""
        project_path = self.get_project_path(project_name)
        
        assets = {}
        
        # Define asset locations
        asset_dirs = {
            "videos": "sources",
            "clips": "clips",
            "frames": "frames",
            "audio_extracted": "audio/extracted",
            "audio_music": "audio/music",
            "audio_voiceover": "audio/voiceover",
            "still_logos": "stills/logos",
            "still_graphics": "stills/graphics",
            "still_photos": "stills/photos",
            "renders": "renders",
            "storyboards": "storyboards"
        }
        
        # Filter by type if specified
        if asset_type:
            if asset_type in asset_dirs:
                asset_dirs = {asset_type: asset_dirs[asset_type]}
            else:
                return {}
        
        # Collect assets
        for asset_key, rel_path in asset_dirs.items():
            asset_path = project_path / rel_path
            if asset_path.exists():
                files = [f.name for f in asset_path.iterdir() if f.is_file() and not f.name.startswith('.')]
                assets[asset_key] = sorted(files)
        
        return assets
    
    def get_unprocessed_videos(self, project_name: str) -> List[Path]:
        """Get list of videos that haven't been processed yet."""
        project_path = self.get_project_path(project_name)
        sources_dir = project_path / "sources"
        clips_dir = project_path / "clips"
        
        if not sources_dir.exists():
            return []
        
        unprocessed = []
        video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}
        
        for video_file in sources_dir.iterdir():
            if video_file.suffix.lower() in video_extensions:
                # Check if clips exist for this video
                clip_pattern = f"{video_file.stem}_*.mp4"
                if not list(clips_dir.glob(clip_pattern)):
                    unprocessed.append(video_file)
        
        return unprocessed
    
    def update_project_stats(self, project_name: str, stats: Dict[str, Any]) -> None:
        """Update project statistics."""
        project_path = self.get_project_path(project_name)
        metadata_path = project_path / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        metadata["stats"].update(stats)
        metadata["updated"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _update_project_metadata(self, project_name: str, asset_type: str, filename: str) -> None:
        """Update project metadata after importing asset."""
        project_path = self.get_project_path(project_name)
        metadata_path = project_path / "metadata.json"
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Update asset lists
        if asset_type == "video":
            if filename not in metadata["assets"]["videos"]:
                metadata["assets"]["videos"].append(filename)
                metadata["stats"]["total_videos"] += 1
        elif asset_type.startswith("audio_"):
            subtype = asset_type.split("_")[1]
            if filename not in metadata["assets"]["audio"][subtype]:
                metadata["assets"]["audio"][subtype].append(filename)
        elif asset_type.startswith("still_"):
            subtype = asset_type.split("_")[1] + "s"
            if filename not in metadata["assets"]["stills"][subtype]:
                metadata["assets"]["stills"][subtype].append(filename)
        
        metadata["updated"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def save_storyboard(self, project_name: str, storyboard: Dict[str, Any]) -> str:
        """Save a storyboard to the project."""
        project_path = self.get_project_path(project_name)
        storyboards_dir = project_path / "storyboards"
        storyboards_dir.mkdir(exist_ok=True)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"storyboard_{timestamp}.json"
        filepath = storyboards_dir / filename
        
        # Add metadata
        storyboard_with_meta = {
            **storyboard,
            "created": datetime.now().isoformat(),
            "filename": filename
        }
        
        # Save storyboard
        with open(filepath, 'w') as f:
            json.dump(storyboard_with_meta, f, indent=2)
        
        # Update project metadata
        metadata_path = project_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        if "storyboards" not in metadata["assets"]:
            metadata["assets"]["storyboards"] = []
        
        metadata["assets"]["storyboards"].append({
            "filename": filename,
            "created": storyboard_with_meta["created"],
            "duration": storyboard.get("duration", 0),
            "clips": len(storyboard.get("timeline", [])),
            "notes": storyboard.get("notes", "")
        })
        
        metadata["updated"] = datetime.now().isoformat()
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Saved storyboard: {filename} to project {project_name}")
        return filename
    
    def list_storyboards(self, project_name: str) -> List[Dict[str, Any]]:
        """List all storyboards in a project."""
        project_path = self.get_project_path(project_name)
        metadata_path = project_path / "metadata.json"
        
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            return metadata.get("assets", {}).get("storyboards", [])
        
        return []
    
    def load_storyboard(self, project_name: str, filename: str) -> Dict[str, Any]:
        """Load a specific storyboard."""
        project_path = self.get_project_path(project_name)
        storyboard_path = project_path / "storyboards" / filename
        
        if not storyboard_path.exists():
            raise ValueError(f"Storyboard not found: {filename}")
        
        with open(storyboard_path, 'r') as f:
            return json.load(f)
    
    def get_project_content_context(self, project_name: str, vector_store) -> Dict[str, Any]:
        """Get comprehensive content information for AI context."""
        project_path = self.get_project_path(project_name)
        
        # Get basic project info
        metadata_path = project_path / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get all assets
        assets = self.get_project_assets(project_name)
        
        # Get all frames from vector store
        frames = vector_store.get_frames_by_project(project_name)
        
        # Analyze content by categories
        content_analysis = {
            "total_videos": len(assets.get("videos", [])),
            "total_clips": len(assets.get("clips", [])),
            "total_frames_analyzed": len(frames),
            "moods": {},
            "tags": {},
            "quality_distribution": {"high": 0, "medium": 0, "low": 0},
            "technical_specs": {
                "resolutions": set(),
                "aspect_ratios": set(),
                "fps_values": set()
            },
            "content_by_video": {}
        }
        
        # Analyze each frame
        for frame in frames:
            meta = frame["metadata"]
            
            # Count moods
            mood = meta.get("mood", "unknown")
            content_analysis["moods"][mood] = content_analysis["moods"].get(mood, 0) + 1
            
            # Count tags
            tags = meta.get("tags", [])
            if isinstance(tags, str):
                tags = json.loads(tags)
            for tag in tags:
                content_analysis["tags"][tag] = content_analysis["tags"].get(tag, 0) + 1
            
            # Quality distribution
            quality = meta.get("quality", 5)
            if quality >= 7:
                content_analysis["quality_distribution"]["high"] += 1
            elif quality >= 4:
                content_analysis["quality_distribution"]["medium"] += 1
            else:
                content_analysis["quality_distribution"]["low"] += 1
            
            # Technical specs
            if meta.get("resolution_name"):
                content_analysis["technical_specs"]["resolutions"].add(meta["resolution_name"])
            if meta.get("aspect_ratio"):
                content_analysis["technical_specs"]["aspect_ratios"].add(meta["aspect_ratio"])
            if meta.get("fps"):
                content_analysis["technical_specs"]["fps_values"].add(int(meta["fps"]))
            
            # Group by video
            video_name = Path(meta.get("video", "")).name
            if video_name not in content_analysis["content_by_video"]:
                content_analysis["content_by_video"][video_name] = {
                    "frame_count": 0,
                    "clips": set(),
                    "moods": [],
                    "key_moments": []
                }
            
            video_content = content_analysis["content_by_video"][video_name]
            video_content["frame_count"] += 1
            video_content["moods"].append(mood)
            
            # Add key moments (high quality frames with good captions)
            if quality >= 7 and len(frame["caption"]) > 50:
                video_content["key_moments"].append({
                    "timestamp": meta.get("timestamp", ""),
                    "description": frame["caption"][:100] + "...",
                    "mood": mood,
                    "tags": tags[:3]  # First 3 tags
                })
        
        # Convert sets to lists for JSON serialization
        content_analysis["technical_specs"]["resolutions"] = list(content_analysis["technical_specs"]["resolutions"])
        content_analysis["technical_specs"]["aspect_ratios"] = list(content_analysis["technical_specs"]["aspect_ratios"])
        content_analysis["technical_specs"]["fps_values"] = sorted(list(content_analysis["technical_specs"]["fps_values"]))
        
        # Clean up content_by_video
        for video_name, video_data in content_analysis["content_by_video"].items():
            video_data["clips"] = list(video_data["clips"])
            video_data["dominant_mood"] = max(set(video_data["moods"]), key=video_data["moods"].count) if video_data["moods"] else "unknown"
            del video_data["moods"]  # Remove raw mood list
            video_data["key_moments"] = video_data["key_moments"][:5]  # Keep top 5 moments
        
        return {
            "project_name": project_name,
            "created": metadata["created"],
            "assets": assets,
            "content_analysis": content_analysis,
            "summary": self._generate_content_summary(content_analysis)
        }
    
    def _generate_content_summary(self, analysis: Dict[str, Any]) -> str:
        """Generate a human-readable summary of project content."""
        summary_parts = []
        
        # Basic stats
        summary_parts.append(f"{analysis['total_videos']} videos with {analysis['total_frames_analyzed']} analyzed frames")
        
        # Top moods
        if analysis["moods"]:
            top_moods = sorted(analysis["moods"].items(), key=lambda x: x[1], reverse=True)[:3]
            mood_str = ", ".join([f"{mood} ({count})" for mood, count in top_moods])
            summary_parts.append(f"Main moods: {mood_str}")
        
        # Top tags
        if analysis["tags"]:
            top_tags = sorted(analysis["tags"].items(), key=lambda x: x[1], reverse=True)[:5]
            tag_str = ", ".join([f"{tag} ({count})" for tag, count in top_tags])
            summary_parts.append(f"Common elements: {tag_str}")
        
        # Technical
        if analysis["technical_specs"]["resolutions"]:
            res_str = ", ".join(analysis["technical_specs"]["resolutions"])
            summary_parts.append(f"Resolutions: {res_str}")
        
        return " | ".join(summary_parts)