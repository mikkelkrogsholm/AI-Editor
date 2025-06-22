"""
Project management module for organizing media files and metadata.
"""
import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import logging

from .config import settings

logger = logging.getLogger(__name__)


class ProjectManager:
    """Manages project folders and assets."""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(settings.storage.projects_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def create_project(self, name: str) -> Dict[str, Any]:
        """Create a new project with folder structure."""
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
            "renders"
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
                }
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
            "renders": "renders"
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