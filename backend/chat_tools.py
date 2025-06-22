"""
Chat tools and storyboard generation using LLM with function calling.
"""
import json
import httpx
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from .vector import VectorStore
from .pipeline import OllamaClient
from .config import settings, get_model_with_fallback

logger = logging.getLogger(__name__)


class ClipSegment(BaseModel):
    """A single clip segment in the timeline."""
    clip_id: str = Field(description="ID of the frame/clip from vector store")
    in_point: float = Field(description="Start time in seconds")
    out_point: float = Field(description="End time in seconds")
    transition: Optional[str] = Field(default="cut", description="Transition type: cut, fade, dissolve")


class Storyboard(BaseModel):
    """Complete storyboard/EDL for video generation."""
    project_name: str = Field(description="Name of the project")
    duration: float = Field(description="Total duration in seconds")
    timeline: List[ClipSegment] = Field(description="Ordered list of clips")
    voiceover: Optional[str] = Field(default=None, description="Voiceover script")
    music_cue: Optional[str] = Field(default=None, description="Background music file")
    notes: str = Field(description="Director's notes and creative direction")
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class StoryboardGenerator:
    """Generate storyboards using LLM with vector search."""
    
    def __init__(self, vector_store: VectorStore, ollama_client: OllamaClient):
        self.vector_store = vector_store
        self.ollama = ollama_client
        
        # Function definitions for LLM
        self.functions = [
            {
                "name": "search_clips",
                "description": "Search for video clips based on visual or text description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query describing desired clips"
                        },
                        "n_results": {
                            "type": "integer",
                            "description": "Number of results to return",
                            "default": 10
                        },
                        "quality_threshold": {
                            "type": "number",
                            "description": "Minimum quality score (0-10)",
                            "default": 5.0
                        }
                    },
                    "required": ["query"]
                }
            },
            {
                "name": "create_storyboard",
                "description": "Create a storyboard with selected clips",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "project_name": {
                            "type": "string",
                            "description": "Name for this storyboard project"
                        },
                        "clips": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "clip_id": {"type": "string"},
                                    "in_point": {"type": "number"},
                                    "out_point": {"type": "number"},
                                    "transition": {"type": "string", "enum": ["cut", "fade", "dissolve"]}
                                },
                                "required": ["clip_id", "in_point", "out_point"]
                            }
                        },
                        "voiceover": {
                            "type": "string",
                            "description": "Optional voiceover script"
                        },
                        "music_cue": {
                            "type": "string",
                            "description": "Optional background music"
                        },
                        "notes": {
                            "type": "string",
                            "description": "Creative notes"
                        }
                    },
                    "required": ["project_name", "clips", "notes"]
                }
            }
        ]
    
    def search_clips(self, query: str, n_results: int = 10, quality_threshold: float = 5.0) -> List[Dict[str, Any]]:
        """Search for clips using the vector store."""
        # Generate embedding for the query
        embedding = self.ollama.generate_embedding(query)
        
        # Search in vector store
        results = self.vector_store.search_frames(
            query_embedding=embedding,
            n_results=n_results,
            quality_threshold=quality_threshold
        )
        
        # Format results for LLM
        formatted_results = []
        for result in results:
            formatted_results.append({
                "clip_id": result["id"],
                "caption": result["caption"],
                "timestamp": result["metadata"]["timestamp"],
                "quality": result["metadata"]["quality"],
                "mood": result["metadata"]["mood"],
                "tags": result["metadata"]["tags"]
            })
        
        return formatted_results
    
    def create_storyboard_from_clips(
        self,
        project_name: str,
        clips: List[Dict[str, Any]],
        voiceover: Optional[str] = None,
        music_cue: Optional[str] = None,
        notes: str = ""
    ) -> Storyboard:
        """Create a storyboard from selected clips."""
        # Convert clips to ClipSegment objects
        timeline = []
        for clip in clips:
            segment = ClipSegment(
                clip_id=clip["clip_id"],
                in_point=clip["in_point"],
                out_point=clip["out_point"],
                transition=clip.get("transition", "cut")
            )
            timeline.append(segment)
        
        # Calculate total duration
        duration = sum(seg.out_point - seg.in_point for seg in timeline)
        
        # Create storyboard
        storyboard = Storyboard(
            project_name=project_name,
            duration=duration,
            timeline=timeline,
            voiceover=voiceover,
            music_cue=music_cue,
            notes=notes
        )
        
        return storyboard
    
    def generate_storyboard(
        self,
        prompt: str,
        project_name: str,
        duration_target: Optional[float] = None,
        model: str = None
    ) -> Storyboard:
        """Generate a complete storyboard from a text prompt."""
        
        # Get appropriate model
        if model is None:
            available = self.ollama.get_available_models()
            model = get_model_with_fallback(
                settings.models.chat_model,
                settings.models.chat_model_fallback,
                available
            )
        
        # System prompt for the LLM
        system_prompt = f"""You are a professional video editor creating storyboards.
You have access to a library of video clips that you can search and arrange.

Guidelines:
1. Search for relevant clips using descriptive queries
2. Select high-quality clips that match the creative vision
3. Arrange clips in a compelling narrative sequence
4. Consider pacing, rhythm, and emotional flow
5. Add transitions where appropriate
6. Include creative notes explaining your choices

Target duration: {duration_target if duration_target else 'flexible, but keep it concise'}
"""
        
        # Prepare the conversation
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Create a storyboard for: {prompt}\nProject name: {project_name}"
            }
        ]
        
        # Call LLM with function calling
        response = self.ollama.client.post(
            f"{self.ollama.base_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "format": "json"
            }
        )
        
        if response.status_code != 200:
            logger.error(f"LLM call failed: {response.text}")
            # Fallback: create a simple storyboard
            return self._create_fallback_storyboard(prompt, project_name)
        
        # Parse response and execute function calls
        # Note: This is simplified - real implementation would handle multiple rounds
        llm_response = response.json()
        
        # For now, create a simple storyboard based on search
        search_results = self.search_clips(prompt, n_results=20)
        
        # Select best clips based on quality and relevance
        selected_clips = []
        current_duration = 0.0
        target = duration_target or 60.0  # Default 60 seconds
        
        for i, clip in enumerate(search_results[:10]):  # Use top 10 results
            clip_duration = 3.0 + (i % 3)  # Vary clip lengths 3-5 seconds
            
            if current_duration + clip_duration > target * 1.2:  # Allow 20% overage
                break
            
            selected_clips.append({
                "clip_id": clip["clip_id"],
                "in_point": 0.0,
                "out_point": clip_duration,
                "transition": "cut" if i % 3 != 0 else "fade"
            })
            
            current_duration += clip_duration
        
        # Create storyboard
        notes = f"Automated storyboard for '{prompt}'. Selected {len(selected_clips)} clips with varied pacing and transitions."
        
        return self.create_storyboard_from_clips(
            project_name=project_name,
            clips=selected_clips,
            notes=notes
        )
    
    def _create_fallback_storyboard(self, prompt: str, project_name: str) -> Storyboard:
        """Create a simple fallback storyboard."""
        # Search for any available clips
        results = self.vector_store.hybrid_search(prompt, n_results=5)
        
        clips = []
        for frame in results.get("frames", [])[:3]:
            clips.append({
                "clip_id": frame["id"],
                "in_point": 0.0,
                "out_point": 3.0,
                "transition": "cut"
            })
        
        if not clips:
            # No clips found, create empty storyboard
            return Storyboard(
                project_name=project_name,
                duration=0.0,
                timeline=[],
                notes="No matching clips found for the prompt"
            )
        
        return self.create_storyboard_from_clips(
            project_name=project_name,
            clips=clips,
            notes=f"Fallback storyboard for '{prompt}'"
        )
    
    def refine_storyboard(
        self,
        storyboard: Storyboard,
        feedback: str,
        model: str = "mistral:latest"
    ) -> Storyboard:
        """Refine an existing storyboard based on feedback."""
        # Convert storyboard to dict for LLM
        storyboard_dict = storyboard.model_dump()
        
        messages = [
            {
                "role": "system",
                "content": "You are refining a video storyboard based on feedback. Adjust the timeline, add/remove clips, or change transitions as needed."
            },
            {
                "role": "user",
                "content": f"Current storyboard: {json.dumps(storyboard_dict, indent=2)}\n\nFeedback: {feedback}\n\nPlease provide an updated storyboard."
            }
        ]
        
        # For now, just return the original with updated notes
        storyboard.notes += f"\n\nRefinement requested: {feedback}"
        return storyboard
    
    def export_edl(self, storyboard: Storyboard, output_path: str) -> None:
        """Export storyboard as EDL JSON file."""
        with open(output_path, 'w') as f:
            json.dump(storyboard.model_dump(), f, indent=2)
    
    def import_edl(self, input_path: str) -> Storyboard:
        """Import storyboard from EDL JSON file."""
        with open(input_path, 'r') as f:
            data = json.load(f)
        return Storyboard(**data)