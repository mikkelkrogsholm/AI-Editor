"""
Enhanced chat tools and storyboard generation with proper function calling.
"""
import json
import httpx
from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field
from datetime import datetime
import logging
import re

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


class EnhancedStoryboardGenerator:
    """Generate storyboards using LLM with proper function calling."""
    
    def __init__(self, vector_store: VectorStore, ollama_client: OllamaClient):
        self.vector_store = vector_store
        self.ollama = ollama_client
        
    def search_clips(self, query: str, project: str = None, n_results: int = 10, 
                     quality_threshold: float = 5.0, mood: str = None) -> List[Dict[str, Any]]:
        """Search for clips using the vector store."""
        # Generate embedding for the query
        embedding = self.ollama.generate_embedding(query)
        
        # Search in vector store
        results = self.vector_store.search_frames(
            query_embedding=embedding,
            n_results=n_results,
            quality_threshold=quality_threshold,
            project=project
        )
        
        # Format results for LLM
        formatted_results = []
        for result in results:
            clip_info = {
                "clip_id": result["id"],
                "caption": result["caption"],
                "timestamp": result["metadata"]["timestamp"],
                "quality": result["metadata"]["quality"],
                "mood": result["metadata"]["mood"],
                "tags": result["metadata"]["tags"],
                "clip_path": result["metadata"].get("clip_path", "")
            }
            
            # Filter by mood if specified
            if mood and result["metadata"]["mood"].lower() != mood.lower():
                continue
                
            formatted_results.append(clip_info)
        
        return formatted_results
    
    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """Analyze the prompt to extract key information."""
        analysis = {
            "mood": "neutral",
            "pace": "medium",
            "style": "standard",
            "keywords": []
        }
        
        # Mood detection
        if any(word in prompt.lower() for word in ["energetic", "exciting", "fast", "action"]):
            analysis["mood"] = "energetic"
            analysis["pace"] = "fast"
        elif any(word in prompt.lower() for word in ["slow", "calm", "peaceful", "serene"]):
            analysis["mood"] = "calm"
            analysis["pace"] = "slow"
        elif any(word in prompt.lower() for word in ["emotional", "sad", "melancholy"]):
            analysis["mood"] = "emotional"
            analysis["pace"] = "slow"
        
        # Extract keywords
        keywords = re.findall(r'\b(?:crowd|people|night|day|outdoor|indoor|stage|concert)\b', 
                            prompt.lower())
        analysis["keywords"] = list(set(keywords))
        
        return analysis
    
    def generate_storyboard_with_tools(
        self,
        prompt: str,
        project_name: str,
        duration_target: Optional[float] = None,
        model: str = None
    ) -> Storyboard:
        """Generate a storyboard using proper function calling with Qwen 2.5."""
        
        # Get appropriate model
        if model is None:
            available = self.ollama.get_available_models()
            model = get_model_with_fallback(
                settings.models.chat_model,
                settings.models.chat_model_fallback,
                available
            )
        
        # Analyze the prompt
        prompt_analysis = self.analyze_prompt(prompt)
        
        # Prepare tools definition for Qwen 2.5 (Hermes-style)
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_clips",
                    "description": "Search for video clips based on visual description or keywords",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query describing desired clips"
                            },
                            "n_results": {
                                "type": "integer",
                                "description": "Number of results to return (default: 10)"
                            },
                            "mood": {
                                "type": "string",
                                "enum": ["energetic", "calm", "emotional", "neutral"],
                                "description": "Filter by mood"
                            }
                        },
                        "required": ["query"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "create_timeline",
                    "description": "Create a timeline with selected clips",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "clips": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "clip_id": {"type": "string"},
                                        "duration": {"type": "number", "description": "Duration in seconds"},
                                        "transition": {
                                            "type": "string",
                                            "enum": ["cut", "fade", "dissolve"],
                                            "description": "Transition type"
                                        }
                                    },
                                    "required": ["clip_id", "duration"]
                                }
                            },
                            "notes": {
                                "type": "string",
                                "description": "Creative notes about the timeline"
                            }
                        },
                        "required": ["clips", "notes"]
                    }
                }
            }
        ]
        
        # System prompt optimized for Qwen 2.5
        system_prompt = f"""You are a professional video editor creating storyboards.
You have access to tools to search for video clips and create timelines.

<tools>
{json.dumps(tools, indent=2)}
</tools>

Task: Create a storyboard for the user's request.
Target duration: {duration_target if duration_target else '30-60 seconds'}
Project mood: {prompt_analysis['mood']}
Pace: {prompt_analysis['pace']}

Process:
1. First, search for relevant clips using descriptive queries like:
   - "crowd people concert music" for energetic scenes
   - "outdoor night event purple lighting" for atmospheric shots
   - "person stage performance" for focal points
   - Or simply "crowd" or "night" if you need general footage
2. If first search returns 0 results, try simpler terms: "crowd", "people", "music", "event"
3. Select the best clips based on quality and relevance from the search results
4. Create a timeline using the EXACT clip_id values from the search results
5. Consider pacing - shorter clips (2-3s) for energetic content, longer (4-6s) for calm
6. Use cuts for fast pacing, fades/dissolves for smooth transitions

CRITICAL: 
- You MUST use the exact clip_id values returned from search_clips (format: "test-project_20250621_004927_f001200")
- NEVER create generic IDs like "clip-1" or "clip-neutral-02"
- If search returns no results, try broader search terms

You may call functions by responding with:
<function_call>
{{"name": "function_name", "arguments": {{"param": "value"}}}}
</function_call>"""
        
        # Initialize conversation
        messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": f"Create a storyboard for: {prompt}"
            }
        ]
        
        # Store function results
        search_results = []
        timeline_created = False
        final_timeline = []
        
        # Allow up to 5 rounds of function calling
        for round in range(5):
            response = self.ollama.client.post(
                f"{self.ollama.base_url}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False
                }
            )
            
            if response.status_code != 200:
                logger.error(f"LLM call failed: {response.text}")
                break
            
            llm_response = response.json()
            assistant_message = llm_response.get("message", {}).get("content", "")
            
            # Add assistant response to conversation
            messages.append({"role": "assistant", "content": assistant_message})
            
            # Parse function calls
            function_calls = re.findall(r'<function_call>(.*?)</function_call>', 
                                      assistant_message, re.DOTALL)
            
            if not function_calls:
                # No more function calls, check if we have a timeline
                if timeline_created:
                    break
                else:
                    # Ask for timeline creation
                    messages.append({
                        "role": "user",
                        "content": "Please create the timeline now with the clips you've found."
                    })
                    continue
            
            # Execute function calls
            for fc_str in function_calls:
                try:
                    fc = json.loads(fc_str)
                    function_name = fc.get("name")
                    arguments = fc.get("arguments", {})
                    
                    if function_name == "search_clips":
                        # Execute search
                        results = self.search_clips(
                            query=arguments.get("query", prompt),
                            project=project_name,
                            n_results=arguments.get("n_results", 10),
                            mood=arguments.get("mood")
                        )
                        search_results.extend(results)
                        
                        logger.info(f"Search returned {len(results)} clips")
                        if results:
                            logger.info(f"First clip ID: {results[0]['clip_id']}")
                        
                        # Add results to conversation
                        # Format results to emphasize clip IDs
                        formatted_results = []
                        for r in results[:5]:
                            formatted_results.append({
                                "clip_id": r["clip_id"],  # MUST use this exact ID
                                "caption": r["caption"][:100] + "...",
                                "quality": r["quality"],
                                "mood": r["mood"]
                            })
                        
                        messages.append({
                            "role": "function",
                            "name": "search_clips",
                            "content": f"Found {len(results)} clips. Here are the top 5:\n" + 
                                     json.dumps(formatted_results, indent=2) +
                                     "\n\nREMEMBER: Use these exact clip_id values in your timeline!"
                        })
                        
                    elif function_name == "create_timeline":
                        # Process timeline creation
                        clips_data = arguments.get("clips", [])
                        notes = arguments.get("notes", "")
                        
                        logger.info(f"Creating timeline with {len(clips_data)} clips")
                        
                        for clip_data in clips_data:
                            # Find the clip in search results
                            clip_id = clip_data.get("clip_id")
                            duration = clip_data.get("duration", 3.0)
                            transition = clip_data.get("transition", "cut")
                            
                            logger.debug(f"Adding clip: {clip_id} ({duration}s)")
                            
                            final_timeline.append({
                                "clip_id": clip_id,
                                "in_point": 0.0,
                                "out_point": duration,
                                "transition": transition
                            })
                        
                        timeline_created = True
                        messages.append({
                            "role": "function",
                            "name": "create_timeline",
                            "content": f"Timeline created with {len(final_timeline)} clips"
                        })
                        
                except json.JSONDecodeError:
                    logger.error(f"Failed to parse function call: {fc_str}")
                    continue
        
        # Create final storyboard
        if final_timeline:
            # Check if LLM used real clip IDs
            uses_real_ids = False
            for clip in final_timeline:
                clip_id = clip.get("clip_id", "")
                # Real IDs have format like "test-project_20250621_004927_f001200"
                if "_" in clip_id and len(clip_id) > 20:
                    uses_real_ids = True
                    break
            
            if not uses_real_ids:
                logger.warning("LLM used generic clip IDs, falling back to intelligent selection")
                # If we have no search results, force a search
                if not search_results:
                    logger.info("No search results available, performing fallback search")
                    for keyword in ["crowd", "people", "event", "music", "outdoor", "night"]:
                        results = self.search_clips(keyword, project=project_name, n_results=20)
                        if results:
                            search_results.extend(results)
                            if len(search_results) >= 10:
                                break
                
                return self._create_intelligent_storyboard(
                    prompt, project_name, duration_target, prompt_analysis, search_results
                )
            
            # Validate all clip IDs exist
            valid_timeline = []
            for clip in final_timeline:
                clip_id = clip.get("clip_id", "")
                # Check if clip exists in database
                try:
                    results = self.vector_store.frame_collection.get(ids=[clip_id])
                    if results['ids'] and results['ids'][0]:
                        valid_timeline.append(clip)
                        logger.info(f"Validated clip: {clip_id}")
                    else:
                        logger.warning(f"Invalid clip ID: {clip_id}, will be replaced")
                except Exception as e:
                    logger.warning(f"Could not validate clip {clip_id}: {e}")
            
            # If we have invalid clips, fallback to intelligent selection
            if len(valid_timeline) < len(final_timeline):
                logger.warning(f"Found {len(final_timeline) - len(valid_timeline)} invalid clips, using intelligent fallback")
                return self._create_intelligent_storyboard(
                    prompt, project_name, duration_target, prompt_analysis, search_results
                )
            
            # Calculate total duration
            total_duration = sum(clip["out_point"] - clip["in_point"] for clip in valid_timeline)
            
            # Create storyboard
            storyboard = Storyboard(
                project_name=project_name,
                duration=total_duration,
                timeline=[ClipSegment(**clip) for clip in valid_timeline],
                notes=f"AI-generated storyboard for '{prompt}'. "
                      f"Mood: {prompt_analysis['mood']}, Pace: {prompt_analysis['pace']}"
            )
            
            return storyboard
        else:
            # Fallback to intelligent selection if LLM didn't create timeline
            return self._create_intelligent_storyboard(
                prompt, project_name, duration_target, prompt_analysis, search_results
            )
    
    def _create_intelligent_storyboard(
        self, 
        prompt: str, 
        project_name: str,
        duration_target: Optional[float],
        analysis: Dict[str, Any],
        search_results: List[Dict[str, Any]]
    ) -> Storyboard:
        """Create an intelligent storyboard based on analysis."""
        
        # If no search results, do a basic search
        if not search_results:
            # Try common terms that likely exist in concert/event footage
            common_searches = ["crowd", "people", "music", "concert", "event", "night", "outdoor"]
            for keyword in common_searches[:3]:  # Try first 3
                results = self.search_clips(keyword, project=project_name, n_results=10)
                if results:
                    logger.info(f"Found {len(results)} clips with keyword '{keyword}'")
                    search_results.extend(results)
                    if len(search_results) >= 5:  # Enough clips
                        break
        
        # If still no results, get ANY clips from the project
        if not search_results:
            logger.warning("No search results found, getting random clips from project")
            try:
                # Get all frames from the project
                all_frames = self.vector_store.get_frames_by_project(project_name, limit=50)
                if all_frames:
                    # Convert to search result format
                    for frame in all_frames[:20]:
                        search_results.append({
                            "clip_id": frame["id"],
                            "caption": frame.get("caption", ""),
                            "timestamp": frame["metadata"].get("timestamp", "00:00.00"),
                            "quality": frame["metadata"].get("quality", 5.0),
                            "mood": frame["metadata"].get("mood", "neutral"),
                            "tags": frame["metadata"].get("tags", []),
                            "clip_path": frame["metadata"].get("clip_path", "")
                        })
                    logger.info(f"Added {len(search_results)} random clips from project")
            except Exception as e:
                logger.error(f"Failed to get random clips: {e}")
        
        # Remove duplicates
        seen_ids = set()
        unique_results = []
        for result in search_results:
            if result["clip_id"] not in seen_ids:
                seen_ids.add(result["clip_id"])
                unique_results.append(result)
        
        # Sort by quality and relevance
        unique_results.sort(key=lambda x: x["quality"], reverse=True)
        
        # Determine clip lengths based on pace
        if analysis["pace"] == "fast":
            min_duration, max_duration = 1.5, 3.0
        elif analysis["pace"] == "slow":
            min_duration, max_duration = 4.0, 6.0
        else:
            min_duration, max_duration = 2.5, 4.5
        
        # Build timeline
        timeline = []
        current_duration = 0.0
        target = duration_target or 30.0
        
        for i, clip in enumerate(unique_results[:20]):  # Consider more clips
            # Vary clip duration
            if analysis["pace"] == "fast":
                clip_duration = min_duration + (i % 2) * 0.5
            else:
                clip_duration = min_duration + (i % 3) * ((max_duration - min_duration) / 2)
            
            # Check if we've reached target - be more generous
            if current_duration >= target and i >= 3:  # At least 3 clips
                break
            
            # Determine transition based on pace and position
            if analysis["pace"] == "fast":
                transition = "cut"
            elif i == 0:
                transition = "fade"  # Fade in
            elif current_duration + clip_duration >= target * 0.9:
                transition = "fade"  # Fade out
            else:
                transition = "cut" if i % 3 != 0 else "dissolve"
            
            timeline.append(ClipSegment(
                clip_id=clip["clip_id"],
                in_point=0.0,
                out_point=clip_duration,
                transition=transition
            ))
            
            current_duration += clip_duration
        
        # Create storyboard
        return Storyboard(
            project_name=project_name,
            duration=current_duration,
            timeline=timeline,
            notes=f"Intelligent storyboard for '{prompt}'. "
                  f"Selected {len(timeline)} clips with {analysis['mood']} mood and {analysis['pace']} pacing."
        )
    
    def generate_storyboard(
        self,
        prompt: str,
        project_name: str,
        duration_target: Optional[float] = None,
        model: str = None
    ) -> Storyboard:
        """Main entry point - tries tool calling first, falls back to intelligent generation."""
        logger.info(f"EnhancedStoryboardGenerator called with prompt: {prompt[:50]}...")
        try:
            # Try with function calling first
            result = self.generate_storyboard_with_tools(
                prompt, project_name, duration_target, model
            )
            logger.info(f"Tool calling succeeded, generated {len(result.timeline)} clips")
            return result
        except Exception as e:
            logger.warning(f"Tool calling failed: {e}, using intelligent fallback")
            # Fallback to intelligent generation
            analysis = self.analyze_prompt(prompt)
            result = self._create_intelligent_storyboard(
                prompt, project_name, duration_target, analysis, []
            )
            logger.info(f"Intelligent fallback generated {len(result.timeline)} clips")
            return result