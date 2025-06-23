"""
FastAPI backend for AI-Klipperen.
"""
from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import asyncio
from pathlib import Path
import tempfile
import shutil
import uuid
import json
import logging
import os

from .vector import VectorStore
from .pipeline import VideoPipeline, OllamaClient
from .chat_tools import StoryboardGenerator, Storyboard
from .chat_tools_v2 import EnhancedStoryboardGenerator
from .render import VideoRenderer
from .project import ProjectManager
from .models import ProjectInfo, ImportAssetRequest, ProjectProcessRequest
from .config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="AI-Klipperen API",
    description="Video processing and editing API powered by AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
vector_store = VectorStore(persist_directory="./chroma_db")
ollama_client = OllamaClient()
project_manager = ProjectManager()
video_pipeline = VideoPipeline(vector_store, ollama_client, project_manager)
storyboard_generator = StoryboardGenerator(vector_store, ollama_client)
# Use enhanced generator if qwen2.5 is available
try:
    available_models = ollama_client.get_available_models()
    logger.info(f"Available models: {available_models}")
    
    if "qwen2.5:14b" in available_models:
        storyboard_generator = EnhancedStoryboardGenerator(vector_store, ollama_client)
        logger.info("Using EnhancedStoryboardGenerator with qwen2.5:14b")
    else:
        logger.info("Using standard StoryboardGenerator - qwen2.5:14b not found")
except Exception as e:
    logger.warning(f"Error checking for qwen2.5:14b: {e}")
    logger.info("Using standard StoryboardGenerator")
video_renderer = VideoRenderer(vector_store)

# Store for background job status
job_status = {}


class IngestRequest(BaseModel):
    """Request model for video ingestion."""
    project_name: str = Field(description="Name of the project")
    video_paths: List[str] = Field(description="List of video file paths to process")


class IngestResponse(BaseModel):
    """Response model for video ingestion."""
    job_id: str = Field(description="Job ID for tracking progress")
    message: str = Field(description="Status message")


class SearchRequest(BaseModel):
    """Request model for searching clips."""
    query: str = Field(description="Search query")
    n_results: int = Field(default=20, description="Number of results to return")
    quality_threshold: Optional[float] = Field(default=5.0, description="Minimum quality score")
    search_type: str = Field(default="hybrid", description="Search type: frames, asr, or hybrid")
    project: Optional[str] = Field(default=None, description="Filter by project")
    aspect_ratio: Optional[str] = Field(default=None, description="Filter by aspect ratio")
    orientation: Optional[str] = Field(default=None, description="Filter by orientation")
    min_resolution: Optional[str] = Field(default=None, description="Minimum resolution")


class ChatRequest(BaseModel):
    """Request model for chat interaction."""
    message: str = Field(description="User message")
    project_name: str = Field(description="Project context")
    conversation_id: Optional[str] = Field(default=None, description="Conversation ID for context")


class ChatResponse(BaseModel):
    """Response model for chat interaction."""
    response: str = Field(description="AI response")
    storyboard: Optional[Dict[str, Any]] = Field(default=None, description="Generated storyboard if applicable")
    conversation_id: str = Field(description="Conversation ID")


class RenderRequest(BaseModel):
    """Request model for video rendering."""
    storyboard: Dict[str, Any] = Field(description="Storyboard data")
    output_filename: str = Field(description="Output filename")
    preview: bool = Field(default=False, description="Render as preview (low-res)")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "AI-Klipperen API",
        "version": "1.0.0",
        "status": "running"
    }


@app.post("/ingest", response_model=IngestResponse)
async def ingest_videos(request: IngestRequest, background_tasks: BackgroundTasks):
    """Ingest videos for processing."""
    job_id = str(uuid.uuid4())
    
    # Initialize job status
    job_status[job_id] = {
        "status": "queued",
        "progress": 0,
        "total": len(request.video_paths),
        "completed": 0,
        "errors": []
    }
    
    # Add background task
    background_tasks.add_task(
        process_videos_background,
        job_id,
        request.project_name,
        request.video_paths
    )
    
    return IngestResponse(
        job_id=job_id,
        message=f"Processing {len(request.video_paths)} videos in background"
    )


async def process_videos_background(job_id: str, project_name: str, video_paths: List[str]):
    """Background task for processing videos."""
    try:
        job_status[job_id]["status"] = "processing"
        
        for i, video_path in enumerate(video_paths):
            # Update progress
            job_status[job_id]["progress"] = (i / len(video_paths)) * 100
            
            # Process video
            result = await video_pipeline.process_video(video_path, project_name)
            
            if result["errors"]:
                job_status[job_id]["errors"].extend(result["errors"])
            
            job_status[job_id]["completed"] += 1
        
        job_status[job_id]["status"] = "completed"
        job_status[job_id]["progress"] = 100
        
    except Exception as e:
        logger.error(f"Background processing failed: {e}")
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["errors"].append(str(e))


@app.get("/ingest/status/{job_id}")
async def get_ingest_status(job_id: str):
    """Get status of an ingest job."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]


@app.post("/search")
async def search_clips(request: SearchRequest):
    """Search for clips based on query."""
    try:
        if request.search_type == "frames":
            # Generate embedding and search frames
            embedding = ollama_client.generate_embedding(request.query)
            results = vector_store.search_frames(
                query_embedding=embedding,
                n_results=request.n_results,
                quality_threshold=request.quality_threshold,
                project=request.project,
                aspect_ratio=request.aspect_ratio,
                orientation=request.orientation,
                min_resolution=request.min_resolution
            )
            return {"frames": results, "asr_segments": []}
        
        elif request.search_type == "asr":
            # Generate embedding and search ASR
            embedding = ollama_client.generate_embedding(request.query)
            results = vector_store.search_asr(
                query_embedding=embedding,
                n_results=request.n_results
            )
            return {"frames": [], "asr_segments": results}
        
        else:  # hybrid
            # Generate embedding for both
            text_embedding = ollama_client.generate_embedding(request.query)
            
            results = vector_store.hybrid_search(
                text_query=request.query,
                text_embedding=text_embedding,
                image_embedding=text_embedding,  # Use same embedding for simplicity
                n_results=request.n_results
            )
            return results
    
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def format_project_content_for_ai(context: Dict[str, Any]) -> str:
    """Format project content information for AI consumption."""
    parts = []
    
    # Basic stats
    analysis = context.get("content_analysis", {})
    parts.append(f"CONTENT OVERVIEW:")
    parts.append(f"- {analysis.get('total_videos', 0)} videos")
    parts.append(f"- {analysis.get('total_clips', 0)} extracted clips")
    parts.append(f"- {analysis.get('total_frames_analyzed', 0)} analyzed frames")
    
    # Mood distribution
    if analysis.get("moods"):
        parts.append(f"\nMOODS CAPTURED:")
        for mood, count in sorted(analysis["moods"].items(), key=lambda x: x[1], reverse=True)[:5]:
            parts.append(f"- {mood}: {count} frames")
    
    # Common elements
    if analysis.get("tags"):
        parts.append(f"\nCOMMON ELEMENTS:")
        for tag, count in sorted(analysis["tags"].items(), key=lambda x: x[1], reverse=True)[:10]:
            parts.append(f"- {tag}: {count} occurrences")
    
    # Technical info
    tech = analysis.get("technical_specs", {})
    if tech.get("resolutions"):
        parts.append(f"\nTECHNICAL SPECS:")
        parts.append(f"- Resolutions: {', '.join(tech['resolutions'])}")
        parts.append(f"- Aspect ratios: {', '.join(tech['aspect_ratios'])}")
        parts.append(f"- Frame rates: {', '.join(map(str, tech['fps_values']))} fps")
    
    # Video-specific highlights
    if analysis.get("content_by_video"):
        parts.append(f"\nVIDEO CONTENT:")
        for video_name, video_data in list(analysis["content_by_video"].items())[:5]:
            parts.append(f"\n{video_name}:")
            parts.append(f"- {video_data['frame_count']} frames analyzed")
            parts.append(f"- Dominant mood: {video_data.get('dominant_mood', 'unknown')}")
            if video_data.get("key_moments"):
                parts.append(f"- Key moments:")
                for moment in video_data["key_moments"][:3]:
                    parts.append(f"  • {moment['timestamp']}: {moment['description']}")
    
    return "\n".join(parts)


@app.post("/chat", response_model=ChatResponse)
async def chat_interaction(request: ChatRequest):
    """Handle chat interaction for storyboard generation."""
    try:
        # Generate conversation ID if not provided
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Check if the message is requesting storyboard generation
        keywords = ["generate", "create", "make", "storyboard", "edit", "compile", "build", "produce", "render"]
        is_generation_request = any(keyword in request.message.lower() for keyword in keywords)
        
        if is_generation_request:
            # Generate storyboard
            storyboard = storyboard_generator.generate_storyboard(
                prompt=request.message,
                project_name=request.project_name,
                duration_target=60.0  # Default 60 seconds
            )
            
            return ChatResponse(
                response=f"I've created a storyboard for your project '{request.project_name}'. It includes {len(storyboard.timeline)} clips with a total duration of {storyboard.duration:.1f} seconds.",
                storyboard=storyboard.model_dump(),
                conversation_id=conversation_id
            )
        
        else:
            # Regular chat response - use AI to answer
            try:
                # Get project context
                project_context = project_manager.get_project_content_context(
                    request.project_name, 
                    vector_store
                )
                
                # Format content information for the AI
                content_info = format_project_content_for_ai(project_context)
                
                # Use the chat model to generate a response
                messages = [
                    {
                        "role": "system",
                        "content": f"""You are AI-Klipperen, an AI video editing assistant with deep knowledge of the user's video content.

PROJECT: {request.project_name}

{content_info}

Your capabilities:
- Answer questions about the available video content
- Suggest creative ideas based on the actual footage
- Help users understand what types of shots they have
- Recommend specific clips for different moods or themes
- Guide users to create storyboards using words like 'create', 'generate', 'make'

Be helpful, creative, and specific. Reference actual content when making suggestions.
Keep responses concise but informative."""
                    },
                    {
                        "role": "user",
                        "content": request.message
                    }
                ]
                
                # Get AI response
                response = ollama_client.client.post(
                    f"{ollama_client.base_url}/api/chat",
                    json={
                        "model": settings.models.chat_model,
                        "messages": messages,
                        "stream": False
                    }
                )
                
                if response.status_code == 200:
                    ai_response = response.json()
                    response_text = ai_response.get("message", {}).get("content", "I can help you create storyboards for your videos.")
                else:
                    logger.warning(f"Ollama chat API returned status {response.status_code}: {response.text}")
                    response_text = "I can help you create storyboards for your videos. Try asking me to 'generate a 30-second energetic montage' or 'create a storyboard focusing on crowd shots'."
                    
            except Exception as e:
                logger.warning(f"Failed to get AI response: {e}", exc_info=True)
                response_text = "I can help you create storyboards for your videos. Try asking me to 'generate a 30-second energetic montage' or 'create a storyboard focusing on crowd shots'."
            
            return ChatResponse(
                response=response_text,
                storyboard=None,
                conversation_id=conversation_id
            )
    
    except Exception as e:
        logger.error(f"Chat interaction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/validate_storyboard")
async def validate_storyboard(request: Dict[str, Any] = Body(...)):
    """Validate a storyboard before rendering."""
    try:
        # Create storyboard object
        storyboard = Storyboard(**request)
        
        # Validate using renderer
        validation_result = video_renderer.validate_storyboard(storyboard)
        
        return validation_result
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/preview")
async def create_preview(request: RenderRequest, background_tasks: BackgroundTasks):
    """Create a preview render of the storyboard."""
    try:
        # Create storyboard object
        storyboard = Storyboard(**request.storyboard)
        
        # Generate output path
        output_dir = Path("./outputs/previews")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / request.output_filename
        
        # Create job ID
        job_id = str(uuid.uuid4())
        job_status[job_id] = {
            "status": "processing",
            "output_path": str(output_path)
        }
        
        # Render in background
        background_tasks.add_task(
            render_preview_background,
            job_id,
            storyboard,
            str(output_path)
        )
        
        return {
            "job_id": job_id,
            "message": "Preview rendering started"
        }
    
    except Exception as e:
        logger.error(f"Preview creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def render_preview_background(job_id: str, storyboard: Storyboard, output_path: str):
    """Background task for rendering preview."""
    try:
        # Debug environment variables
        logger.info(f"=== Background Task Environment Debug ===")
        logger.info(f"PATH: {os.environ.get('PATH', 'NOT SET')}")
        logger.info(f"IMAGEMAGICK_BINARY: {os.environ.get('IMAGEMAGICK_BINARY', 'NOT SET')}")
        logger.info(f"FFMPEG_BINARY: {os.environ.get('FFMPEG_BINARY', 'NOT SET')}")
        
        # Set environment variables explicitly
        os.environ["IMAGEMAGICK_BINARY"] = "/opt/homebrew/bin/magick"
        os.environ["FFMPEG_BINARY"] = "/opt/miniconda3/bin/ffmpeg"
        logger.info(f"Set IMAGEMAGICK_BINARY to: {os.environ['IMAGEMAGICK_BINARY']}")
        
        video_renderer.render_preview(storyboard, output_path)
        job_status[job_id]["status"] = "completed"
    except Exception as e:
        logger.error(f"Preview rendering failed: {e}", exc_info=True)
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)


@app.post("/render")
async def render_final(request: RenderRequest, background_tasks: BackgroundTasks):
    """Render final high-quality video."""
    try:
        # Create storyboard object
        storyboard = Storyboard(**request.storyboard)
        
        # Generate output path
        output_dir = Path("./outputs/final")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / request.output_filename
        
        # Create job ID
        job_id = str(uuid.uuid4())
        job_status[job_id] = {
            "status": "processing",
            "output_path": str(output_path)
        }
        
        # Render in background
        background_tasks.add_task(
            render_final_background,
            job_id,
            storyboard,
            str(output_path)
        )
        
        return {
            "job_id": job_id,
            "message": "Final rendering started"
        }
    
    except Exception as e:
        logger.error(f"Final render failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def render_final_background(job_id: str, storyboard: Storyboard, output_path: str):
    """Background task for rendering final video."""
    try:
        # Set environment variables explicitly (same as preview)
        os.environ["IMAGEMAGICK_BINARY"] = "/opt/homebrew/bin/magick"
        os.environ["FFMPEG_BINARY"] = "/opt/miniconda3/bin/ffmpeg"
        
        video_renderer.render_final(storyboard, output_path)
        job_status[job_id]["status"] = "completed"
    except Exception as e:
        logger.error(f"Final rendering failed: {e}", exc_info=True)
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["error"] = str(e)


@app.post("/render_sync")
def render_sync(request: RenderRequest):
    """Test synchronous rendering without background tasks."""
    try:
        # Set environment variables
        os.environ["IMAGEMAGICK_BINARY"] = "/opt/homebrew/bin/magick"
        os.environ["FFMPEG_BINARY"] = "/opt/miniconda3/bin/ffmpeg"
        
        storyboard = Storyboard(**request.storyboard)
        output_filename = request.output_filename or f"sync_test_{uuid.uuid4()}.mp4"
        output_path = Path("./outputs/sync_test") / output_filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting synchronous render to {output_path}")
        video_renderer.render_preview(storyboard, str(output_path))
        
        return {
            "status": "success",
            "path": str(output_path),
            "message": "Synchronous render completed"
        }
    except Exception as e:
        logger.error(f"Sync render failed: {e}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e),
            "message": "Synchronous render failed"
        }


@app.get("/render/status/{job_id}")
async def get_render_status(job_id: str):
    """Get status of a render job."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status[job_id]


@app.get("/download/{job_id}")
async def download_render(job_id: str):
    """Download rendered video."""
    if job_id not in job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_status[job_id]["status"] != "completed":
        raise HTTPException(status_code=400, detail="Render not completed")
    
    output_path = job_status[job_id].get("output_path")
    if not output_path or not Path(output_path).exists():
        raise HTTPException(status_code=404, detail="Output file not found")
    
    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        filename=Path(output_path).name
    )


@app.post("/upload")
async def upload_video(file: UploadFile = File(...), project_name: str = Body(...)):
    """Upload a video file for processing."""
    try:
        # Save uploaded file
        temp_dir = Path("./uploads")
        temp_dir.mkdir(exist_ok=True)
        
        file_path = temp_dir / f"{uuid.uuid4()}_{file.filename}"
        
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Process the video
        result = await video_pipeline.process_video(str(file_path), project_name)
        
        return {
            "message": "Video uploaded and processed",
            "file_path": str(file_path),
            "result": result
        }
    
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects")
async def list_projects():
    """List all projects."""
    try:
        projects = project_manager.list_projects()
        return projects
    except Exception as e:
        logger.error(f"Failed to list projects: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/create")
async def create_project(name: str = Body(..., embed=True)):
    """Create a new project."""
    try:
        project_info = project_manager.create_project(name)
        return {
            "message": f"Project '{name}' created successfully",
            "path": str(project_manager.get_project_path(name)),
            **project_info
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to create project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/projects/{project_name}")
async def delete_project(project_name: str):
    """Delete a project and all its files."""
    try:
        project_manager.delete_project(project_name)
        return {"message": f"Project '{project_name}' deleted successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to delete project: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{project_name}/import")
async def import_asset(project_name: str, request: ImportAssetRequest):
    """Import an asset into a project."""
    try:
        result = project_manager.import_asset(
            project_name,
            request.file_path,
            request.asset_type
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to import asset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_name}/assets")
async def get_project_assets(project_name: str, asset_type: Optional[str] = None):
    """Get list of assets in a project."""
    try:
        assets = project_manager.get_project_assets(project_name, asset_type)
        return assets
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to get project assets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


async def process_project_videos(project_name: str, job_id: str):
    """Background task to process all unprocessed videos in a project."""
    try:
        job_status[job_id] = {
            "status": "processing",
            "progress": 0,
            "errors": []
        }
        
        # Get unprocessed videos
        unprocessed = project_manager.get_unprocessed_videos(project_name)
        total_videos = len(unprocessed)
        
        if total_videos == 0:
            job_status[job_id]["status"] = "completed"
            return
        
        # Process each video
        for i, video_path in enumerate(unprocessed):
            try:
                # Copy video to project sources if not already there
                project_path = project_manager.get_project_path(project_name)
                sources_dir = project_path / "sources"
                
                if video_path.parent != sources_dir:
                    dest_path = sources_dir / video_path.name
                    shutil.copy2(video_path, dest_path)
                    video_path = dest_path
                
                # Process the video
                result = await video_pipeline.process_video(str(video_path), project_name)
                
                # Update progress
                job_status[job_id]["progress"] = int((i + 1) / total_videos * 100)
                
            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                job_status[job_id]["errors"].append(str(e))
        
        job_status[job_id]["status"] = "completed"
        
    except Exception as e:
        logger.error(f"Project processing failed: {e}")
        job_status[job_id]["status"] = "failed"
        job_status[job_id]["errors"].append(str(e))


@app.post("/projects/{project_name}/process")
async def process_project(project_name: str, background_tasks: BackgroundTasks, force: bool = False):
    """Process all unprocessed videos in a project."""
    try:
        # Check if project exists
        project_path = project_manager.get_project_path(project_name)
        
        # Get unprocessed videos
        unprocessed = project_manager.get_unprocessed_videos(project_name)
        
        if not unprocessed and not force:
            return {
                "message": "No unprocessed videos found",
                "videos_to_process": 0,
                "job_id": None
            }
        
        # Create job
        job_id = str(uuid.uuid4())
        
        # Start background processing
        background_tasks.add_task(process_project_videos, project_name, job_id)
        
        return {
            "message": f"Processing {len(unprocessed)} videos",
            "videos_to_process": len(unprocessed),
            "job_id": job_id
        }
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to start project processing: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_name}/storyboards")
async def list_storyboards(project_name: str):
    """List all storyboards in a project."""
    try:
        storyboards = project_manager.list_storyboards(project_name)
        return storyboards
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to list storyboards: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/projects/{project_name}/storyboards")
async def save_storyboard(project_name: str, storyboard: Dict[str, Any] = Body(...)):
    """Save a storyboard to a project."""
    try:
        filename = project_manager.save_storyboard(project_name, storyboard)
        return {"filename": filename, "message": "Storyboard saved successfully"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to save storyboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/projects/{project_name}/storyboards/{filename}")
async def load_storyboard(project_name: str, filename: str):
    """Load a specific storyboard."""
    try:
        storyboard = project_manager.load_storyboard(project_name, filename)
        return storyboard
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Failed to load storyboard: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8765)