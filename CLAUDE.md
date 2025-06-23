# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important: Use Context7 for Documentation

**ALWAYS use the Context7 MCP tool to search for up-to-date documentation when:**
- Building new features or integrations
- Debugging issues with dependencies
- Looking for best practices or API usage
- Troubleshooting errors

**How to use Context7:**
1. First use `mcp__context7__resolve-library-id` to find the library
2. Then use `mcp__context7__get-library-docs` with relevant topics
3. This ensures you get the latest, accurate documentation instead of relying on training data

**Example:**
```
# For MoviePy issues:
mcp__context7__resolve-library-id("moviepy")
mcp__context7__get-library-docs("/zulko/moviepy", "ImageMagick error environment")
```

## Project Overview

AI-Klipperen is an AI-powered video clipper and editor that processes videos locally using Ollama models to understand content, enable semantic search, and automatically generate edited videos through a chat interface.

## Architecture

The system consists of four main components that interact through a REST API:

1. **Video Processing Pipeline** (`backend/pipeline.py`)
   - Extracts frames at 1 FPS using FFmpeg
   - Generates captions using Ollama vision models (LLaVA)
   - Creates embeddings using text models (nomic-embed-text)
   - Stores frame metadata and embeddings in ChromaDB

2. **Vector Search System** (`backend/vector.py`)
   - ChromaDB collections for frames and ASR segments
   - Supports hybrid search (text + embedding similarity)
   - Maintains video-to-segment relationships

3. **Storyboard Generation** (`backend/chat_tools_v2.py`)
   - Uses enhanced LLM (qwen2.5:14b) with function calling
   - Validates all clip IDs exist before creating timeline
   - Intelligent fallback ensures valid clips are always found
   - Generates EDL JSON with timeline, transitions, and metadata

4. **Video Rendering** (`backend/render.py`)
   - MoviePy-based composition from storyboard
   - Supports transitions (cut, fade, dissolve)
   - Two modes: preview (640x360 with watermark) and final (1920x1080)

## Essential Commands

```bash
# Development setup
./quickstart.sh  # One-time setup
source venv/bin/activate  # Activate environment

# Start services
ollama serve  # In separate terminal - required for AI models
ai-clip server  # Or: uvicorn backend.main:app --reload

# Video processing
ai-clip ingest video.mp4 --project myproject
ai-clip ingest /path/to/videos/ --recursive --project myproject

# Search and storyboard
ai-clip search "sunset crowd energetic" --results 20 --quality 5.0
ai-clip storyboard "60 second energetic montage" --output story.json

# Rendering
ai-clip render story.json --preview -o preview.mp4  # Quick preview
ai-clip render story.json -o final.mp4  # High quality

# UI alternative
streamlit run ui/app.py

# Testing
pytest tests/  # All tests
pytest tests/test_basic.py::test_vector_store_init -v  # Single test
```

## Key Technical Details

### Ollama Model Requirements
- `minicpm-v:8b-2.6-q4_0` - Vision captioning (8B parameter vision model)
- `snowflake-arctic-embed2:latest` - Text embeddings (1024-dim)
- `qwen2.5:14b` - Chat and storyboard generation with function calling (14B model)
- `deepseek-r1:32b` - Alternative for complex reasoning tasks (32B model)

### Data Flow
1. Video → FFmpeg → Frames + Audio
2. Frames → Vision Model → Captions → Embeddings → ChromaDB
3. User Query → Text Embedding → Vector Search → Matched Clips
4. Matched Clips → LLM → Storyboard JSON → MoviePy → Output Video

### API Endpoints
- `POST /ingest` - Process videos (returns job_id for tracking)
- `POST /search` - Semantic search (supports frames/asr/hybrid)
- `POST /chat` - Storyboard generation via chat
- `POST /preview` & `POST /render` - Video rendering (async with job tracking)
- `GET /ingest/status/{job_id}` & `GET /render/status/{job_id}` - Job monitoring

### Background Jobs
Long-running operations (ingest, render) use FastAPI BackgroundTasks with job status tracking. Poll status endpoints for progress.

### ChromaDB Collections
- `frames`: Stores frame metadata (video, timestamp, caption, tags, mood, quality)
- `asr_segments`: Stores transcript segments (currently placeholder)

### Storyboard Format
```json
{
  "project_name": "string",
  "duration": 60.0,
  "timeline": [
    {"clip_id": "vid01_f0123", "in_point": 0.0, "out_point": 3.5, "transition": "cut"}
  ],
  "voiceover": "optional script",
  "music_cue": "optional.mp3",
  "notes": "Director's notes"
}
```

## Development Notes

- The system is designed for MacBook Pro with 128GB RAM to handle multiple Ollama models
- Frame extraction defaults to 1 FPS (configurable in pipeline)
- Quality scores (0-10) are derived from caption analysis
- ASR/transcription is placeholder - ready for Whisper integration
- All video paths must be absolute, not relative
- Temporary files are auto-cleaned in pipeline and render modules

## Session Notes (2025-01-06)

### Completed:
- ✅ Implemented complete project-based storage system with organized folders
- ✅ Added technical metadata extraction (resolution, aspect ratio, FPS, codecs)
- ✅ Created clip extraction (10-second segments with 2-second overlap)
- ✅ Enhanced search with filters for aspect ratio, orientation, resolution
- ✅ Updated CLI with project management commands
- ✅ Added support for multiple asset types (video, audio, stills)
- ✅ Changed default port from 8000 to 8765 to avoid conflicts
- ✅ Fixed server connection issue - server now runs properly on port 8765

## Session Notes (2025-06-22/23)

### Completed Today:
- ✅ Successfully tested project system with user's videos
- ✅ Processed 2 videos, created 7 clips, analyzed 46 frames
- ✅ Fixed UI to use project dropdown instead of text input
- ✅ Implemented automatic storyboard saving to projects
- ✅ Added storyboard browser in render page
- ✅ Fixed ChromaDB query issue with multiple filters (using $and operator)
- ✅ Chat-based storyboard generation working correctly
- ✅ All Danish UI improvements implemented

### ✅ MoviePy/ImageMagick Issue RESOLVED (2025-06-23):
**Solution**: Environment variables were not preserved in FastAPI background tasks.

**Fix Applied**:
```python
# In background tasks (main.py):
os.environ["IMAGEMAGICK_BINARY"] = "/opt/homebrew/bin/magick"
os.environ["FFMPEG_BINARY"] = "/opt/miniconda3/bin/ffmpeg"
```

**Results**:
- ✅ Both synchronous and asynchronous rendering now work
- ✅ MoviePy can find ImageMagick in all contexts
- ✅ Preview and final render modes both functional
- ✅ Fixed final render to use concatenate_videoclips for full duration
- See tests/debug/DEBUG_MOVIEPY_ISSUE.md for full troubleshooting history

### ✅ Enhanced Storyboard Generation Implemented (2025-06-23):
**Improvements**:
- Implemented `EnhancedStoryboardGenerator` in chat_tools_v2.py
- Uses qwen2.5:14b model with function calling support
- Features intelligent clip selection based on mood/pace analysis
- Real frame IDs instead of generic placeholders
- Automatic fallback to intelligent generation if LLM fails

**Configuration**:
```python
# Updated config.py default:
chat_model: str = "qwen2.5:14b"
```

### Testing Complete:
- ✅ Preview rendering: 39s video with 10 clips
- ✅ Final rendering: Full duration videos (not truncated)
- ✅ Enhanced storyboard generation with real clip IDs
- ✅ API endpoints working correctly

### ✅ Chat Functionality Enhanced (2025-06-23):
**Improvements**:
- Chat now uses AI for all responses (not just storyboard generation)
- ✅ **Project Content Awareness** - AI knows about available videos and their content
- ✅ **Fixed Black Video Rendering** - All generated clips are validated before use
- ✅ **English-only Communication** - Removed Danish keywords for consistency
- Natural conversation flow with helpful guidance
- Maintains context about current project

**UI Chat Features**:
- Full conversational AI assistant with content knowledge
- Automatic storyboard generation when requested
- Saves storyboards automatically to project
- Shows timeline details and clip counts
- Validates all clips exist before rendering

### ✅ Black Video Rendering Fixed (2025-06-23):
**Root Cause**: AI was generating invalid clip IDs when searches returned no results

**Solution Implemented**:
1. **Clip Validation** in `chat_tools_v2.py`:
   - Validates all clip IDs exist in database
   - Falls back to intelligent selection if invalid IDs detected
   - Always finds valid clips even if searches fail

2. **Improved Search Fallbacks**:
   - Tries common keywords if initial search fails
   - Gets random clips from project as last resort
   - Ensures timeline always has valid clips

3. **Pre-render Validation** in `render.py`:
   - `validate_storyboard()` method checks all clips
   - New `/validate_storyboard` API endpoint
   - Clear error messages for missing clips

4. **Project Context** in chat:
   - `get_project_content_context()` provides content analysis
   - AI knows about moods, tags, and available footage
   - Better clip recommendations based on actual content

### API Updates:
- `POST /chat` - Enhanced with project content awareness
- `POST /validate_storyboard` - New endpoint to check clips before rendering

### Future Features:
- Audio/ASR processing (when Ollama supports it)
- Advanced editing (color correction, effects)
- Timeline editor UI
- Multiple export formats
- Conversation history persistence
- Manual clip selection in UI