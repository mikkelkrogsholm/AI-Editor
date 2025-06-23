# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

3. **Storyboard Generation** (`backend/chat_tools.py`)
   - Uses LLM (Mistral) to interpret user prompts
   - Searches relevant clips based on semantic understanding
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
- `deepseek-r1:32b` - Storyboard generation (32B reasoning model)

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

### Current Blocker - MoviePy/ImageMagick Rendering:
**Error**: `[Errno 2] No such file or directory: 'unset'`
- ImageMagick is installed (`/opt/homebrew/bin/magick`)
- Individual clip tests work fine in isolation
- Full rendering pipeline consistently fails on the last clip (20250620_185203_001.mp4)
- Tried multiple fixes without success:
  - Setting IMAGEMAGICK_BINARY environment variable
  - Configuring MoviePy settings before import
  - Disabling watermarks (preview_watermark=False)
  - Removing all transitions
  - Using concatenate_videoclips with 'chain' method instead of 'compose'
  - Skipping resize operations entirely
  
**Latest Findings**:
- The error occurs specifically when MoviePy processes the last clip in the storyboard
- Using 'chain' method works in isolated tests but still fails in full pipeline
- The error message suggests MoviePy is trying to execute literal string "unset"
- This might be related to async/background task environment variables

### TODO When Returning:
1. **Fix Rendering Issue** (Priority #1):
   ```bash
   # Option 1: Use pure FFmpeg instead of MoviePy
   # Option 2: Debug the environment variable issue
   # Option 3: Try different MoviePy version
   pip install moviepy==1.0.3  # Try older stable version
   ```

2. **Complete Testing**:
   - Once rendering works, test full pipeline end-to-end
   - Verify preview and final render modes
   - Test with different video formats

3. **Future Features**:
   - Audio/ASR processing (when Ollama supports it)
   - Advanced editing (color correction, effects)
   - Timeline editor UI
   - Multiple export formats