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