# 🎬 AI-Klipperen

[![Tests](https://github.com/mikkelkrogsholm/AI-Editor/actions/workflows/tests.yml/badge.svg)](https://github.com/mikkelkrogsholm/AI-Editor/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mikkelkrogsholm/AI-Editor/branch/main/graph/badge.svg)](https://codecov.io/gh/mikkelkrogsholm/AI-Editor)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

AI-powered video clipper and editor that uses local AI models to understand, search, and automatically edit videos.

## 🌟 Features

- **Video Analysis**: Automatically extract frames and generate captions using vision AI
- **Clip Extraction**: Pre-extract 10-second clips for fast editing
- **Technical Metadata**: Track resolution, aspect ratio, FPS, codecs, and more
- **Semantic Search**: Find clips using natural language queries
- **Advanced Filtering**: Search by aspect ratio, orientation, resolution
- **Project Organization**: Structured folders for all media assets
- **AI Storyboarding**: Generate video storyboards through chat interface
- **Automated Editing**: Render professional videos with transitions
- **Multiple Interfaces**: CLI, Web API, and Streamlit UI

## 📌 Current Status

**Working Features:**
- ✅ Project-based storage and organization
- ✅ Video import and processing
- ✅ Frame extraction and AI analysis
- ✅ Clip extraction (10-second segments)
- ✅ Technical metadata extraction
- ✅ Semantic search with filters
- ✅ Storyboard generation via chat
- ✅ Streamlit UI with project management

**Known Issues:**
- ⚠️ Video rendering fails with MoviePy/ImageMagick error
- ⚠️ Audio/ASR processing not yet implemented (waiting for Ollama support)

## 🖥️ System Requirements

- MacBook Pro (M3 Pro or similar) with 128GB unified memory
- Python 3.12+
- Ollama installed
- FFmpeg installed

## 📦 Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/ai-klipperen.git
cd ai-klipperen
```

2. **Create virtual environment:**
```bash
python3.12 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Install FFmpeg (if not already installed):**
```bash
brew install ffmpeg  # On macOS
```

5. **Install Ollama (if not already installed):**
```bash
brew install ollama  # On macOS
```

6. **Download AI models:**
```bash
ollama pull minicpm-v:8b-2.6-q4_0       # Vision captioning
ollama pull snowflake-arctic-embed2     # Text embeddings  
ollama pull deepseek-r1:32b             # Chat and storyboarding
```

## 🚀 Quick Start

### 1. Create a project:
```bash
ai-clip project create myproject
```

### 2. Start the API server:
```bash
ai-clip server
# Or directly with: uvicorn backend.main:app --port 8765
# Note: Server runs on port 8765 by default
```

### 3. Import and process videos:
```bash
# Import videos to project
ai-clip ingest /path/to/videos/ --project myproject --recursive

# Or import individual assets
ai-clip import-asset video.mp4 --project myproject --type video
ai-clip import-asset music.mp3 --project myproject --type audio_music
ai-clip import-asset logo.png --project myproject --type still_logo
```

### 4. Search for clips:
```bash
# Basic search
ai-clip search "sunset crowd energetic" --results 10

# Search with filters
ai-clip search "person dancing" --project myproject --aspect-ratio "9:16" --orientation portrait
ai-clip search "landscape" --min-resolution "4K" --quality 8.0
```

### 5. Generate storyboard:
```bash
ai-clip storyboard "Create a 60-second energetic montage" --output storyboard.json
```

### 6. Render video:
```bash
# Preview (low-res with watermark)
ai-clip render storyboard.json --preview --output preview.mp4

# Final render (high quality)
ai-clip render storyboard.json --output final.mp4
```

## 📁 Project Management

AI-Klipperen uses a project-based workflow where all media assets are organized in structured folders:

### Project Structure:
```
projects/myproject/
├── metadata.json         # Project information
├── sources/             # Original video files
├── clips/               # Pre-extracted 10-second clips
├── frames/              # Frame thumbnails for preview
├── audio/               # Audio assets
│   ├── extracted/       # Audio from videos
│   ├── voiceover/       # Voiceover recordings
│   └── music/           # Background music
├── stills/              # Still images
│   ├── logos/           # Company/brand logos
│   ├── graphics/        # Overlays and graphics
│   └── photos/          # Photography
└── renders/             # Rendered output videos
```

### Project Commands:
```bash
# Create a new project
ai-clip project create myproject

# List all projects
ai-clip project list

# View project assets
ai-clip project assets myproject

# Process unprocessed videos
ai-clip project process myproject

# Delete a project
ai-clip project delete myproject --force
```

### Benefits:
- **Fast Editing**: Pre-extracted clips ready for instant use
- **Organized**: All assets in one place
- **Portable**: Easy to backup or share projects
- **Efficient**: No need to reprocess videos

## 🎨 Streamlit UI

For a graphical interface, run:
```bash
streamlit run ui/app.py
```

Then open http://localhost:8501 in your browser.

## 📡 API Endpoints

The FastAPI backend provides these endpoints:

### Project Management:
- `POST /projects/create` - Create a new project
- `GET /projects` - List all projects
- `DELETE /projects/{name}` - Delete a project
- `POST /projects/{name}/import` - Import asset to project
- `GET /projects/{name}/assets` - List project assets
- `POST /projects/{name}/process` - Process project videos

### Video Processing:
- `POST /ingest` - Process videos (deprecated, use project endpoints)
- `POST /search` - Search for clips with filters
- `POST /chat` - Chat interface for storyboard generation
- `POST /preview` - Generate preview render
- `POST /render` - Generate final render

API documentation available at http://localhost:8000/docs

## 🏗️ Architecture

```
ai-klipperen/
├── backend/
│   ├── vector.py      # ChromaDB vector store
│   ├── pipeline.py    # Video processing pipeline
│   ├── chat_tools.py  # Storyboard generation
│   ├── render.py      # Video rendering
│   └── main.py        # FastAPI application
├── cli/
│   └── __init__.py    # Typer CLI commands
├── ui/
│   └── app.py         # Streamlit interface
└── tests/
    └── test_*.py      # Unit tests
```

## 🎙️ Audio Transcription (ASR)

Since Ollama doesn't support audio models yet, AI-Klipperen provides integration with:

### Recommended: whisper.cpp
Fast C++ implementation of Whisper:
```bash
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
make

# Download model
bash ./models/download-ggml-model.sh base

# Enable ASR in config
AI_CLIP_ENABLE_ASR=true
```

### Alternative Options:
- **faster-whisper**: `pip install faster-whisper` (GPU accelerated)
- **openai-whisper**: `pip install openai-whisper` (original implementation)

## 📊 How It Works - Processing Pipeline

### Processing Two Videos - Step by Step

#### 1. **Start the API server**
```bash
ai-clip server
# Or: uvicorn backend.main:app --reload
```

#### 2. **Process your videos**
```bash
# Process both videos into the same project
ai-clip ingest video1.mp4 --project myproject
ai-clip ingest video2.mp4 --project myproject

# Or process an entire directory
ai-clip ingest /path/to/videos/ --project myproject --recursive
```

### What Happens During Processing?

For each video, the pipeline performs:

1. **Frame Extraction** (default 1 FPS)
   - FFmpeg extracts frames as JPEG images
   - Stored temporarily with timestamps

2. **Image Analysis** (via Ollama)
   - Each frame sent to `minicpm-v:8b-2.6-q4_0`
   - Generates detailed descriptions of:
     - People and actions
     - Objects and settings
     - Mood and visual quality

3. **Metadata Extraction**
   - Tags extracted from descriptions
   - Mood detection (energetic, calm, happy, etc.)
   - Quality score calculation (0-10)

4. **Embedding Generation**
   - Descriptions converted to 1024-dimensional vectors
   - Using `snowflake-arctic-embed2` model

5. **Storage in ChromaDB**
   - Frame ID: `myproject_video1_f000100` (project_video_frame)
   - Metadata: timestamp, description, tags, mood, quality
   - Vector embedding for semantic search

6. **Audio Processing** (if enabled)
   - Audio extracted as 16kHz WAV
   - ASR transcription via whisper.cpp
   - Text segments stored with timestamps

### After Processing

#### Search for clips:
```bash
# Find specific scenes
ai-clip search "sunset with people" --results 10
ai-clip search "energetic concert mood" --quality-min 7.0
```

#### Generate storyboard via chat:
```bash
ai-clip storyboard "Create a 60-second energetic montage with the best scenes"
```

#### Render video:
```bash
# Preview (low resolution)
ai-clip render storyboard.json --preview

# Final (high quality)
ai-clip render storyboard.json --output final.mp4
```

### Data Flow Example

```
Video1.mp4 → 300 frames → 300 descriptions → 300 embeddings → ChromaDB
           → 30 clips   → Technical metadata → Searchable by resolution/aspect ratio
           → Audio WAV  → ASR transcription  → Text embeddings

Video2.mp4 → 450 frames → 450 descriptions → 450 embeddings → ChromaDB
           → 45 clips   → Technical metadata → Searchable by resolution/aspect ratio
           → Audio WAV  → ASR transcription  → Text embeddings
                                                                    ↓
                                        Multimodal searchable database
```

Each frame becomes a searchable entity with rich metadata, allowing you to find exact clips by:
- Natural language description
- Technical specifications (resolution, aspect ratio, orientation)
- Audio transcription content
- Visual quality scores

## 🔧 Configuration

AI-Klipperen supports multiple configuration methods:

### 1. Configuration Files

Copy `config.example.json` to `config.json` and customize:
```bash
cp config.example.json config.json
```

Or use environment variables by copying `.env.example`:
```bash
cp .env.example .env
```

### 2. CLI Configuration Management

```bash
# Show current configuration
ai-clip config --show

# List available Ollama models
ai-clip config --list-models

# Export configuration
ai-clip config --export my-config.json

# Import configuration
ai-clip config --import my-config.json
```

### 3. Key Settings

**Model Configuration:**
- `vision_caption_model`: Model for image captioning (default: minicpm-v:8b-2.6-q4_0)
- `text_embedding_model`: Model for embeddings (default: snowflake-arctic-embed2:latest)
- `chat_model`: Model for storyboard generation (default: deepseek-r1:32b)

**Processing Settings:**
- `frame_extraction_fps`: Frames per second to extract (default: 1.0)
- `default_quality_threshold`: Minimum quality for search (default: 5.0)

**Render Settings:**
- Preview: 640x360 @ 24fps with watermark
- Final: 1920x1080 @ 30fps, 10Mbps bitrate

### 4. Model Fallbacks

The system automatically falls back to available models if the configured model isn't found. You can specify fallback models in the configuration.

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- Ollama for local AI model serving
- ChromaDB for vector storage
- MoviePy for video editing
- FFmpeg for video processing

## 📈 Changelog

### Version 1.1.0 (2025)
- Added project-based storage system
- Implemented clip extraction (10-second segments)
- Added technical metadata extraction
- Enhanced search with aspect ratio/resolution filters
- Added support for audio and still image assets
- Improved rendering performance with pre-extracted clips
- Added ASR support options (whisper.cpp, faster-whisper)

### Version 1.0.0
- Initial release with basic video processing
- Ollama integration for AI models
- CLI and API interfaces

## 📞 Support

For issues and feature requests, please use the GitHub issue tracker.