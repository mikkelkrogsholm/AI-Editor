# 🎬 AI-Klipperen

AI-powered video clipper and editor that uses local AI models to understand, search, and automatically edit videos.

## 🌟 Features

- **Video Analysis**: Automatically extract frames and generate captions using vision AI
- **Semantic Search**: Find clips using natural language queries
- **AI Storyboarding**: Generate video storyboards through chat interface
- **Automated Editing**: Render professional videos with transitions
- **Multiple Interfaces**: CLI, Web API, and Streamlit UI

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
ollama run minicpm-v:8b-2.6-q4_0  # Vision captioning
ollama pull nomic-embed-text      # Text embeddings  
ollama pull mistral:latest        # Chat and storyboarding
```

## 🚀 Quick Start

### 1. Start the API server:
```bash
ai-clip server
# Or directly with: uvicorn backend.main:app --reload
```

### 2. Process videos (CLI):
```bash
# Process a single video
ai-clip ingest video.mp4 --project myproject

# Process a directory
ai-clip ingest /path/to/videos/ --project myproject --recursive
```

### 3. Search for clips:
```bash
ai-clip search "sunset crowd energetic" --results 10
```

### 4. Generate storyboard:
```bash
ai-clip storyboard "Create a 60-second energetic montage" --output storyboard.json
```

### 5. Render video:
```bash
# Preview (low-res with watermark)
ai-clip render storyboard.json --preview --output preview.mp4

# Final render (high quality)
ai-clip render storyboard.json --output final.mp4
```

## 🎨 Streamlit UI

For a graphical interface, run:
```bash
streamlit run ui/app.py
```

Then open http://localhost:8501 in your browser.

## 📡 API Endpoints

The FastAPI backend provides these endpoints:

- `POST /ingest` - Process videos
- `POST /search` - Search for clips
- `POST /chat` - Chat interface for storyboard generation
- `POST /preview` - Generate preview render
- `POST /render` - Generate final render
- `GET /projects` - List all projects

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

## 🔧 Configuration

### Environment Variables

- `OLLAMA_HOST`: Ollama server URL (default: http://localhost:11434)
- `CHROMA_PERSIST_DIR`: ChromaDB storage location (default: ./chroma_db)

### Processing Settings

- Frame extraction: 1 FPS by default
- Video formats: MP4, MOV, AVI, MKV, WebM
- Output format: MP4 (H.264)

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

## 📞 Support

For issues and feature requests, please use the GitHub issue tracker.