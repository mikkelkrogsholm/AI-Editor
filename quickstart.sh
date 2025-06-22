#!/bin/bash
# Quick start script for AI-Klipperen

echo "🎬 AI-Klipperen Quick Start"
echo "=========================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $python_version"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.12 -m venv venv
fi

# Activate venv
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Check Ollama
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is installed"
    echo "Available models:"
    ollama list
else
    echo "⚠️  Ollama not found. Please install with: brew install ollama"
fi

# Check FFmpeg
if command -v ffmpeg &> /dev/null; then
    echo "✓ FFmpeg is installed"
else
    echo "⚠️  FFmpeg not found. Please install with: brew install ffmpeg"
fi

echo ""
echo "🚀 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Start Ollama service: ollama serve"
echo "2. Download models: ollama pull llava:latest && ollama pull nomic-embed-text && ollama pull mistral:latest"
echo "3. Start API server: ai-clip server"
echo "4. (Optional) Start UI: streamlit run ui/app.py"
echo ""
echo "Example usage:"
echo "  ai-clip ingest video.mp4 --project myproject"
echo "  ai-clip search 'sunset crowd'"
echo "  ai-clip storyboard 'Create 60s energetic montage' --output story.json"
echo "  ai-clip render story.json --preview -o preview.mp4"