"""
Streamlit UI for AI-Klipperen.
"""
import streamlit as st
import httpx
import json
from pathlib import Path
import tempfile
import time
from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="AI-Klipperen",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = "http://localhost:8000"

# Initialize session state
if "current_project" not in st.session_state:
    st.session_state.current_project = "default"
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_storyboard" not in st.session_state:
    st.session_state.current_storyboard = None
if "job_status" not in st.session_state:
    st.session_state.job_status = {}


def check_api_connection():
    """Check if API is available."""
    try:
        response = httpx.get(f"{API_BASE_URL}/", timeout=2.0)
        return response.status_code == 200
    except:
        return False


def main():
    """Main Streamlit app."""
    st.title("🎬 AI-Klipperen")
    st.markdown("AI-powered video clipper and editor")
    
    # Check API connection
    if not check_api_connection():
        st.error("⚠️ Cannot connect to API server. Please start the server with: `ai-clip server`")
        st.stop()
    
    # Sidebar
    with st.sidebar:
        st.header("Project Settings")
        
        # Project selection
        st.session_state.current_project = st.text_input(
            "Project Name",
            value=st.session_state.current_project
        )
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["📤 Upload & Process", "🔍 Search Library", "💬 Chat & Storyboard", "🎥 Render", "⚙️ Settings"]
        )
    
    # Main content
    if page == "📤 Upload & Process":
        upload_page()
    elif page == "🔍 Search Library":
        search_page()
    elif page == "💬 Chat & Storyboard":
        chat_page()
    elif page == "🎥 Render":
        render_page()
    elif page == "⚙️ Settings":
        settings_page()


def upload_page():
    """Upload and process videos."""
    st.header("📤 Upload & Process Videos")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose video files",
        type=['mp4', 'mov', 'avi', 'mkv', 'webm'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"Selected {len(uploaded_files)} files")
        
        if st.button("🚀 Process Videos", type="primary"):
            with st.spinner("Uploading and processing..."):
                # Save files temporarily
                temp_paths = []
                for file in uploaded_files:
                    temp_path = Path(tempfile.mktemp(suffix=f"_{file.name}"))
                    with open(temp_path, 'wb') as f:
                        f.write(file.read())
                    temp_paths.append(str(temp_path))
                
                # Send ingest request
                try:
                    response = httpx.post(
                        f"{API_BASE_URL}/ingest",
                        json={
                            "project_name": st.session_state.current_project,
                            "video_paths": temp_paths
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        job_id = result["job_id"]
                        
                        # Monitor progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        while True:
                            status_response = httpx.get(f"{API_BASE_URL}/ingest/status/{job_id}")
                            if status_response.status_code == 200:
                                status = status_response.json()
                                
                                # Update progress
                                progress = status.get("progress", 0) / 100
                                progress_bar.progress(progress)
                                status_text.text(f"Status: {status['status']} - {status.get('completed', 0)}/{status.get('total', 0)} videos")
                                
                                if status["status"] == "completed":
                                    st.success("✅ Videos processed successfully!")
                                    break
                                elif status["status"] == "failed":
                                    st.error(f"❌ Processing failed: {status.get('errors', [])}")
                                    break
                            
                            time.sleep(1)
                    else:
                        st.error(f"Failed to start processing: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")
    
    # Show existing videos
    st.divider()
    st.subheader("📁 Existing Videos")
    
    try:
        response = httpx.get(f"{API_BASE_URL}/projects")
        if response.status_code == 200:
            projects = response.json().get("projects", [])
            if projects:
                df = pd.DataFrame(projects)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No videos processed yet")
    except Exception as e:
        st.error(f"Failed to load projects: {e}")


def search_page():
    """Search for clips in the library."""
    st.header("🔍 Search Library")
    
    # Search interface
    col1, col2, col3 = st.columns([3, 1, 1])
    
    with col1:
        query = st.text_input("Search query", placeholder="e.g., sunset, crowd, energetic...")
    
    with col2:
        search_type = st.selectbox("Search type", ["hybrid", "frames", "asr"])
    
    with col3:
        quality_threshold = st.slider("Min quality", 0.0, 10.0, 5.0, 0.5)
    
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching..."):
            try:
                response = httpx.post(
                    f"{API_BASE_URL}/search",
                    json={
                        "query": query,
                        "n_results": 20,
                        "quality_threshold": quality_threshold,
                        "search_type": search_type
                    },
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    results = response.json()
                    
                    # Display frame results
                    if results.get("frames"):
                        st.subheader("📸 Frame Results")
                        
                        # Create columns for thumbnail grid
                        cols = st.columns(4)
                        for i, frame in enumerate(results["frames"][:12]):
                            with cols[i % 4]:
                                st.markdown(f"**{frame['id']}**")
                                st.caption(f"⏱️ {frame['metadata']['timestamp']}")
                                st.caption(f"⭐ {frame['metadata']['quality']:.1f}")
                                st.text(frame['caption'][:100] + "...")
                                st.markdown("---")
                    
                    # Display ASR results
                    if results.get("asr_segments"):
                        st.subheader("🎤 Transcript Results")
                        
                        for segment in results["asr_segments"][:10]:
                            start = segment["metadata"]["start"]
                            end = segment["metadata"]["end"]
                            st.markdown(f"**{segment['id']}** ({start:.1f}s - {end:.1f}s)")
                            st.text(segment['text'])
                            st.markdown("---")
                    
                    if not results.get("frames") and not results.get("asr_segments"):
                        st.info("No results found")
                        
                else:
                    st.error(f"Search failed: {response.text}")
                    
            except Exception as e:
                st.error(f"Error: {e}")


def chat_page():
    """Chat interface for storyboard generation."""
    st.header("💬 Chat & Storyboard Generation")
    
    # Chat interface
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])
            
            # Show storyboard if available
            if message.get("storyboard"):
                with st.expander("📋 View Storyboard"):
                    st.json(message["storyboard"])
    
    # Chat input
    if prompt := st.chat_input("Describe the video you want to create..."):
        # Add user message
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("Generating..."):
                try:
                    response = httpx.post(
                        f"{API_BASE_URL}/chat",
                        json={
                            "message": prompt,
                            "project_name": st.session_state.current_project
                        },
                        timeout=60.0
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display response
                        st.write(result["response"])
                        
                        # Store storyboard if generated
                        if result.get("storyboard"):
                            st.session_state.current_storyboard = result["storyboard"]
                            
                            # Show storyboard details
                            st.success("✅ Storyboard generated!")
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Duration", f"{result['storyboard']['duration']:.1f}s")
                            with col2:
                                st.metric("Clips", len(result['storyboard']['timeline']))
                            with col3:
                                st.metric("Project", result['storyboard']['project_name'])
                            
                            # Preview timeline
                            with st.expander("📋 View Timeline"):
                                timeline_data = []
                                for clip in result['storyboard']['timeline']:
                                    timeline_data.append({
                                        "Clip ID": clip['clip_id'],
                                        "Duration": f"{clip['out_point'] - clip['in_point']:.1f}s",
                                        "Transition": clip.get('transition', 'cut')
                                    })
                                
                                df = pd.DataFrame(timeline_data)
                                st.dataframe(df, use_container_width=True)
                            
                            # Save storyboard button
                            if st.button("💾 Save Storyboard"):
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                filename = f"storyboard_{timestamp}.json"
                                
                                st.download_button(
                                    label="Download EDL JSON",
                                    data=json.dumps(result['storyboard'], indent=2),
                                    file_name=filename,
                                    mime="application/json"
                                )
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "role": "assistant",
                            "content": result["response"],
                            "storyboard": result.get("storyboard")
                        })
                        
                    else:
                        st.error(f"Failed to generate response: {response.text}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")


def render_page():
    """Render videos from storyboards."""
    st.header("🎥 Render Video")
    
    # Check if we have a storyboard
    if not st.session_state.current_storyboard:
        st.info("💡 Generate a storyboard in the Chat section first")
        
        # Option to upload EDL
        uploaded_edl = st.file_uploader("Or upload an EDL JSON file", type=['json'])
        
        if uploaded_edl:
            try:
                st.session_state.current_storyboard = json.load(uploaded_edl)
                st.success("✅ Storyboard loaded!")
            except Exception as e:
                st.error(f"Failed to load EDL: {e}")
    
    if st.session_state.current_storyboard:
        # Show current storyboard info
        st.subheader("Current Storyboard")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Project", st.session_state.current_storyboard['project_name'])
        with col2:
            st.metric("Duration", f"{st.session_state.current_storyboard['duration']:.1f}s")
        with col3:
            st.metric("Clips", len(st.session_state.current_storyboard['timeline']))
        
        # Render options
        st.divider()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎬 Preview Render")
            st.write("Low-resolution with watermark")
            
            if st.button("Generate Preview", type="secondary"):
                with st.spinner("Rendering preview..."):
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"preview_{timestamp}.mp4"
                        
                        response = httpx.post(
                            f"{API_BASE_URL}/preview",
                            json={
                                "storyboard": st.session_state.current_storyboard,
                                "output_filename": filename,
                                "preview": True
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            job_id = result["job_id"]
                            
                            # Wait for completion
                            while True:
                                status_response = httpx.get(f"{API_BASE_URL}/render/status/{job_id}")
                                if status_response.status_code == 200:
                                    status = status_response.json()
                                    
                                    if status["status"] == "completed":
                                        st.success("✅ Preview ready!")
                                        
                                        # Download button
                                        download_response = httpx.get(f"{API_BASE_URL}/download/{job_id}")
                                        if download_response.status_code == 200:
                                            st.download_button(
                                                label="⬇️ Download Preview",
                                                data=download_response.content,
                                                file_name=filename,
                                                mime="video/mp4"
                                            )
                                        break
                                    elif status["status"] == "failed":
                                        st.error(f"Rendering failed: {status.get('error')}")
                                        break
                                
                                time.sleep(2)
                        else:
                            st.error(f"Failed to start render: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with col2:
            st.subheader("🎯 Final Render")
            st.write("Full resolution, high quality")
            
            if st.button("Generate Final", type="primary"):
                with st.spinner("Rendering final video... This may take a while."):
                    try:
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"final_{timestamp}.mp4"
                        
                        response = httpx.post(
                            f"{API_BASE_URL}/render",
                            json={
                                "storyboard": st.session_state.current_storyboard,
                                "output_filename": filename,
                                "preview": False
                            }
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            job_id = result["job_id"]
                            
                            # Wait for completion with progress
                            progress_bar = st.progress(0)
                            start_time = time.time()
                            
                            while True:
                                status_response = httpx.get(f"{API_BASE_URL}/render/status/{job_id}")
                                if status_response.status_code == 200:
                                    status = status_response.json()
                                    
                                    # Estimate progress based on time
                                    elapsed = time.time() - start_time
                                    estimated_progress = min(elapsed / 60, 0.95)  # Assume ~1 minute
                                    progress_bar.progress(estimated_progress)
                                    
                                    if status["status"] == "completed":
                                        progress_bar.progress(1.0)
                                        st.success("✅ Final video ready!")
                                        
                                        # Download button
                                        download_response = httpx.get(f"{API_BASE_URL}/download/{job_id}")
                                        if download_response.status_code == 200:
                                            st.download_button(
                                                label="⬇️ Download Final Video",
                                                data=download_response.content,
                                                file_name=filename,
                                                mime="video/mp4"
                                            )
                                        break
                                    elif status["status"] == "failed":
                                        st.error(f"Rendering failed: {status.get('error')}")
                                        break
                                
                                time.sleep(2)
                        else:
                            st.error(f"Failed to start render: {response.text}")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")


def settings_page():
    """Settings and configuration."""
    st.header("⚙️ Settings")
    
    # API Settings
    st.subheader("API Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("API URL", value=API_BASE_URL, disabled=True)
    with col2:
        if st.button("Test Connection"):
            if check_api_connection():
                st.success("✅ Connected to API")
            else:
                st.error("❌ Cannot connect to API")
    
    # Model Settings
    st.subheader("Model Configuration")
    st.info("Models are managed through Ollama. Use the CLI to download models.")
    
    st.code("""
# Download required models:
ollama pull llava:latest      # Vision captioning
ollama pull nomic-embed-text  # Text embeddings
ollama pull mistral:latest    # Chat and storyboarding
""")
    
    # Processing Settings
    st.subheader("Processing Settings")
    
    col1, col2 = st.columns(2)
    with col1:
        fps = st.number_input("Frame extraction FPS", min_value=0.1, max_value=10.0, value=1.0, step=0.1)
        st.caption("Frames per second to extract from videos")
    
    with col2:
        quality_threshold = st.slider("Default quality threshold", 0.0, 10.0, 5.0, 0.5)
        st.caption("Minimum quality score for search results")
    
    # About
    st.divider()
    st.subheader("About AI-Klipperen")
    st.markdown("""
    **AI-Klipperen** is an AI-powered video clipper and editor that can:
    
    - 📤 Import and analyze video content
    - 🔍 Search clips using natural language
    - 💬 Generate storyboards through chat
    - 🎥 Render professional videos automatically
    
    Built with Python, FastAPI, Streamlit, and local AI models via Ollama.
    """)


if __name__ == "__main__":
    main()