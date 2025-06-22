"""
AI-Klipperen CLI - Command line interface for video processing and editing.
"""
import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
import json
import httpx
import time
from typing import List, Optional
import shutil

# Initialize Typer app
app = typer.Typer(
    name="ai-clip",
    help="AI-powered video clipper and editor",
    add_completion=False
)

# Rich console for pretty output
console = Console()

# API base URL
API_BASE_URL = "http://localhost:8000"


def wait_for_job(job_id: str, job_type: str = "processing") -> bool:
    """Wait for a background job to complete."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"[cyan]{job_type.capitalize()}...", total=100)
        
        while True:
            try:
                # Get job status
                response = httpx.get(f"{API_BASE_URL}/ingest/status/{job_id}")
                if response.status_code == 200:
                    status = response.json()
                    
                    # Update progress
                    progress.update(task, completed=status.get("progress", 0))
                    
                    # Check if completed
                    if status["status"] == "completed":
                        progress.update(task, completed=100)
                        return True
                    elif status["status"] == "failed":
                        console.print(f"[red]Job failed: {status.get('errors', [])}")
                        return False
                
                time.sleep(1)
                
            except Exception as e:
                console.print(f"[red]Error checking job status: {e}")
                return False


@app.command()
def ingest(
    path: Path = typer.Argument(..., help="Path to video file or directory"),
    project: str = typer.Option(..., "--project", "-p", help="Project name"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process directories recursively"),
    copy: bool = typer.Option(True, "--copy/--no-copy", help="Copy files to project (vs. process in place)")
):
    """Ingest videos into a project for processing."""
    console.print(Panel.fit(f"[bold cyan]Ingesting videos for project: {project}[/bold cyan]"))
    
    # Find video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    video_files = []
    
    if path.is_file():
        if path.suffix.lower() in video_extensions:
            video_files.append(path)
    elif path.is_dir():
        pattern = "**/*" if recursive else "*"
        for ext in video_extensions:
            video_files.extend(path.glob(f"{pattern}{ext}"))
            video_files.extend(path.glob(f"{pattern}{ext.upper()}"))
    
    if not video_files:
        console.print("[red]No video files found![/red]")
        raise typer.Exit(1)
    
    console.print(f"Found [green]{len(video_files)}[/green] video files")
    
    # Import videos to project
    imported_count = 0
    for video_file in video_files:
        try:
            if copy:
                # Import to project
                response = httpx.post(
                    f"{API_BASE_URL}/projects/{project}/import",
                    json={
                        "file_path": str(video_file),
                        "asset_type": "video"
                    }
                )
                if response.status_code == 200:
                    imported_count += 1
                    console.print(f"  [green]✓[/green] Imported: {video_file.name}")
                else:
                    console.print(f"  [red]✗[/red] Failed: {video_file.name}")
        except Exception as e:
            console.print(f"  [red]✗[/red] Error importing {video_file.name}: {e}")
    
    console.print(f"\n[green]Imported {imported_count}/{len(video_files)} videos[/green]")
    
    # Process the project
    if imported_count > 0:
        console.print("\n[cyan]Processing imported videos...[/cyan]")
        try:
            response = httpx.post(f"{API_BASE_URL}/projects/{project}/process")
            if response.status_code == 200:
                result = response.json()
                if result['job_id']:
                    wait_for_job(result['job_id'], "processing")
                    console.print("[green]✓ Processing complete![/green]")
        except Exception as e:
            console.print(f"[red]Processing failed: {e}[/red]")


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    results: int = typer.Option(20, "--results", "-n", help="Number of results"),
    quality: float = typer.Option(5.0, "--quality", "-q", help="Minimum quality threshold"),
    type: str = typer.Option("hybrid", "--type", "-t", help="Search type: frames, asr, or hybrid"),
    project: Optional[str] = typer.Option(None, "--project", "-p", help="Filter by project"),
    aspect_ratio: Optional[str] = typer.Option(None, "--aspect-ratio", help="Filter by aspect ratio (e.g., 16:9)"),
    orientation: Optional[str] = typer.Option(None, "--orientation", help="Filter by orientation (landscape/portrait/square)"),
    min_resolution: Optional[str] = typer.Option(None, "--min-resolution", help="Minimum resolution (e.g., 1080p, 4K)")
):
    """Search for clips in the database."""
    console.print(f"[cyan]Searching for:[/cyan] {query}")
    
    try:
        search_params = {
            "query": query,
            "n_results": results,
            "quality_threshold": quality,
            "search_type": type
        }
        
        # Add optional filters
        if project:
            search_params["project"] = project
        if aspect_ratio:
            search_params["aspect_ratio"] = aspect_ratio
        if orientation:
            search_params["orientation"] = orientation
        if min_resolution:
            search_params["min_resolution"] = min_resolution
        
        response = httpx.post(
            f"{API_BASE_URL}/search",
            json=search_params
        )
        
        if response.status_code == 200:
            results = response.json()
            
            # Display frame results
            if results.get("frames"):
                table = Table(title="Frame Results")
                table.add_column("ID", style="cyan")
                table.add_column("Timestamp", style="green")
                table.add_column("Caption", style="white")
                table.add_column("Quality", style="yellow")
                
                for frame in results["frames"][:10]:  # Show top 10
                    table.add_row(
                        frame["id"],
                        frame["metadata"]["timestamp"],
                        frame["caption"][:60] + "...",
                        f"{frame['metadata']['quality']:.1f}"
                    )
                
                console.print(table)
            
            # Display ASR results
            if results.get("asr_segments"):
                table = Table(title="Transcript Results")
                table.add_column("ID", style="cyan")
                table.add_column("Time", style="green")
                table.add_column("Text", style="white")
                
                for segment in results["asr_segments"][:10]:
                    start = segment["metadata"]["start"]
                    end = segment["metadata"]["end"]
                    table.add_row(
                        segment["id"],
                        f"{start:.1f}s - {end:.1f}s",
                        segment["text"][:60] + "..."
                    )
                
                console.print(table)
                
        else:
            console.print(f"[red]Search failed: {response.text}[/red]")
            
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def storyboard(
    prompt: str = typer.Argument(..., help="Description of desired video"),
    project: str = typer.Option("storyboard", "--project", "-p", help="Project name"),
    duration: int = typer.Option(60, "--duration", "-d", help="Target duration in seconds"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output EDL file path")
):
    """Generate a storyboard from a text prompt."""
    console.print(Panel.fit(f"[bold cyan]Generating storyboard[/bold cyan]\n{prompt}"))
    
    try:
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": prompt,
                "conversation_id": f"storyboard_{int(time.time())}",
                "target_duration": duration
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Save storyboard if generated
            if result.get("storyboard"):
                storyboard = result["storyboard"]
                
                # Save to file if output specified
                if output:
                    with open(output, 'w') as f:
                        json.dump(storyboard, f, indent=2)
                    console.print(f"[green]Storyboard saved to:[/green] {output}")
                
                # Display summary
                if storyboard.get("timeline"):
                    table = Table(title="Generated Storyboard")
                    table.add_column("Clip", style="cyan")
                    table.add_column("Duration", style="green")
                    table.add_column("Transition", style="yellow")
                    
                    for i, clip in enumerate(storyboard["timeline"]):
                        duration = clip["out_point"] - clip["in_point"]
                        table.add_row(
                            clip["clip_id"],
                            f"{duration:.1f}s",
                            clip.get("transition", "cut")
                        )
                    
                    console.print(table)
            else:
                console.print(f"[yellow]Response:[/yellow] {result['response']}")
                
        else:
            console.print(f"[red]Generation failed: {response.text}[/red]")
            
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def render(
    edl_file: Path = typer.Argument(..., help="Path to EDL JSON file"),
    output: Path = typer.Option(Path("output.mp4"), "--output", "-o", help="Output video file"),
    preview: bool = typer.Option(False, "--preview", "-p", help="Render low-res preview"),
    download: bool = typer.Option(True, "--download", "-d", help="Download after rendering")
):
    """Render video from an EDL/storyboard file."""
    if not edl_file.exists():
        console.print(f"[red]EDL file not found: {edl_file}[/red]")
        raise typer.Exit(1)
    
    console.print(Panel.fit(f"[bold cyan]Rendering {'preview' if preview else 'final'} video[/bold cyan]"))
    
    try:
        # Load storyboard
        with open(edl_file, 'r') as f:
            storyboard = json.load(f)
        
        # Send render request
        endpoint = "/preview" if preview else "/render"
        response = httpx.post(
            f"{API_BASE_URL}{endpoint}",
            json={
                "storyboard": storyboard,
                "output_filename": output.name,
                "preview": preview
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            job_id = result["job_id"]
            
            console.print(f"[green]Render job started:[/green] {job_id}")
            
            # Wait for rendering
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console
            ) as progress:
                task = progress.add_task("[cyan]Rendering video...", total=None)
                
                while True:
                    status_response = httpx.get(f"{API_BASE_URL}/render/status/{job_id}")
                    if status_response.status_code == 200:
                        status = status_response.json()
                        
                        if status["status"] == "completed":
                            progress.update(task, description="[green]Rendering complete!")
                            break
                        elif status["status"] == "failed":
                            console.print(f"[red]Rendering failed: {status.get('error', 'Unknown error')}")
                            raise typer.Exit(1)
                    
                    time.sleep(2)
            
            # Download the file if requested
            if download:
                console.print(f"[cyan]Downloading to:[/cyan] {output}")
                
                download_response = httpx.get(
                    f"{API_BASE_URL}/download/{job_id}",
                    follow_redirects=True
                )
                
                if download_response.status_code == 200:
                    with open(output, 'wb') as f:
                        f.write(download_response.content)
                    console.print(f"[green][/green] Video saved to: {output}")
                else:
                    console.print("[red]Download failed![/red]")
            else:
                console.print(f"[green][/green] Video rendered successfully!")
                console.print(f"Download with: [cyan]ai-clip download {job_id}[/cyan]")
                
        else:
            console.print(f"[red]Render request failed: {response.text}[/red]")
            
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload")
):
    """Start the API server."""
    console.print(Panel.fit("[bold cyan]Starting AI-Klipperen API Server[/bold cyan]"))
    
    import uvicorn
    
    uvicorn.run(
        "backend.main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


@app.command()
def project(
    action: str = typer.Argument(..., help="Action: create, list, delete, process, assets"),
    name: Optional[str] = typer.Argument(None, help="Project name"),
    asset_type: Optional[str] = typer.Option(None, "--type", "-t", help="Asset type filter"),
    force: bool = typer.Option(False, "--force", "-f", help="Force reprocessing")
):
    """Manage projects and their assets."""
    
    if action == "create":
        if not name:
            console.print("[red]Project name required for create action[/red]")
            raise typer.Exit(1)
        
        try:
            response = httpx.post(f"{API_BASE_URL}/projects/create", json={"name": name})
            if response.status_code == 200:
                console.print(f"[green]✓ Created project: {name}[/green]")
                project_info = response.json()
                console.print(f"Location: {project_info['path']}")
            else:
                console.print(f"[red]Failed to create project: {response.json().get('detail')}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    elif action == "list":
        try:
            response = httpx.get(f"{API_BASE_URL}/projects")
            if response.status_code == 200:
                projects = response.json()
                
                if not projects:
                    console.print("[yellow]No projects found[/yellow]")
                    return
                
                table = Table(title="Projects")
                table.add_column("Name", style="cyan")
                table.add_column("Created", style="green")
                table.add_column("Videos", justify="right")
                table.add_column("Clips", justify="right")
                table.add_column("Frames", justify="right")
                
                for project in projects:
                    stats = project.get("stats", {})
                    table.add_row(
                        project["name"],
                        project["created"][:10],
                        str(stats.get("total_videos", 0)),
                        str(stats.get("total_clips", 0)),
                        str(stats.get("total_frames", 0))
                    )
                
                console.print(table)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    elif action == "delete":
        if not name:
            console.print("[red]Project name required for delete action[/red]")
            raise typer.Exit(1)
        
        if not force:
            confirm = typer.confirm(f"Are you sure you want to delete project '{name}'?")
            if not confirm:
                console.print("[yellow]Deletion cancelled[/yellow]")
                return
        
        try:
            response = httpx.delete(f"{API_BASE_URL}/projects/{name}")
            if response.status_code == 200:
                console.print(f"[green]✓ Deleted project: {name}[/green]")
            else:
                console.print(f"[red]Failed to delete project: {response.json().get('detail')}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    elif action == "process":
        if not name:
            console.print("[red]Project name required for process action[/red]")
            raise typer.Exit(1)
        
        try:
            response = httpx.post(f"{API_BASE_URL}/projects/{name}/process", json={"force": force})
            if response.status_code == 200:
                result = response.json()
                console.print(f"[green]Processing project: {name}[/green]")
                console.print(f"Videos to process: {result['videos_to_process']}")
                
                if result['job_id']:
                    wait_for_job(result['job_id'], "processing")
                else:
                    console.print("[yellow]No videos to process[/yellow]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    elif action == "assets":
        if not name:
            console.print("[red]Project name required for assets action[/red]")
            raise typer.Exit(1)
        
        try:
            params = {"asset_type": asset_type} if asset_type else {}
            response = httpx.get(f"{API_BASE_URL}/projects/{name}/assets", params=params)
            if response.status_code == 200:
                assets = response.json()
                
                for category, files in assets.items():
                    if files:
                        console.print(f"\n[bold cyan]{category}:[/bold cyan]")
                        for file in files:
                            console.print(f"  • {file}")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
    
    else:
        console.print(f"[red]Unknown action: {action}[/red]")
        console.print("Available actions: create, list, delete, process, assets")


@app.command()
def import_asset(
    file_path: Path = typer.Argument(..., help="Path to file to import"),
    project: str = typer.Option(..., "--project", "-p", help="Project name"),
    asset_type: str = typer.Option(..., "--type", "-t", help="Asset type: video, audio_music, audio_voiceover, still_logo, still_graphic, still_photo")
):
    """Import an asset into a project."""
    if not file_path.exists():
        console.print(f"[red]File not found: {file_path}[/red]")
        raise typer.Exit(1)
    
    valid_types = ["video", "audio_music", "audio_voiceover", "still_logo", "still_graphic", "still_photo"]
    if asset_type not in valid_types:
        console.print(f"[red]Invalid asset type. Choose from: {', '.join(valid_types)}[/red]")
        raise typer.Exit(1)
    
    try:
        response = httpx.post(
            f"{API_BASE_URL}/projects/{project}/import",
            json={
                "file_path": str(file_path),
                "asset_type": asset_type
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            console.print(f"[green]✓ Imported {asset_type}: {file_path.name}[/green]")
            console.print(f"Location: {result['path']}")
        else:
            console.print(f"[red]Import failed: {response.json().get('detail')}[/red]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")


@app.command()
def version():
    """Show version information."""
    console.print(Panel.fit(
        "[bold cyan]AI-Klipperen[/bold cyan]\n"
        "Version: 1.0.0\n"
        "AI-powered video clipper and editor"
    ))


@app.command()
def config(
    show: bool = typer.Option(False, "--show", "-s", help="Show current configuration"),
    edit: bool = typer.Option(False, "--edit", "-e", help="Edit configuration interactively"),
    export: Optional[Path] = typer.Option(None, "--export", help="Export config to file"),
    import_file: Optional[Path] = typer.Option(None, "--import", help="Import config from file"),
    list_models: bool = typer.Option(False, "--list-models", "-l", help="List available Ollama models")
):
    """Manage AI-Klipperen configuration."""
    from backend.config import settings, Settings
    from backend.pipeline import OllamaClient
    
    if list_models:
        console.print("[cyan]Checking available Ollama models...[/cyan]")
        try:
            client = OllamaClient()
            models = client.get_available_models()
            
            if models:
                table = Table(title="Available Ollama Models")
                table.add_column("Model Name", style="cyan")
                table.add_column("Recommended For", style="white")
                
                # Categorize models
                vision_models = [m for m in models if any(v in m for v in ["llava", "vision", "bakllava"])]
                embed_models = [m for m in models if any(e in m for e in ["embed", "nomic", "bge", "e5"])]
                chat_models = [m for m in models if any(c in m for c in ["llama", "mistral", "mixtral", "qwen", "phi"])]
                
                for model in vision_models:
                    table.add_row(model, "Vision/Caption")
                for model in embed_models:
                    table.add_row(model, "Embeddings")
                for model in chat_models:
                    if model not in vision_models and model not in embed_models:
                        table.add_row(model, "Chat/Storyboard")
                
                console.print(table)
            else:
                console.print("[yellow]No models found. Make sure Ollama is running.[/yellow]")
                
        except Exception as e:
            console.print(f"[red]Error connecting to Ollama: {e}[/red]")
            
    elif show:
        console.print(Panel.fit("[bold cyan]Current Configuration[/bold cyan]"))
        
        # Models
        console.print("\n[yellow]Models:[/yellow]")
        console.print(f"  Vision: {settings.models.vision_caption_model}")
        console.print(f"  Embeddings: {settings.models.text_embedding_model}")
        console.print(f"  Chat: {settings.models.chat_model}")
        
        # Processing
        console.print("\n[yellow]Processing:[/yellow]")
        console.print(f"  Frame extraction: {settings.processing.frame_extraction_fps} fps")
        console.print(f"  Quality threshold: {settings.processing.default_quality_threshold}")
        
        # Render
        console.print("\n[yellow]Rendering:[/yellow]")
        console.print(f"  Preview: {settings.render.preview_resolution} @ {settings.render.preview_fps}fps")
        console.print(f"  Final: {settings.render.final_resolution} @ {settings.render.final_fps}fps")
        
    elif export:
        settings.save_to_file(str(export))
        console.print(f"[green]Configuration exported to: {export}[/green]")
        
    elif import_file:
        if not import_file.exists():
            console.print(f"[red]File not found: {import_file}[/red]")
            raise typer.Exit(1)
            
        try:
            new_settings = Settings.load_from_file(str(import_file))
            # Save as default
            new_settings.save_to_file("config.json")
            console.print(f"[green]Configuration imported from: {import_file}[/green]")
        except Exception as e:
            console.print(f"[red]Failed to import config: {e}[/red]")
            
    elif edit:
        console.print("[yellow]Interactive configuration editing not yet implemented.[/yellow]")
        console.print("Please edit config.json directly or use environment variables.")
    else:
        console.print("[yellow]Use --help to see available options.[/yellow]")


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()