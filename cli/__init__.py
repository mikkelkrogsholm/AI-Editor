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
    project: str = typer.Option("default", "--project", "-p", help="Project name"),
    recursive: bool = typer.Option(False, "--recursive", "-r", help="Process directories recursively")
):
    """Ingest videos for processing."""
    console.print(Panel.fit(f"[bold cyan]Ingesting videos for project: {project}[/bold cyan]"))
    
    # Find video files
    video_extensions = ['.mp4', '.mov', '.avi', '.mkv', '.webm']
    video_files = []
    
    if path.is_file():
        if path.suffix.lower() in video_extensions:
            video_files.append(str(path))
    elif path.is_dir():
        pattern = "**/*" if recursive else "*"
        for ext in video_extensions:
            video_files.extend([str(p) for p in path.glob(f"{pattern}{ext}")])
            video_files.extend([str(p) for p in path.glob(f"{pattern}{ext.upper()}")])
    
    if not video_files:
        console.print("[red]No video files found![/red]")
        raise typer.Exit(1)
    
    console.print(f"Found [green]{len(video_files)}[/green] video files")
    
    try:
        # Send ingest request
        response = httpx.post(
            f"{API_BASE_URL}/ingest",
            json={
                "project_name": project,
                "video_paths": video_files
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            console.print(f"[green]Job started:[/green] {result['job_id']}")
            
            # Wait for completion
            if wait_for_job(result['job_id'], "processing"):
                console.print("[green][/green] Videos processed successfully!")
            else:
                console.print("[red][/red] Processing failed!")
                raise typer.Exit(1)
        else:
            console.print(f"[red]API error: {response.text}[/red]")
            raise typer.Exit(1)
            
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        console.print("[yellow]Make sure the API server is running (uvicorn backend.main:app)[/yellow]")
        raise typer.Exit(1)


@app.command()
def search(
    query: str = typer.Argument(..., help="Search query"),
    results: int = typer.Option(20, "--results", "-n", help="Number of results"),
    quality: float = typer.Option(5.0, "--quality", "-q", help="Minimum quality threshold"),
    type: str = typer.Option("hybrid", "--type", "-t", help="Search type: frames, asr, or hybrid")
):
    """Search for clips in the database."""
    console.print(f"[cyan]Searching for:[/cyan] {query}")
    
    try:
        response = httpx.post(
            f"{API_BASE_URL}/search",
            json={
                "query": query,
                "n_results": results,
                "quality_threshold": quality,
                "search_type": type
            }
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
        # Send chat request
        response = httpx.post(
            f"{API_BASE_URL}/chat",
            json={
                "message": f"Generate a {duration}-second video: {prompt}",
                "project_name": project
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            
            if result.get("storyboard"):
                storyboard = result["storyboard"]
                
                # Display storyboard summary
                console.print(f"\n[green][/green] Storyboard created!")
                console.print(f"Duration: [yellow]{storyboard['duration']:.1f}s[/yellow]")
                console.print(f"Clips: [yellow]{len(storyboard['timeline'])}[/yellow]")
                console.print(f"\nNotes: {storyboard['notes']}")
                
                # Save to file if requested
                if output:
                    with open(output, 'w') as f:
                        json.dump(storyboard, f, indent=2)
                    console.print(f"\n[green]Saved to:[/green] {output}")
                else:
                    # Display timeline
                    table = Table(title="Timeline")
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
                    console.print(f"[green][/green] Video saved to: {output}")
                else:
                    console.print("[red]Download failed![/red]")
            else:
                console.print(f"[green][/green] Video rendered successfully!")
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
def projects():
    """List all projects in the database."""
    try:
        response = httpx.get(f"{API_BASE_URL}/projects")
        
        if response.status_code == 200:
            data = response.json()
            projects = data.get("projects", [])
            
            if projects:
                table = Table(title="Projects")
                table.add_column("Name", style="cyan")
                table.add_column("Path", style="white")
                table.add_column("Frames", style="green")
                
                for project in projects:
                    table.add_row(
                        project["name"],
                        project["path"],
                        str(project["frame_count"])
                    )
                
                console.print(table)
            else:
                console.print("[yellow]No projects found[/yellow]")
                
        else:
            console.print(f"[red]Failed to list projects: {response.text}[/red]")
            
    except httpx.RequestError as e:
        console.print(f"[red]Connection error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def version():
    """Show version information."""
    console.print(Panel.fit(
        "[bold cyan]AI-Klipperen[/bold cyan]\n"
        "Version: 1.0.0\n"
        "AI-powered video clipper and editor"
    ))


def main():
    """Main entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()