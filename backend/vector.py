"""
Vector database operations using ChromaDB for multimodal search.
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import json


class VectorStore:
    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client with persistence."""
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create collections for frames and ASR segments
        self.frame_collection = self.client.get_or_create_collection(
            name="frames",
            metadata={"hnsw:space": "cosine"}
        )
        
        self.asr_collection = self.client.get_or_create_collection(
            name="asr_segments",
            metadata={"hnsw:space": "cosine"}
        )
    
    def add_frame(
        self,
        frame_id: str,
        video_path: str,
        timestamp: str,
        caption: str,
        tags: List[str],
        mood: str,
        quality: float,
        embedding: List[float],
        project: str = None,
        clip_path: str = None,
        media_metadata: Dict[str, Any] = None
    ) -> None:
        """Add a frame with metadata to the vector store."""
        metadata = {
            "video": video_path,
            "timestamp": timestamp,
            "caption": caption,
            "tags": json.dumps(tags),
            "mood": mood,
            "quality": quality
        }
        
        # Add project info
        if project:
            metadata["project"] = project
        
        # Add clip path if available
        if clip_path:
            metadata["clip_path"] = clip_path
        
        # Add technical metadata
        if media_metadata:
            # Store technical metadata as JSON string
            metadata["media_metadata"] = json.dumps(media_metadata)
            # Also store key fields directly for filtering
            metadata["width"] = media_metadata.get("width", 0)
            metadata["height"] = media_metadata.get("height", 0)
            metadata["aspect_ratio"] = media_metadata.get("aspect_ratio", "")
            metadata["orientation"] = media_metadata.get("orientation", "")
            metadata["resolution_name"] = media_metadata.get("resolution_name", "")
            metadata["fps"] = media_metadata.get("fps", 0.0)
        
        self.frame_collection.add(
            ids=[frame_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[caption]  # For text search fallback
        )
    
    def add_asr_segment(
        self,
        segment_id: str,
        video_path: str,
        start_time: float,
        end_time: float,
        text: str,
        speaker: str,
        embedding: List[float],
        project: str = None
    ) -> None:
        """Add an ASR segment to the vector store."""
        metadata = {
            "video": video_path,
            "start": start_time,
            "end": end_time,
            "speaker": speaker
        }
        
        # Add project info
        if project:
            metadata["project"] = project
        
        self.asr_collection.add(
            ids=[segment_id],
            embeddings=[embedding],
            metadatas=[metadata],
            documents=[text]
        )
    
    def search_frames(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        quality_threshold: Optional[float] = None,
        tags_filter: Optional[List[str]] = None,
        project: Optional[str] = None,
        aspect_ratio: Optional[str] = None,
        orientation: Optional[str] = None,
        min_resolution: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar frames based on embedding similarity."""
        # Build where clause with AND operator for multiple conditions
        where_conditions = []
        if quality_threshold:
            where_conditions.append({"quality": {"$gte": quality_threshold}})
        if project:
            where_conditions.append({"project": {"$eq": project}})
        if aspect_ratio:
            where_conditions.append({"aspect_ratio": {"$eq": aspect_ratio}})
        if orientation:
            where_conditions.append({"orientation": {"$eq": orientation}})
        if min_resolution:
            where_conditions.append({"resolution_name": {"$eq": min_resolution}})
        
        # Construct the where clause
        where_clause = None
        if len(where_conditions) > 1:
            where_clause = {"$and": where_conditions}
        elif len(where_conditions) == 1:
            where_clause = where_conditions[0]
        
        results = self.frame_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause
        )
        
        # Post-process results to filter by tags if specified
        processed_results = []
        for i in range(len(results['ids'][0])):
            result = {
                "id": results['ids'][0][i],
                "distance": results['distances'][0][i],
                "metadata": results['metadatas'][0][i],
                "caption": results['documents'][0][i]
            }
            
            # Parse tags from metadata
            result["metadata"]["tags"] = json.loads(result["metadata"]["tags"])
            
            # Parse media metadata if present
            if "media_metadata" in result["metadata"]:
                result["metadata"]["media_metadata"] = json.loads(result["metadata"]["media_metadata"])
            
            # Filter by tags if specified
            if tags_filter:
                if any(tag in result["metadata"]["tags"] for tag in tags_filter):
                    processed_results.append(result)
            else:
                processed_results.append(result)
        
        return processed_results
    
    def search_asr(
        self,
        query_embedding: List[float],
        n_results: int = 10,
        speaker_filter: Optional[str] = None,
        project: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Search for similar ASR segments."""
        where_conditions = []
        if speaker_filter:
            where_conditions.append({"speaker": {"$eq": speaker_filter}})
        if project:
            where_conditions.append({"project": {"$eq": project}})
        
        where_clause = None
        if len(where_conditions) > 1:
            where_clause = {"$and": where_conditions}
        elif len(where_conditions) == 1:
            where_clause = where_conditions[0]
        
        results = self.asr_collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None
        )
        
        processed_results = []
        for i in range(len(results['ids'][0])):
            result = {
                "id": results['ids'][0][i],
                "distance": results['distances'][0][i],
                "metadata": results['metadatas'][0][i],
                "text": results['documents'][0][i]
            }
            processed_results.append(result)
        
        return processed_results
    
    def hybrid_search(
        self,
        text_query: str,
        image_embedding: Optional[List[float]] = None,
        text_embedding: Optional[List[float]] = None,
        n_results: int = 20
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Perform hybrid search across frames and ASR segments."""
        results = {
            "frames": [],
            "asr_segments": []
        }
        
        # Search frames if we have image embedding
        if image_embedding:
            results["frames"] = self.search_frames(image_embedding, n_results)
        
        # Search ASR if we have text embedding
        if text_embedding:
            results["asr_segments"] = self.search_asr(text_embedding, n_results)
        
        # Fallback to text search if no embeddings provided
        if not image_embedding and not text_embedding and text_query:
            # ChromaDB text search
            frame_results = self.frame_collection.query(
                query_texts=[text_query],
                n_results=n_results
            )
            
            asr_results = self.asr_collection.query(
                query_texts=[text_query],
                n_results=n_results
            )
            
            # Process text search results
            for i in range(len(frame_results['ids'][0])):
                result = {
                    "id": frame_results['ids'][0][i],
                    "distance": frame_results['distances'][0][i],
                    "metadata": frame_results['metadatas'][0][i],
                    "caption": frame_results['documents'][0][i]
                }
                result["metadata"]["tags"] = json.loads(result["metadata"]["tags"])
                results["frames"].append(result)
            
            for i in range(len(asr_results['ids'][0])):
                result = {
                    "id": asr_results['ids'][0][i],
                    "distance": asr_results['distances'][0][i],
                    "metadata": asr_results['metadatas'][0][i],
                    "text": asr_results['documents'][0][i]
                }
                results["asr_segments"].append(result)
        
        return results
    
    def get_video_segments(self, video_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Get all frames and ASR segments for a specific video."""
        # Get all frames for the video
        frame_results = self.frame_collection.get(
            where={"video": video_path}
        )
        
        frames = []
        if frame_results['ids']:
            for i in range(len(frame_results['ids'])):
                frame = {
                    "id": frame_results['ids'][i],
                    "metadata": frame_results['metadatas'][i],
                    "caption": frame_results['documents'][i] if frame_results['documents'] else ""
                }
                frame["metadata"]["tags"] = json.loads(frame["metadata"]["tags"])
                frames.append(frame)
        
        # Get all ASR segments for the video
        asr_results = self.asr_collection.get(
            where={"video": video_path}
        )
        
        asr_segments = []
        if asr_results['ids']:
            for i in range(len(asr_results['ids'])):
                segment = {
                    "id": asr_results['ids'][i],
                    "metadata": asr_results['metadatas'][i],
                    "text": asr_results['documents'][i] if asr_results['documents'] else ""
                }
                asr_segments.append(segment)
        
        return {
            "frames": frames,
            "asr_segments": asr_segments
        }
    
    def get_frames_by_project(self, project: str, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get all frames for a specific project."""
        # Get all frames for the project without using embeddings
        results = self.frame_collection.get(
            where={"project": {"$eq": project}},
            limit=limit
        )
        
        # Process results
        frames = []
        if results['ids']:
            for i in range(len(results['ids'])):
                frame = {
                    "id": results['ids'][i],
                    "caption": results['documents'][i] if results['documents'] else "",
                    "metadata": results['metadatas'][i] if results['metadatas'] else {}
                }
                # Parse tags if present
                if frame["metadata"].get("tags"):
                    try:
                        frame["metadata"]["tags"] = json.loads(frame["metadata"]["tags"])
                    except:
                        frame["metadata"]["tags"] = []
                frames.append(frame)
        
        return frames
    
    def delete_video_data(self, video_path: str) -> None:
        """Delete all data associated with a video."""
        # Get all IDs for the video
        frame_results = self.frame_collection.get(where={"video": video_path})
        asr_results = self.asr_collection.get(where={"video": video_path})
        
        # Delete frames
        if frame_results['ids']:
            self.frame_collection.delete(ids=frame_results['ids'])
        
        # Delete ASR segments
        if asr_results['ids']:
            self.asr_collection.delete(ids=asr_results['ids'])
    
    def reset(self) -> None:
        """Reset the entire vector store."""
        self.client.reset()
        self.__init__(persist_directory=self.client._settings.persist_directory)