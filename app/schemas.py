"""Pydantic schemas for request/response validation."""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class SearchRequest(BaseModel):
    """Request schema for search endpoint."""
    
    query: str = Field(
        ...,
        description="Search query text",
        min_length=1,
        max_length=1000
    )
    top_k: int = Field(
        default=8,
        description="Number of results to return after re-ranking",
        ge=1,
        le=20
    )
    dataset_id: Optional[str] = Field(
        default=None,
        description="Optional dataset/collection ID for RAGFlow"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are energy-based models?",
                "top_k": 5,
                "dataset_id": "my-dataset"
            }
        }


class ChunkMetadata(BaseModel):
    """Metadata for a document chunk."""
    
    source: Optional[str] = None
    page: Optional[int] = None
    additional_info: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = "allow"


class RankedChunk(BaseModel):
    """A single ranked chunk result."""
    
    id: str = Field(..., description="Chunk identifier")
    content: str = Field(..., description="Chunk text content")
    original_score: float = Field(
        ..., 
        description="Original similarity score from RAGFlow"
    )
    energy_score: float = Field(
        ..., 
        description="Energy score from EBM (lower is better)"
    )
    final_rank: int = Field(
        ..., 
        description="Final rank after EBM re-ranking (1-based)"
    )
    metadata: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Additional metadata"
    )


class SearchResponse(BaseModel):
    """Response schema for search endpoint."""
    
    query: str = Field(..., description="Original search query")
    results: List[RankedChunk] = Field(
        ..., 
        description="Re-ranked chunks"
    )
    total_results: int = Field(
        ..., 
        description="Total number of results returned"
    )
    processing_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional processing information"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are energy-based models?",
                "results": [
                    {
                        "id": "chunk_1",
                        "content": "Energy-based models learn by associating...",
                        "original_score": 0.92,
                        "energy_score": 0.15,
                        "final_rank": 1,
                        "metadata": {"source": "doc1.pdf", "page": 1}
                    }
                ],
                "total_results": 1,
                "processing_info": {
                    "initial_retrieval_count": 8,
                    "reranking_method": "EBM"
                }
            }
        }


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    model_loaded: bool = Field(..., description="Whether EBM model is loaded")
