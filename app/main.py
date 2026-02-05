"""Main FastAPI application for EBM re-ranking."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from .schemas import SearchRequest, SearchResponse, HealthResponse, RankedChunk
from models import create_pretrained_reranker, EnergyBasedReranker
from utils import settings, RAGFlowClient, MockRAGFlowClient

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global model instance
ebm_model: EnergyBasedReranker = None
ragflow_client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    global ebm_model, ragflow_client
    
    logger.info("Starting EBM Re-ranking API...")
    
    # Initialize EBM model
    logger.info(f"Loading EBM model with embedding_dim={settings.ebm_embedding_dim}")
    ebm_model = create_pretrained_reranker(embedding_dim=settings.ebm_embedding_dim)
    
    if settings.model_path:
        try:
            ebm_model.load_model(settings.model_path)
            logger.info(f"Loaded model weights from {settings.model_path}")
        except Exception as e:
            logger.warning(f"Could not load model weights: {e}. Using initialized weights.")
    
    # Initialize RAGFlow client
    if settings.use_mock_ragflow:
        logger.info("Using Mock RAGFlow client")
        ragflow_client = MockRAGFlowClient()
    else:
        logger.info(f"Connecting to RAGFlow at {settings.ragflow_base_url}")
        ragflow_client = RAGFlowClient(
            base_url=settings.ragflow_base_url,
            api_key=settings.ragflow_api_key
        )
    
    logger.info("API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down EBM Re-ranking API...")


# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="Energy-Based Model Re-ranking API for RAGFlow results",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {
        "message": "EBM Re-ranking API",
        "version": settings.api_version,
        "docs": "/docs"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.api_version,
        model_loaded=ebm_model is not None
    )


@app.post("/search", response_model=SearchResponse)
async def search_and_rerank(request: SearchRequest):
    """
    Search RAGFlow and re-rank results using EBM.
    
    This endpoint:
    1. Retrieves top-k chunks from RAGFlow based on vector similarity
    2. Re-ranks them using the Energy-Based Model
    3. Returns the re-ranked results
    
    Args:
        request: SearchRequest with query and parameters
        
    Returns:
        SearchResponse with re-ranked chunks
    """
    try:
        logger.info(f"Processing search request: query='{request.query}', top_k={request.top_k}")
        
        # Step 1: Get initial results from RAGFlow
        logger.info(f"Fetching {settings.top_k_initial} chunks from RAGFlow")
        chunks = await ragflow_client.search(
            query=request.query,
            top_k=settings.top_k_initial,
            dataset_id=request.dataset_id
        )
        
        if not chunks:
            logger.warning(f"No chunks returned from RAGFlow for query: {request.query}")
            return SearchResponse(
                query=request.query,
                results=[],
                total_results=0,
                processing_info={
                    "initial_retrieval_count": 0,
                    "reranking_method": "EBM",
                    "message": "No results found"
                }
            )
        
        # Step 2: Re-rank using EBM
        logger.info(f"Re-ranking {len(chunks)} chunks using EBM")
        ranked_results = ebm_model.rerank(query=request.query, chunks=chunks)
        
        # Step 3: Format results
        formatted_results = []
        for rank, (chunk, energy_score) in enumerate(ranked_results[:request.top_k], start=1):
            formatted_results.append(
                RankedChunk(
                    id=chunk.get('id', f'chunk_{rank}'),
                    content=chunk.get('content', chunk.get('text', '')),
                    original_score=chunk.get('score', 0.0),
                    energy_score=float(energy_score),
                    final_rank=rank,
                    metadata=chunk.get('metadata', {})
                )
            )
        
        logger.info(f"Returning {len(formatted_results)} re-ranked results")
        
        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results),
            processing_info={
                "initial_retrieval_count": len(chunks),
                "reranking_method": "EBM",
                "top_k_requested": request.top_k
            }
        )
        
    except Exception as e:
        logger.error(f"Error processing search request: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing search request: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded EBM model."""
    if ebm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "embedding_dim": ebm_model.embedding_dim,
        "encoder_model": "all-MiniLM-L6-v2",
        "architecture": "Energy-Based Neural Network",
        "status": "loaded"
    }
