"""
RAGFlow client for fetching initial search results.
This module simulates or connects to RAGFlow to get the top 8 chunks.
"""

import httpx
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RAGFlowClient:
    """Client for interacting with RAGFlow API."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None, timeout: int = 30):
        """
        Initialize RAGFlow client.
        
        Args:
            base_url: RAGFlow API base URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.headers = {}
        
        if api_key:
            self.headers['Authorization'] = f'Bearer {api_key}'
    
    async def search(
        self, 
        query: str, 
        top_k: int = 8,
        dataset_id: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Search RAGFlow for relevant chunks.
        
        Args:
            query: Search query
            top_k: Number of chunks to retrieve
            dataset_id: Optional dataset/collection ID
            
        Returns:
            List of chunk dictionaries with content and metadata
        """
        endpoint = f"{self.base_url}/api/search"
        
        payload = {
            "query": query,
            "top_k": top_k
        }
        
        if dataset_id:
            payload["dataset_id"] = dataset_id
        
        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.post(
                    endpoint,
                    json=payload,
                    headers=self.headers
                )
                response.raise_for_status()
                
                data = response.json()
                chunks = data.get('chunks', data.get('results', []))
                
                logger.info(f"Retrieved {len(chunks)} chunks from RAGFlow for query: {query}")
                return chunks
                
        except httpx.HTTPError as e:
            logger.error(f"RAGFlow API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error when calling RAGFlow: {e}")
            raise


class MockRAGFlowClient:
    """Mock RAGFlow client for testing and development."""
    
    def __init__(self, *args, **kwargs):
        """Initialize mock client."""
        logger.info("Using MockRAGFlowClient for development")
    
    async def search(
        self, 
        query: str, 
        top_k: int = 8,
        dataset_id: Optional[str] = None
    ) -> List[Dict[str, any]]:
        """
        Return mock chunks for testing.
        
        In production, replace this with actual RAGFlow integration.
        """
        mock_chunks = [
            {
                "id": "chunk_1",
                "content": f"Energy-based models are a class of models that learn by associating low energy values with correct or desired configurations. They are related to the query: {query}",
                "score": 0.92,
                "metadata": {"source": "doc1.pdf", "page": 1}
            },
            {
                "id": "chunk_2",
                "content": "Neural networks can be trained to minimize energy functions, which helps in learning representations that capture the underlying data distribution.",
                "score": 0.88,
                "metadata": {"source": "doc2.pdf", "page": 3}
            },
            {
                "id": "chunk_3",
                "content": "Re-ranking algorithms improve search results by reordering initial retrievals based on more sophisticated scoring mechanisms than simple vector similarity.",
                "score": 0.85,
                "metadata": {"source": "doc3.pdf", "page": 5}
            },
            {
                "id": "chunk_4",
                "content": "The concept of energy minimization dates back to physics, where systems naturally evolve toward states of minimum energy.",
                "score": 0.82,
                "metadata": {"source": "doc1.pdf", "page": 7}
            },
            {
                "id": "chunk_5",
                "content": "Transformer architectures have revolutionized natural language processing by enabling better context understanding through self-attention mechanisms.",
                "score": 0.80,
                "metadata": {"source": "doc4.pdf", "page": 2}
            },
            {
                "id": "chunk_6",
                "content": "Vector databases enable efficient similarity search by organizing embeddings in structures like HNSW or FAISS indexes.",
                "score": 0.78,
                "metadata": {"source": "doc5.pdf", "page": 4}
            },
            {
                "id": "chunk_7",
                "content": "Retrieval-augmented generation combines the power of large language models with external knowledge retrieval for more accurate responses.",
                "score": 0.75,
                "metadata": {"source": "doc6.pdf", "page": 1}
            },
            {
                "id": "chunk_8",
                "content": "Machine learning models benefit from contrastive learning, where they learn to distinguish between similar and dissimilar examples.",
                "score": 0.72,
                "metadata": {"source": "doc7.pdf", "page": 6}
            }
        ]
        
        logger.info(f"Mock RAGFlow returned {len(mock_chunks)} chunks for query: {query}")
        return mock_chunks[:top_k]
