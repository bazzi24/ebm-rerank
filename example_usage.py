"""
Example usage of the EBM Re-ranking API.
Demonstrates how to integrate the API into your application.
"""

import asyncio
import httpx
from typing import List, Dict


class EBMRerankClient:
    """Client for the EBM Re-ranking API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    async def search(
        self, 
        query: str, 
        top_k: int = 5,
        dataset_id: str = None
    ) -> Dict:
        """
        Search and re-rank using the EBM API.
        
        Args:
            query: Search query
            top_k: Number of results to return
            dataset_id: Optional dataset ID
            
        Returns:
            Dict with re-ranked results
        """
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/search",
                json={
                    "query": query,
                    "top_k": top_k,
                    "dataset_id": dataset_id
                },
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()
    
    async def health_check(self) -> Dict:
        """Check API health."""
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{self.base_url}/health")
            response.raise_for_status()
            return response.json()


async def main():
    """Example usage."""
    
    # Initialize client
    client = EBMRerankClient(base_url="http://localhost:8000")
    
    # Check health
    print("🏥 Checking API health...")
    health = await client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}\n")
    
    # Example queries
    queries = [
        "What are energy-based models in machine learning?",
        "How does neural network re-ranking work?",
        "Explain retrieval-augmented generation"
    ]
    
    for query in queries:
        print(f"🔍 Query: {query}")
        print("-" * 80)
        
        # Search and re-rank
        results = await client.search(query=query, top_k=3)
        
        print(f"📊 Retrieved {results['total_results']} results:")
        print(f"   Initial retrieval: {results['processing_info']['initial_retrieval_count']} chunks")
        print(f"   Method: {results['processing_info']['reranking_method']}\n")
        
        # Display top results
        for result in results['results']:
            print(f"   Rank {result['final_rank']}:")
            print(f"   • ID: {result['id']}")
            print(f"   • Content: {result['content'][:100]}...")
            print(f"   • Original Score: {result['original_score']:.3f}")
            print(f"   • Energy Score: {result['energy_score']:.3f}")
            if result.get('metadata'):
                print(f"   • Source: {result['metadata'].get('source', 'N/A')}")
            print()
        
        print("=" * 80)
        print()


if __name__ == "__main__":
    print("🚀 EBM Re-ranking API - Example Usage\n")
    asyncio.run(main())
