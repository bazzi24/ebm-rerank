"""
Energy-Based Model for re-ranking RAGFlow chunks.
Uses a neural network to compute energy scores for query-chunk pairs.
Lower energy indicates better relevance.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer


class EnergyBasedReranker(nn.Module):
    """
    Energy-Based Model for re-ranking.
    
    The model computes an energy score for each query-chunk pair.
    Lower energy = higher relevance.
    """
    
    def __init__(self, embedding_dim: int = 768, hidden_dim: int = 512):
        super().__init__()
        
        # Energy function network
        self.energy_net = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Sentence transformer for encoding
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.embedding_dim = embedding_dim
        
    def encode(self, texts: List[str]) -> torch.Tensor:
        """Encode texts to embeddings."""
        with torch.no_grad():
            embeddings = self.encoder.encode(
                texts,
                convert_to_tensor=True,
                show_progress_bar=False
            )
        return embeddings
    
    def compute_energy(
        self, 
        query_embedding: torch.Tensor, 
        chunk_embedding: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute energy score for a query-chunk pair.
        
        Args:
            query_embedding: Query embedding tensor
            chunk_embedding: Chunk embedding tensor
            
        Returns:
            Energy score (lower = better)
        """
        # Concatenate query and chunk embeddings
        combined = torch.cat([query_embedding, chunk_embedding], dim=-1)
        
        # Compute energy through the network
        energy = self.energy_net(combined)
        
        return energy
    
    def forward(
        self, 
        query_embeddings: torch.Tensor, 
        chunk_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass: compute energy for multiple query-chunk pairs.
        
        Args:
            query_embeddings: Batch of query embeddings [batch_size, embedding_dim]
            chunk_embeddings: Batch of chunk embeddings [batch_size, embedding_dim]
            
        Returns:
            Energy scores [batch_size, 1]
        """
        combined = torch.cat([query_embeddings, chunk_embeddings], dim=-1)
        energies = self.energy_net(combined)
        return energies
    
    def rerank(
        self, 
        query: str, 
        chunks: List[Dict[str, any]]
    ) -> List[Tuple[Dict[str, any], float]]:
        """
        Re-rank chunks based on energy scores.
        
        Args:
            query: Query string
            chunks: List of chunk dictionaries from RAGFlow
            
        Returns:
            List of (chunk, energy_score) tuples, sorted by energy (ascending)
        """
        self.eval()
        
        with torch.no_grad():
            # Encode query
            query_embedding = self.encode([query])
            
            # Extract chunk texts and encode
            chunk_texts = [chunk.get('content', chunk.get('text', '')) for chunk in chunks]
            chunk_embeddings = self.encode(chunk_texts)
            
            # Expand query embedding to match chunk count
            query_embeddings = query_embedding.repeat(len(chunks), 1)
            
            # Compute energies
            energies = self.forward(query_embeddings, chunk_embeddings)
            energies = energies.squeeze().cpu().numpy()
            
            # Handle single chunk case
            if len(chunks) == 1:
                energies = np.array([energies])
            
            # Create ranked list
            ranked_results = list(zip(chunks, energies))
            ranked_results.sort(key=lambda x: x[1])  # Sort by energy (lower is better)
            
            return ranked_results
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights."""
        self.load_state_dict(torch.load(path, map_location='cpu'))


def create_pretrained_reranker(embedding_dim: int = 384) -> EnergyBasedReranker:
    """
    Create a pre-trained reranker with initialized weights.
    
    Note: In production, you would train this model on relevance data.
    For now, we use initialized weights that can be fine-tuned.
    """
    model = EnergyBasedReranker(embedding_dim=embedding_dim)
    model.eval()
    return model
