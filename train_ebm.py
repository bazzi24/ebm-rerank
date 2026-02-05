"""
Example training script for the Energy-Based Model.

This demonstrates how to train the EBM on relevance data.
In practice, you would need a dataset of (query, relevant_chunk, irrelevant_chunk) triplets.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple
import logging

from models import EnergyBasedReranker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TripletDataset(Dataset):
    """Dataset of (query, positive_chunk, negative_chunk) triplets."""
    
    def __init__(self, triplets: List[Tuple[str, str, str]]):
        """
        Args:
            triplets: List of (query, positive, negative) text triplets
        """
        self.triplets = triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        return self.triplets[idx]


def create_sample_data() -> List[Tuple[str, str, str]]:
    """
    Create sample training data.
    
    In production, this would come from your relevance judgments:
    - Query logs with click data
    - Human annotations
    - RAGFlow usage analytics
    """
    return [
        (
            "What are energy-based models?",
            "Energy-based models are a class of models that learn by associating low energy with correct configurations.",
            "The weather today is sunny with a high of 75 degrees."
        ),
        (
            "How does neural re-ranking work?",
            "Neural re-ranking uses deep learning to reorder search results based on learned relevance patterns.",
            "Chocolate cake recipes typically include flour, sugar, and cocoa powder."
        ),
        (
            "Explain RAG in AI",
            "Retrieval-Augmented Generation combines language models with external knowledge retrieval for more accurate responses.",
            "The capital of France is Paris, known for the Eiffel Tower."
        ),
        # Add more triplets...
    ]


def contrastive_loss(energy_pos: torch.Tensor, energy_neg: torch.Tensor, margin: float = 1.0):
    """
    Contrastive loss: positive examples should have lower energy than negatives.
    
    Args:
        energy_pos: Energy scores for positive pairs
        energy_neg: Energy scores for negative pairs
        margin: Margin for separation
        
    Returns:
        Loss value
    """
    # We want: energy_pos < energy_neg - margin
    # Loss = max(0, energy_pos - energy_neg + margin)
    loss = torch.relu(energy_pos - energy_neg + margin)
    return loss.mean()


def train_epoch(
    model: EnergyBasedReranker,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    device: str = 'cpu'
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch_idx, (queries, positives, negatives) in enumerate(dataloader):
        optimizer.zero_grad()
        
        # Encode texts
        query_embeddings = model.encode(list(queries))
        pos_embeddings = model.encode(list(positives))
        neg_embeddings = model.encode(list(negatives))
        
        # Compute energies
        energy_pos = model.forward(query_embeddings, pos_embeddings)
        energy_neg = model.forward(query_embeddings, neg_embeddings)
        
        # Compute loss
        loss = contrastive_loss(energy_pos, energy_neg, margin=1.0)
        
        # Backprop
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Batch {batch_idx + 1}: loss = {loss.item():.4f}")
    
    return total_loss / len(dataloader)


def evaluate(
    model: EnergyBasedReranker,
    dataloader: DataLoader,
    device: str = 'cpu'
) -> float:
    """Evaluate the model."""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for queries, positives, negatives in dataloader:
            # Encode
            query_embeddings = model.encode(list(queries))
            pos_embeddings = model.encode(list(positives))
            neg_embeddings = model.encode(list(negatives))
            
            # Compute energies
            energy_pos = model.forward(query_embeddings, pos_embeddings)
            energy_neg = model.forward(query_embeddings, neg_embeddings)
            
            # Check if positive has lower energy
            correct += (energy_pos < energy_neg).sum().item()
            total += len(queries)
    
    accuracy = correct / total if total > 0 else 0
    return accuracy


def main():
    """Main training function."""
    
    # Configuration
    config = {
        'embedding_dim': 384,
        'hidden_dim': 512,
        'batch_size': 8,
        'num_epochs': 10,
        'learning_rate': 1e-4,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    logger.info(f"Training configuration: {config}")
    
    # Create model
    model = EnergyBasedReranker(
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim']
    )
    model = model.to(config['device'])
    
    # Create dataset
    logger.info("Creating dataset...")
    triplets = create_sample_data()
    
    # Split into train/val (80/20)
    split_idx = int(0.8 * len(triplets))
    train_triplets = triplets[:split_idx]
    val_triplets = triplets[split_idx:]
    
    train_dataset = TripletDataset(train_triplets)
    val_dataset = TripletDataset(val_triplets)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )
    
    logger.info(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(config['num_epochs']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config['num_epochs']}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, config['device'])
        logger.info(f"Train loss: {train_loss:.4f}")
        
        # Evaluate
        val_acc = evaluate(model, val_loader, config['device'])
        logger.info(f"Validation accuracy: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_model('models/ebm_best.pth')
            logger.info(f"✓ Saved new best model (acc: {val_acc:.4f})")
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training complete! Best validation accuracy: {best_val_acc:.4f}")
    logger.info(f"Model saved to: models/ebm_best.pth")
    logger.info(f"{'='*60}")


if __name__ == "__main__":
    main()
