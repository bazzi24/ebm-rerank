"""Configuration management using pydantic-settings."""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    api_title: str = "EBM Re-ranking API"
    api_version: str = "0.1.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # RAGFlow Settings
    ragflow_base_url: str = Field(
        default="http://localhost:9380",
        description="RAGFlow API base URL"
    )
    ragflow_api_key: Optional[str] = Field(
        default=None,
        description="RAGFlow API key"
    )
    use_mock_ragflow: bool = Field(
        default=True,
        description="Use mock RAGFlow client for development"
    )
    
    # Model Settings
    ebm_embedding_dim: int = Field(
        default=384,
        description="Embedding dimension for EBM (matches all-MiniLM-L6-v2)"
    )
    ebm_hidden_dim: int = Field(
        default=512,
        description="Hidden dimension for EBM network"
    )
    model_path: Optional[str] = Field(
        default=None,
        description="Path to pre-trained model weights"
    )
    
    # Search Settings
    top_k_initial: int = Field(
        default=8,
        description="Number of chunks to retrieve from RAGFlow"
    )
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


# Global settings instance
settings = Settings()
