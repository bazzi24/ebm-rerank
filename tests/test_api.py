"""Tests for the EBM Re-ranking API."""

import pytest
from httpx import AsyncClient
from app.main import app


@pytest.mark.asyncio
async def test_root_endpoint():
    """Test the root endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data


@pytest.mark.asyncio
async def test_health_check():
    """Test the health check endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True


@pytest.mark.asyncio
async def test_search_endpoint():
    """Test the search and re-rank endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        request_data = {
            "query": "What are energy-based models?",
            "top_k": 5
        }
        response = await client.post("/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        assert data["query"] == request_data["query"]
        assert len(data["results"]) <= request_data["top_k"]
        assert data["total_results"] >= 0
        
        # Check result structure
        if data["results"]:
            result = data["results"][0]
            assert "id" in result
            assert "content" in result
            assert "original_score" in result
            assert "energy_score" in result
            assert "final_rank" in result
            
            # Check ranking order
            for i in range(len(data["results"]) - 1):
                assert data["results"][i]["final_rank"] < data["results"][i + 1]["final_rank"]


@pytest.mark.asyncio
async def test_search_with_empty_query():
    """Test search with empty query."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        request_data = {
            "query": "",
            "top_k": 5
        }
        response = await client.post("/search", json=request_data)
        # Should fail validation
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_model_info_endpoint():
    """Test the model info endpoint."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        response = await client.get("/model/info")
        assert response.status_code == 200
        
        data = response.json()
        assert "embedding_dim" in data
        assert "encoder_model" in data
        assert data["status"] == "loaded"


@pytest.mark.asyncio
async def test_search_top_k_limits():
    """Test that top_k is properly limited."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        # Test upper limit
        request_data = {
            "query": "test query",
            "top_k": 25  # Above max of 20
        }
        response = await client.post("/search", json=request_data)
        assert response.status_code == 422
        
        # Test lower limit
        request_data = {
            "query": "test query",
            "top_k": 0
        }
        response = await client.post("/search", json=request_data)
        assert response.status_code == 422


@pytest.mark.asyncio
async def test_energy_scores_ordering():
    """Test that results are ordered by energy score (ascending)."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        request_data = {
            "query": "machine learning models",
            "top_k": 8
        }
        response = await client.post("/search", json=request_data)
        assert response.status_code == 200
        
        data = response.json()
        
        # Energy scores should be in ascending order (lower is better)
        energy_scores = [r["energy_score"] for r in data["results"]]
        assert energy_scores == sorted(energy_scores)
