# Quick Start Guide - EBM Re-ranking API

Get the API running in 5 minutes!

## Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

Install uv if you haven't:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Step 1: Install Dependencies

```bash
cd ebm-rerank-api
uv sync
```

This will:
- Create a virtual environment
- Install all dependencies (FastAPI, PyTorch, etc.)
- Set up the project

## Step 2: Configure

```bash
cp .env.example .env
```

The default configuration uses a mock RAGFlow client, perfect for testing!

## Step 3: Start the API

### Option A: Using the startup script
```bash
./start.sh
```

### Option B: Manual start
```bash
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Option C: Using Docker
```bash
docker-compose up -d
```

## Step 4: Test the API

### Check health
```bash
curl http://localhost:8000/health
```

### Make a search request
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are energy-based models?",
    "top_k": 5
  }'
```

### Use the example script
```bash
uv run python example_usage.py
```

## Step 5: View API Documentation

Open your browser to:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Expected Output

You should see re-ranked results like:

```json
{
  "query": "What are energy-based models?",
  "results": [
    {
      "id": "chunk_1",
      "content": "Energy-based models are...",
      "original_score": 0.92,
      "energy_score": 0.15,
      "final_rank": 1,
      "metadata": {"source": "doc1.pdf", "page": 1}
    }
  ],
  "total_results": 5,
  "processing_info": {
    "initial_retrieval_count": 8,
    "reranking_method": "EBM"
  }
}
```

## Next Steps

### Connect to Real RAGFlow

1. Edit `.env`:
```env
USE_MOCK_RAGFLOW=false
RAGFLOW_BASE_URL=http://your-ragflow-instance:9380
RAGFLOW_API_KEY=your-api-key
```

2. Restart the API

### Train Your Own Model

```bash
uv run python train_ebm.py
```

Then configure the trained model:
```env
MODEL_PATH=models/ebm_best.pth
```

### Run Tests

```bash
uv run pytest tests/ -v
```

### Deploy to Production

See [ARCHITECTURE.md](ARCHITECTURE.md) for deployment strategies and [README.md](README.md) for detailed documentation.

## Troubleshooting

### Port already in use
```bash
# Change the port
uv run uvicorn app.main:app --port 8001
```

### Dependencies fail to install
```bash
# Update uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clean and reinstall
rm -rf .venv
uv sync
```

### Model fails to load
- Ensure you have enough RAM (≥4GB recommended)
- PyTorch will download sentence-transformers model on first run (~100MB)

### RAGFlow connection fails
- Verify RAGFlow is running: `curl http://localhost:9380/health`
- Check network settings
- Use mock mode for testing: `USE_MOCK_RAGFLOW=true`

## Support

- 📖 Full docs: [README.md](README.md)
- 🏗️ Architecture: [ARCHITECTURE.md](ARCHITECTURE.md)
- 💡 Examples: [example_usage.py](example_usage.py)
- 🧪 Tests: [tests/test_api.py](tests/test_api.py)

Happy re-ranking! 🚀
