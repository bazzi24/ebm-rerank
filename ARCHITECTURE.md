# EBM Re-ranking API - Architecture Documentation

## System Overview

The EBM Re-ranking API is a microservice that enhances RAGFlow search results using Energy-Based Models (EBMs). It acts as an intelligent re-ranking layer that improves result relevance beyond simple vector similarity.

## High-Level Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP POST /search
       │ {query, top_k}
       ▼
┌─────────────────────────────────────────┐
│      FastAPI Application Layer          │
│  ┌────────────────────────────────┐     │
│  │  Endpoint: /search             │     │
│  │  - Validates request           │     │
│  │  - Orchestrates workflow       │     │
│  └────────────────────────────────┘     │
└──────────┬──────────────┬───────────────┘
           │              │
           ▼              ▼
    ┌──────────┐   ┌─────────────────┐
    │ RAGFlow  │   │  EBM Re-ranker  │
    │  Client  │   │   (PyTorch)     │
    └──────────┘   └─────────────────┘
           │              │
           │ 8 chunks     │ Energy scores
           ▼              ▼
    ┌───────────────────────────────┐
    │   Response Formatter          │
    │   - Combines scores           │
    │   - Orders by energy          │
    │   - Returns top_k results     │
    └───────────────────────────────┘
```

## Component Details

### 1. API Layer (`app/`)

**main.py**: Core FastAPI application
- Lifespan management for model initialization
- Request/response handling
- Error handling and logging
- CORS configuration

**schemas.py**: Pydantic models
- `SearchRequest`: Input validation
- `SearchResponse`: Structured output
- `RankedChunk`: Individual result format
- `HealthResponse`: Health check format

### 2. Model Layer (`models/`)

**ebm_reranker.py**: Energy-Based Model implementation

```python
class EnergyBasedReranker(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=512):
        # Energy network architecture:
        # Input: [query_emb | chunk_emb] (768*2 = 1536 dims)
        # Hidden: 512 → 256 → 1
        # Output: Energy score (scalar)
```

**Architecture Details**:
- **Encoder**: SentenceTransformers (all-MiniLM-L6-v2)
  - Produces 384-dim embeddings
  - Fast CPU inference (~50ms per batch)
  - Pre-trained on semantic similarity

- **Energy Network**:
  ```
  Layer 1: Linear(768, 512) + LayerNorm + ReLU + Dropout(0.1)
  Layer 2: Linear(512, 256) + LayerNorm + ReLU + Dropout(0.1)
  Layer 3: Linear(256, 1)
  ```

- **Energy Function**:
  - E(query, chunk) → ℝ
  - Lower energy = higher relevance
  - Trained via contrastive learning

### 3. Utility Layer (`utils/`)

**ragflow_client.py**: RAGFlow integration
- `RAGFlowClient`: Production client for real RAGFlow API
- `MockRAGFlowClient`: Development/testing mock
- Async HTTP requests via httpx

**config.py**: Configuration management
- Environment variable loading via pydantic-settings
- Type-safe settings with defaults
- `.env` file support

## Data Flow

### Request Processing Pipeline

```
1. Request Reception
   ├─ POST /search
   ├─ Validate: query (1-1000 chars), top_k (1-20)
   └─ Log request

2. Initial Retrieval (RAGFlow)
   ├─ Query RAGFlow API
   ├─ Retrieve top-8 chunks by vector similarity
   ├─ Chunks include: {id, content, score, metadata}
   └─ Handle empty results

3. EBM Re-ranking
   ├─ Encode query → embedding_q
   ├─ Encode chunks → [embedding_c1, ..., embedding_c8]
   ├─ For each chunk:
   │   ├─ Concatenate [embedding_q | embedding_ci]
   │   ├─ Pass through energy network
   │   └─ Get energy score E_i
   ├─ Sort by energy (ascending)
   └─ Return top_k results

4. Response Formatting
   ├─ Create RankedChunk objects
   ├─ Add metadata (original_score, energy_score, rank)
   ├─ Include processing_info
   └─ Send JSON response
```

### Timing Breakdown (typical)

```
Total latency: ~150-250ms

RAGFlow retrieval:   50-100ms  (network + vector search)
Encoding:            30-50ms   (query + 8 chunks)
Energy computation:  10-20ms   (8 forward passes)
Formatting:          5-10ms    (JSON serialization)
Network overhead:    20-40ms   (HTTP)
```

## Energy-Based Model Theory

### Why Energy-Based Models?

Traditional vector similarity (cosine/dot product) assumes:
- Linear relationship between embeddings
- Same importance for all dimensions
- No learned interaction between query and document

EBMs learn to:
- Model complex query-document interactions
- Weight different semantic aspects differently
- Capture non-linear relevance patterns

### Training Objective

Given triplets (query, positive, negative):

```
Loss = max(0, E(q, pos) - E(q, neg) + margin)

where:
  E(q, d) = energy_network([embed(q) | embed(d)])
  margin = separation threshold (typically 1.0)
```

Goal: Minimize energy for relevant pairs, maximize for irrelevant pairs.

### Inference

```python
# Compute energy for each chunk
energies = []
for chunk in chunks:
    energy = model.compute_energy(query_emb, chunk_emb)
    energies.append(energy)

# Sort ascending (lower energy = better)
ranked_indices = argsort(energies)
return [chunks[i] for i in ranked_indices]
```

## Deployment Architectures

### Option 1: Standalone Service

```
Client → EBM API → RAGFlow
```

**Pros**:
- Simple deployment
- Easy to scale API independently
- Clear separation of concerns

**Cons**:
- Additional network hop
- Latency from RAGFlow communication

### Option 2: Sidecar Pattern

```
Client → RAGFlow (with EBM sidecar)
```

**Pros**:
- Reduced latency (local communication)
- Tighter integration
- Single endpoint for clients

**Cons**:
- Coupled deployment
- RAGFlow must be modified

### Option 3: Batch Re-ranking

```
Periodic job: RAGFlow results → EBM → Cache
Client → Cached results
```

**Pros**:
- Zero query-time latency
- Can use heavier models
- Pre-computed rankings

**Cons**:
- Stale results
- Higher storage requirements
- Not suitable for dynamic queries

## Scalability Considerations

### Horizontal Scaling

```
Load Balancer
    ├─ EBM API Instance 1
    ├─ EBM API Instance 2
    └─ EBM API Instance 3
```

Each instance:
- Loads model independently
- Stateless (no shared state)
- Can handle ~10-50 QPS per instance (CPU)

### Optimization Strategies

1. **Model Optimization**
   - Quantization (INT8): 4x smaller, 2-3x faster
   - ONNX export: ~20% faster inference
   - Batch processing: Process multiple queries together

2. **Caching**
   - Cache embeddings for frequent chunks
   - Cache RAGFlow results (TTL: 5-60 min)
   - Use Redis for distributed cache

3. **GPU Acceleration**
   - 10-50x faster for batch inference
   - Use CUDA-enabled PyTorch
   - GPU queue for request batching

### Monitoring Metrics

```python
# Key metrics to track
- Requests per second (RPS)
- P50, P95, P99 latency
- Model inference time
- RAGFlow API latency
- Error rate
- Cache hit rate (if implemented)
- Energy score distribution
```

## Security Considerations

1. **API Security**
   - Rate limiting (e.g., 100 req/min per IP)
   - API key authentication (optional)
   - Request size limits (max query length)

2. **Input Validation**
   - Pydantic schema validation
   - SQL injection prevention (N/A for this API)
   - XSS prevention in responses

3. **RAGFlow Integration**
   - Secure API key storage (environment variables)
   - TLS/SSL for production
   - Timeout configuration (prevent hanging)

## Error Handling

```python
Error Types:
├─ 400 Bad Request
│   └─ Invalid query format, top_k out of range
├─ 422 Validation Error
│   └─ Pydantic validation failure
├─ 500 Internal Server Error
│   ├─ Model inference failure
│   ├─ RAGFlow connection error
│   └─ Unexpected exceptions
└─ 503 Service Unavailable
    └─ Model not loaded
```

## Future Enhancements

1. **Multi-Model Support**
   - Load multiple EBM architectures
   - A/B testing framework
   - Ensemble re-ranking

2. **Advanced Training**
   - Online learning from click data
   - Personalized ranking models
   - Domain-specific fine-tuning

3. **Performance**
   - GPU support
   - Model distillation
   - Approximate nearest neighbor search

4. **Features**
   - Explain rankings (attention visualization)
   - Diversity re-ranking
   - Multi-stage retrieval

5. **Observability**
   - Prometheus metrics export
   - Distributed tracing (OpenTelemetry)
   - Grafana dashboards
