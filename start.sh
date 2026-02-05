#!/bin/bash

# EBM Re-ranking API Startup Script

set -e

echo "🚀 Starting EBM Re-ranking API..."

# Check if .env exists, if not copy from example
if [ ! -f .env ]; then
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo "⚠️  Please configure .env with your settings before production use"
fi

# Install dependencies with uv
echo "📦 Installing dependencies with uv..."
uv sync

# Start the API
echo "✨ Starting FastAPI server..."
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
