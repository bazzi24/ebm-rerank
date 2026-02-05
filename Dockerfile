FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install uv
RUN pip install uv

# Copy project files
COPY pyproject.toml .
COPY app/ ./app/
COPY models/ ./models/
COPY utils/ ./utils/
COPY .env.example .env

# Install dependencies
RUN uv sync --no-dev

# Expose port
EXPOSE 8000

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV USE_MOCK_RAGFLOW=true

# Run the application
CMD ["uv", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
