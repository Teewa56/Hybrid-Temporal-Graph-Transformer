FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for PyTorch Geometric and Neo4j
RUN apt-get update && apt-get install -y \
    gcc g++ curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --no-build-isolation -r requirements.txt

# Copy application code
COPY . .

# Create log and checkpoint directories
RUN mkdir -p logs checkpoints

EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]