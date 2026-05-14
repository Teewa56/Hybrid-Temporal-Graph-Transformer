FROM python:3.11-slim

# 1. Install system dependencies + Redis + Java (for Neo4j)
RUN apt-get update && apt-get install -y \
    gcc g++ curl git wget gnupg \
    redis-server \
    openjdk-17-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Neo4j Community Edition
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | apt-key add - && \
    echo 'deb https://debian.neo4j.com stable latest' > /etc/apt/sources.list.d/neo4j.list && \
    apt-get update && \
    apt-get install -y neo4j=1:5.19.0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# 3. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy app code
COPY . .
RUN mkdir -p logs checkpoints

# 5. Create the "Super-Startup" Script
# This starts Redis, then Neo4j, then FastAPI app.
RUN echo '#!/bin/bash \n\
service redis-server start \n\
neo4j-admin server passwd --set-password Hybrid_Temporal_Graph_Transformer \n\
neo4j start \n\
echo "Waiting for databases to wake up..." \n\
sleep 15 \n\
uvicorn app.main:app --host 0.0.0.0 --port 7860' > start.sh

RUN chmod +x start.sh

EXPOSE 7860
CMD ["./start.sh"]