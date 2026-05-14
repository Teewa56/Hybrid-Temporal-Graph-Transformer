FROM python:3.11-slim

# 1. Install system dependencies + Redis + Java 21 (for Neo4j)
RUN apt-get update && apt-get install -y \
    gcc g++ curl git wget gnupg \
    redis-server \
    openjdk-21-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# 2. Install Neo4j Community Edition (Modern Debian Method)
RUN wget -O - https://debian.neo4j.com/neotechnology.gpg.key | gpg --dearmor > /usr/share/keyrings/neo4j.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/neo4j.gpg] https://debian.neo4j.com stable latest" > /etc/apt/sources.list.d/neo4j.list && \
    apt-get update && \
    apt-get install -y neo4j=1:5.19.0 && \
    rm -rf /var/lib/apt/lists/*

# 3. Memory Optimization for Neo4j (Crucial for Hugging Face)
RUN sed -i 's/#server.memory.heap.initial_size=512m/server.memory.heap.initial_size=512m/' /etc/neo4j/neo4j.conf && \
    sed -i 's/#server.memory.heap.max_size=1G/server.memory.heap.max_size=1G/' /etc/neo4j/neo4j.conf

WORKDIR /app
COPY requirements.txt .

# 4. Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy app code and setup environment
COPY . .
RUN mkdir -p logs

# 6. The "Single Container" Startup Script
RUN echo '#!/bin/bash \n\
service redis-server start \n\
neo4j-admin server set-initial-password Hybrid_Temporal_Graph_Transformer || true \n\
neo4j start \n\
echo "Waiting for databases to wake up..." \n\
sleep 20 \n\
uvicorn app.main:app --host 0.0.0.0 --port 7860' > start.sh

RUN chmod +x start.sh

EXPOSE 7860
CMD ["./start.sh"]