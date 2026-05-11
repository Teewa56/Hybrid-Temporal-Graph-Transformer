## Getting the Project Running

### Prerequisites
Before anything else, make sure you have these installed:

```bash
python --version        # 3.10+
docker --version        # any recent version
pip install -r requirements.txt
cp .env.example .env    # fill in your Squad API keys
```

---

### Step 1 — Start Infrastructure

Redis and Neo4j must be running before the API starts. Open a terminal and run both:

```bash
# Redis — in-memory transaction cache
docker run -d \
  --name trustguard_redis \
  -p 6379:6379 \
  redis:7.2-alpine \
  redis-server --maxmemory 512mb --maxmemory-policy allkeys-lru

# Neo4j — live transaction graph
docker run -d \
  --name trustguard_neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/trustguard \
  neo4j:5.19-community

# Verify both are up
docker ps
```

Wait about 15 seconds for Neo4j to fully initialize, then confirm:

```bash
docker logs trustguard_redis   # should end with: Ready to accept connections
docker logs trustguard_neo4j   # should end with: Started
```

---

### Step 2 — Generate Synthetic Training Data

This runs all five data generators and exports model-ready datasets to `synthetic_data_generator/outputs/`:

```bash
python -m synthetic_data_generator.pipeline.run_all
```

Expected output:
```
============================================================
  1/5  Behavioral Data (Transformer)
============================================================
 180,000 transactions | Fraud rate: 8.12% | 14.2s

============================================================
  2/5  Graph Data (GraphSAGE)
...
```

When complete, verify outputs exist:

```bash
ls synthetic_data_generator/outputs/
# behavioral/  graph/  payload/  sim_swap/  kyc/  data_manifest.json
```

---

### Step 3 — Train All 5 Models

Launch Jupyter and run `model_training.ipynb` **top to bottom**. Every cell must complete before moving to the next:

```bash
jupyter notebook notebooks/model_training.ipynb
```

The notebook trains in this order:

| Order | Model | Saves To |
|---|---|---|
| 1 | Behavioral Transformer | `checkpoints/transformer.pt` |
| 2 | GAN + Autoencoder (KYC) | `checkpoints/gan_autoencoder.pt` |
| 3 | CNN-GNN Hybrid | `checkpoints/cnn_gnn.pt` |
| 4 | TSSGC (SIM Swap) | `checkpoints/tssgc.pt` |

Then run `graph_analysis.ipynb` to train GraphSAGE:

```bash
jupyter notebook notebooks/graph_analysis.ipynb
```

| Order | Model | Saves To |
|---|---|---|
| 5 | GraphSAGE | `checkpoints/graphsage.pt` |

Confirm all five checkpoints exist:

```bash
ls checkpoints/
# transformer.pt  graphsage.pt  cnn_gnn.pt  tssgc.pt  gan_autoencoder.pt
```

---

### Step 4 — Export Models to ONNX

Converts all five trained PyTorch models to ONNX format for quantized sub-200ms inference:

```bash
python scripts/export_onnx.py
```

Expected output:
```
Behavioral Transformer          284.3 KB
GraphSAGE (classifier head)      18.1 KB
CNN-GNN Hybrid                  156.7 KB
TSSGC (SIM Swap)                 92.4 KB
GAN + Autoencoder (KYC)         201.5 KB
─────────────────────────────────────────
  Total                            753.0 KB
All 5 models exported and verified successfully.
```

---

### Step 5 — Verify All Models Load Correctly

Before starting the API, confirm all five models load their checkpoints cleanly:

```bash
python -c "
from app.models.serve import ModelServer
import asyncio

async def check():
    s = ModelServer()
    await s.load_all()
    print()
    for k, v in s.checkpoint_status().items():
        print(f'  {k:<22} {v}')

asyncio.run(check())
"
```

Expected output — all five must show `trained`:
```
  transformer            trained
  graphsage              trained
  cnn_gnn                trained
  tssgc                  trained
  gan_autoencoder        trained
```

If any show ` random weights`, re-run the relevant training cell in the notebook before continuing.

---

### Step 6 — Run Tests

With infrastructure running and checkpoints in place, run the full test suite:

```bash
# Full suite
pytest tests/ -v

# Individual files
pytest tests/test_models.py -v           # model architecture tests
pytest tests/test_decision_engine.py -v  # zone routing and Squad API tests
pytest tests/test_integration.py -v      # webhook and end-to-end tests
```

All tests should pass before starting the API in any environment you intend to demo.

---

### Step 7 — Start the API

```bash
uvicorn app.main:app --reload --port 8000
```

Expected startup output:
```
 Loading models from checkpoints/
   transformer          loaded from checkpoints/transformer.pt
   graphsage            loaded from checkpoints/graphsage.pt
   cnn_gnn              loaded from checkpoints/cnn_gnn.pt
   tssgc                loaded from checkpoints/tssgc.pt
   gan_autoencoder      loaded from checkpoints/gan_autoencoder.pt

  5/5 models loaded from checkpoints.
 Redis connected.
 TrustGuard online — all models loaded, cache connected.

INFO:     Uvicorn running on http://0.0.0.0:8000
```

Confirm the API is live:

```bash
curl http://localhost:8000/health
# {"status": "ok", "service": "TrustGuard"}
```

---

### Alternative — Run Everything via Docker Compose

If you prefer one command over the manual steps above, after completing Steps 2–4 (data generation, training, ONNX export):

```bash
# Start Redis + Neo4j + API together
docker compose up

# Or rebuild first if you changed code
docker compose up --build

# Run data generator as a one-shot container
docker compose --profile datagen up data_generator
```

---

### Quick Reference

```
INFRASTRUCTURE    →   docker run (Redis + Neo4j)
TRAINING DATA     →   python -m synthetic_data_generator.pipeline.run_all
MODEL TRAINING    →   jupyter notebook (model_training.ipynb + graph_analysis.ipynb)
ONNX EXPORT       →   python scripts/export_onnx.py
VERIFY MODELS     →   python -c "..." checkpoint status check
RUN TESTS         →   pytest tests/ -v
START API         →   uvicorn app.main:app --reload --port 8000
```