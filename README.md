# Advanced Fraud Detection for African FinTechs

A real-time, AI-powered fraud detection and trust-scoring engine built on top of the **Squad payment API**. Hybrid-Temporal-Graph-Transformer uses a Hybrid Temporal Graph Transformer (TGT) architecture — an ensemble of five specialized deep learning models — to detect, score, and intercept fraudulent transactions before settlement.

---

## 📌 Table of Contents

- [Problem Statement](#-problem-statement)
- [Solution Overview](#-solution-overview)
- [Architecture](#-architecture)
  - [Core Models](#core-models)
  - [System Integration](#system-integration)
  - [Decision Engine](#decision-engine)
  - [Adaptive Learning](#adaptive-learning)
- [Tech Stack](#-tech-stack)
- [Squad API Integration](#-squad-api-integration)
- [Getting Started](#-getting-started)
- [Project Structure](#-project-structure)
- [Low-Data Strategies](#-low-data-strategies)
- [Compliance & Explainability](#-compliance--explainability)

---

## 🚨 Problem Statement

Africa's FinTech ecosystem is under siege from fraud that existing systems were never built to catch. Nigerian FinTechs face a uniquely dangerous combination of:

- **SIM Swap Fraud** — criminals hijack mobile numbers to bypass authentication and authorize transfers
- **Synthetic/Identity KYC Fraud** — AI-generated or forged documents used to open fraudulent accounts
- **Social Engineering (Vishing/Smishing)** — victims manipulated into authorizing transfers to mule accounts
- **Payment Injection** — unauthorized transaction requests injected directly into the API layer
- **Coordinated Fraud Rings** — networks of linked accounts operating as money laundering infrastructure

Classical rule-based systems and standard ML models (Random Forest, Logistic Regression) fail here because fraud in Africa is **relational, behavioral, and adaptive** — it mutates faster than static models can respond. Every verification failure has a real cost: real people lose money, real businesses lose trust, and financial inclusion gets pushed further out of reach.

---

## 💡 Solution Overview

This is a **Hybrid Temporal Graph Transformer (TGT)** — a real-time AI fraud detection and trust-scoring engine deployed as a closed-loop intelligence layer directly on top of the Squad payment API.

Rather than a single model making a single guess, the TGT is an **ensemble of five specialized deep learning sub-architectures**, each purpose-built for a specific fraud surface, running in parallel and feeding a unified Decision Engine that produces a single, interpretable trust score for every transaction.

**Key outcomes:**
-  Sub-200ms inference latency — fraud check completes before the user sees "Processing"
-  Unified Fraud Score (0–1) with three-zone routing: Green / Amber / Red
-  Automatic Squad Dispute API call on Red Zone detections — funds frozen before settlement
-  SHAP-based explainability on every blocked transaction for CBN compliance
-  Continual learning loop — the model gets sharper with every new fraud attempt

---

## 🧠 Architecture

### Core Models

The TGT ensemble consists of five sub-architectures running **in parallel**:

| # | Fraud Type | Architecture | Key Detection Logic |
|---|---|---|---|
| 1 | Behavioral Analysis | Transformer Encoder / LBSF | Flags deviations from long-term spending rhythm using self-attention over last 50 transactions |
| 2 | Social Engineering | GraphSAGE (Inductive GNN) | Detects mule account networks via connection topology — even on accounts created hours ago |
| 3 | Payment Injection | CNN-GNN Hybrid | CNN catches payload-level tampering; GNN catches fund flow inconsistencies |
| 4 | SIM Swap | TSSGC (Temporal-Spatial-Semantic GNN) | Detects device metadata mismatches vs. historical account fingerprint |
| 5 | Identity / KYC Fraud | GAN + Autoencoder | Flags pixel-level artifacts, biometric divergence, and metadata mismatches in KYC docs |

---

#### 2.1 Behavioral Analysis → Transformer Encoder / LBSF

The model ingests a rolling window of the user's last 50 transactions and builds a **spending rhythm profile** — encoding time-of-day patterns, merchant categories, location clusters, and transaction velocity. The **Self-Attention mechanism** learns which past actions are most relevant to the current one, enabling detection of anomalies like a 3 AM large transfer from a new device in a different city — even when each individual signal alone seems benign.

#### 2.2 Social Engineering → GraphSAGE (Inductive GNN)

The entire payment ecosystem is modeled as a heterogeneous graph: accounts, devices, IPs, and merchants are *nodes*; transactions and shared device usage are *edges*. **Inductive learning** allows detection of new mule accounts by their connection topology to known fraudulent clusters — no prior transaction history required. A 2025 study on Nigerian FinTechs using GNNs + Isolation Forest achieved an **F1-score of 0.92 and AUC-ROC of 0.98** on real anonymized data.

#### 2.3 Payment Injection → CNN-GNN Hybrid

A two-stage hybrid: the **CNN** analyzes raw transaction payloads for structural anomalies — malformed fields, unusual metadata, or tampered request signatures. The **GNN layer** then validates whether the transaction makes logical sense within the current network state. The CNN catches *data-level tampering*; the GNN catches *flow-level inconsistency*.

#### 2.4 SIM Swap → TSSGC

**TSSGC** monitors the **Handover Event** — the moment a SIM swap causes device metadata (IMEI, IMSI, carrier signals) to change abruptly while the account identity remains the same. The architecture identifies the **semantic mismatch** between the behavioral fingerprint of the new device and the historical profile anchored to that account, in real-time, before any transfer is authorized.

#### 2.5 Identity & KYC Fraud → GAN + Autoencoder

A **GAN** is trained to understand how fraudulent IDs are synthesized, learning the distribution of forgeries. The **Discriminator** is then fine-tuned to flag high-dimensional inconsistencies in submitted KYC documents — pixel-level artifacts, metadata mismatches, font irregularities, and biometric divergence — invisible to human reviewers. GANs and VAEs also generate synthetic fraud samples to address chronic class imbalance in African FinTech fraud datasets.

---

### System Integration

```
User Transaction
      │
      ▼
 Squad Webhook  ───────────────────────────────────────────────────┐
 (transaction.success / transfer.initiated)                        │
      │                                                            │
      ▼                                                            │
  Redis Cache (sub-ms retrieval)                                   │
      │                                                            │
      ├──────────────────┬─────────────────────────────────────┐   │
      ▼                  ▼                                     │   │
Sequential Service   Graph Service                             │   │
(last 50 txns →     (live transaction                          │   │
 sequence vector)    graph: accounts,                          │   │
                     devices, IPs)                             │   │
      │                   │                                    │   │
      └────────┬──────────┘                                    │   │
               ▼                                               │   │
     ┌─────────────────────────────────┐                       │   │
     │     TGT Model Ensemble          │                       │   │
     │  (All 5 models run in parallel) │                       │   │
     │                                 │                       │   │
     │  Transformer ──► Risk Score     │                       │   │
     │  GraphSAGE   ──► Risk Score     │                       │   │
     │  CNN-GNN     ──► Risk Score     │                       │   │
     │  TSSGC       ──► Risk Score     │                       │   │
     │  GAN Disc.   ──► Risk Score     │                       │   │
     └──────────────┬──────────────────┘                       │   │
                    ▼                                          │   │
           Decision Engine                                     │   │
           (Weighted Average)                                  │   │
                    │                                          │   │
        ┌───────────┼────────────────┐                         │   │
        ▼           ▼                ▼                         │   │
  Green (<0.65) Amber (0.65-0.89) Red (≥0.90)                  │   │
  Proceed       Step-up Auth      Squad Dispute API ───────────┘   │
                (Face ID / OTP)   Freeze funds                     │
                                                                   │
                                  ◄────────────────────────────────┘
                              Squad API handles execution
```

---

### Decision Engine

A weighted average of all five risk scores produces a **Unified Fraud Score**. Routing logic:

| Zone | Score Range | Action |
|---|---|---|
| 🟢 Green | < 0.65 | Transaction proceeds normally |
| 🟡 Amber | 0.65 – 0.89 | Step-up authentication triggered (Face ID or OTP). Squad settlement held pending re-verification |
| 🔴 Red | ≥ 0.90 | Squad Dispute/Reverse API called automatically. Funds frozen in virtual account before merchant settlement |

To achieve **sub-200ms inference** — the hard latency ceiling for payment systems — all models are compressed using **ONNX / TensorRT model quantization**.

---

### Adaptive Learning

**Target:** Staying ahead of novel fraud tactics without retraining from scratch.

The TGT system incorporates a **Continual Learning Loop** built on:

- **Concept Drift Detector (ADWIN / Page-Hinkley)** — monitors the statistical distribution of incoming transaction features in real-time. Significant shifts trigger an alert.
- **Active Learning Queue** — uncertain predictions are routed for priority human review. Confirmed labels are fed back into the training pipeline.
- **Elastic Weight Consolidation (EWC)** — incrementally fine-tunes weights for new fraud patterns while mathematically penalizing changes to weights critical for already-learned types, preventing **catastrophic forgetting**.
- **MA-GAD Meta-Learning Module** — for entirely novel fraud variants, bootstraps detection from as few as **5–10 confirmed cases** by generalizing from structurally similar past patterns.
- **Shadow Deployment Pipeline** — updated models run in parallel with the live model on real traffic before promotion, ensuring zero production degradation.

---

## 🛠️ Tech Stack

| Layer | Tool |
|---|---|
| API Framework | Python + FastAPI |
| GNN Models | PyTorch Geometric |
| Transformer Models | HuggingFace Transformers |
| Graph Database | Neo4j (account relationship store) |
| In-memory Cache | Redis (sub-ms behavioral lookups) |
| Model Serving | ONNX / TensorRT (quantized inference) |
| Infrastructure | AWS |
| Explainability | SHAP + Attention Weights (CBN compliance) |
| Payment API | SquadCo API |

---

## 🔌 Squad API Integration

Squad API is not peripheral to this system — it is **the execution layer**. The integration is deep and bidirectional:

**Inbound (Trigger):**
```
POST /squad/webhook
Events: transaction.success | transfer.initiated | payment_link.paid
```
Every transaction event fires a webhook that initiates the fraud detection pipeline.

**Outbound (Response):**
```
POST /squad/dispute          → Freeze funds on Red Zone detection
POST /squad/transfer/reverse → Reverse settled transactions flagged post-hoc
GET  /squad/transaction/{id} → Retrieve transaction details for feature engineering
```

**Feature inputs from Squad payload:**
- `amount`, `currency`, `transaction_ref`
- `customer_email`, `customer_name`
- `ip_address`, `device_metadata`
- `merchant_id`, `payment_link_ref`

Squad settlement is **held programmatically** during Amber Zone step-up authentication, ensuring funds are not released to merchants until identity is re-verified.

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.10+
Redis (running locally or via Docker)
Neo4j 5.x
Node.js 18+ (for webhook listener)
Squad API credentials (test environment)
```

### Installation

```bash
# Clone the repository
git clone https://github.com/Teewa56/Hybrid-Temporal-Graph-Transformer.git
cd Hybrid-Temporal-Graph-Transformer

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Fill in: SQUAD_SECRET_KEY, SQUAD_BASE_URL, NEO4J_URI, REDIS_URL
```

### Running the System

```bash
# Start Redis
docker run -d -p 6379:6379 redis

# Start Neo4j
docker run -d -p 7474:7474 -p 7687:7687 neo4j

# Start the API server
uvicorn app.main:app --reload --port 8000

# In a separate terminal, run model inference server
python -m app.models.serve
```

### Running Tests

```bash
pytest tests/ -v
pytest tests/test_integration.py -v  # Squad API integration tests
```

### Order to run project
```
# 1. Generate all synthetic training data
python -m synthetic_data_generator.pipeline.run_all

# 2. Train all 5 models (run model_training.ipynb top to bottom)
jupyter notebook notebooks/model_training.ipynb

# 3. Export all models to ONNX
python scripts/export_onnx.py

# 4. Verify checkpoint status
python -c "
from app.models.serve import ModelServer
import asyncio

async def check():
    s = ModelServer()
    await s.load_all()
    for k, v in s.checkpoint_status().items():
        print(f'{k:<22} {v}')

asyncio.run(check())
"

# 5. Start the API
uvicorn app.main:app --reload --port 8000
```

### Environment Variables

```env
SQUAD_SECRET_KEY=your_squad_secret_key
SQUAD_BASE_URL=https://sandbox-api-d.squadco.com
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
REDIS_URL=redis://localhost:6379
MODEL_QUANTIZATION=onnx
INFERENCE_TIMEOUT_MS=200
```

---

## 📁 Project Structure

```
Hybrid-Temporal-Graph-Transformer/
├── app/
│   ├── main.py                    # FastAPI entrypoint
│   ├── api/
│   │   ├── webhooks.py            # Squad webhook receiver
│   │   ├── disputes.py            # Squad Dispute API caller
│   │   └── transactions.py        # Transaction data fetcher
│   ├── models/
│   │   ├── transformer.py         # Behavioral Transformer model
│   │   ├── graphsage.py           # Social Engineering GNN
│   │   ├── cnn_gnn.py             # Payment Injection hybrid
│   │   ├── tssgc.py               # SIM Swap detector
│   │   ├── gan_autoencoder.py     # KYC Fraud detector
│   │   ├── ensemble.py            # Model ensemble + scoring
│   │   └── serve.py               # ONNX/TensorRT inference server
│   ├── services/
│   │   ├── sequential_service.py  # Builds transaction sequence vectors
│   │   ├── graph_service.py       # Manages live transaction graph
│   │   ├── cache_service.py       # Redis integration
│   │   └── decision_engine.py     # Unified Fraud Score + routing
│   ├── continual_learning/
│   │   ├── drift_detector.py      # ADWIN / Page-Hinkley
│   │   ├── active_learning.py     # Uncertain prediction queue
│   │   ├── ewc.py                 # Elastic Weight Consolidation
│   │   └── shadow_pipeline.py     # Shadow deployment manager
│   └── explainability/
│       ├── shap_logger.py         # SHAP value computation
│       └── audit_trail.py         # CBN-compliant decision logs
├── data/
│   ├── synthetic/                 # GAN-generated synthetic fraud samples
│   ├── paysim/                    # PaySim base dataset
│   └── preprocessing/             # Feature engineering scripts
├── tests/
│   ├── test_models.py
│   ├── test_integration.py        # Squad API integration tests
│   └── test_decision_engine.py
├── notebooks/
│   ├── model_training.ipynb
│   ├── graph_analysis.ipynb
│   └── drift_simulation.ipynb
├── synthetic_data_generator/
│   ├── config.py
│   ├── README.md
│   │
│   ├── behavioral/
│   │   ├── __init__.py
│   │   ├── user_profile_generator.py
│   │   ├── transaction_sequence_generator.py
│   │   └── anomaly_injector.py
│   │
│   ├── graph/
│   │   ├── __init__.py
│   │   ├── graph_builder.py
│   │   ├── mule_network_simulator.py
│   │   └── fraud_ring_injector.py
│   │
│   ├── payload/
│   │   ├── __init__.py
│   │   ├── legitimate_payload_generator.py
│   │   ├── payload_anomaly_injector.py
│   │   └── squad_payload_schema.py
│   │
│   ├── sim_swap/
│   │   ├── __init__.py
│   │   ├── device_profile_generator.py
│   │   └── handover_event_simulator.py
│   │
│   ├── kyc/
│   │   ├── __init__.py
│   │   ├── document_metadata_generator.py
│   │   └── forgery_simulator.py
│   │
│   └── pipeline/
│       ├── __init__.py
│       ├── run_all.py
│       └── export.py
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Data Reference Table

| Model | Primary Dataset | Backup / Augmentation |
|---|---|---|
| Transformer | PaySim + IEEE-CIS | VAE-generated sequences |
| GraphSAGE | Elliptic + PaySim graph | Barabási–Albert synthetic graphs |
| CNN-GNN | UNSW-NB15 + Squad sandbox payloads | Programmatic anomaly injection |
| TSSGC | Synthetic simulation | GSMA-calibrated generators |
| GAN + Autoencoder | FantasyID | Self-generated GAN forgeries |

---

## 📊 Low-Data Strategies

African FinTechs rarely have large, clean, labeled fraud datasets. Hybrid-Temporal-Graph-Transformer addresses this through five strategies:

| Strategy | Implementation |
|---|---|
| **Self-supervised pre-training** | Transformers and GNNs pre-trained via masked transaction prediction and contrastive learning on unlabeled normal transaction data — no fraud labels required |
| **Synthetic data augmentation** | GANs and VAEs generate realistic fraud samples; PaySim (public mobile money simulation dataset) used as transfer learning base |
| **Few-shot meta-learning (MA-GAD)** | Adapts to new fraud variants from as few as 5–10 confirmed examples |
| **Active Learning** | System prioritizes uncertain predictions for human review, maximizing value of every new labeled example |
| **PU Contrastive Learning** | Handles sparse fraud labels by learning from abundant confirmed-normal transaction data |

---

## 🔍 Compliance & Explainability

Every decision made by Hybrid-Temporal-Graph-Transformer is explainable and auditable:

- **SHAP Values** — computed for every blocked transaction, surfacing the top contributing features to the fraud score
- **Transformer Attention Weights** — expose which transactions in the user's history most influenced the behavioral anomaly flag
- **Audit Trail Logging** — every Red Zone decision is stored with timestamp, score breakdown, contributing features, and action taken
- **CBN Regulatory Alignment** — explainability outputs satisfy Central Bank of Nigeria requirements for automated financial decision-making

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---
