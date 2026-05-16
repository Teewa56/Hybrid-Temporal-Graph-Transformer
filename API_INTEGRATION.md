# TGT API Integration Documentation

## Integrating TGT Fraud Detection with Your Payment Backend

---

## Overview

TGT sits as an intelligence layer between your payment application and your NeoBank backend. Your backend sends signed transaction webhook events to TGT, which runs fraud detection in the background and escalates suspicious transactions through the backend refund API.

---

## Architecture Position

```
Your Payment App
      │
      │  initiates payment
      ▼
Payment Backend  ───── fires webhook ──────► TGT
      │                                       │
      │                            runs 5 models in parallel
      │                                       │
      │◄────── Request refund / notify review ──┘
      │
      ▼
  Settlement
```

TGT receives webhook events from the backend, returns an immediate 200 ACK, and processes fraud detection asynchronously.

---

## Base URLs

```
Production:  https://TGT-domain.com
Sandbox:     http://localhost:8000
```

---

## Authentication

TGT authenticates inbound payment webhook events using HMAC-SHA512 signature verification. Set your webhook secret in TGT's `.env`:

```env
NEOBANK_WEBHOOK_SECRET=your_webhook_signing_secret
```

The backend signs every webhook it sends. TGT verifies the signature on every inbound event and rejects anything that doesn't match with HTTP 401.

---

## Step 1 — Register Your Backend Webhook

In your backend configuration, point your webhook URL to TGT's webhook receiver:

```
POST https://TGT-domain.com/webhook
```

Subscribe to your transaction completion events such as:

```
charge_successful
transaction.completed
payment_link.paid
```

The backend sends a signed POST request to this endpoint for every matching event. TGT returns HTTP 200 immediately and processes fraud detection in the background.

---

## Step 2 — Webhook Payload

The backend sends a transaction payload. TGT reads all fields automatically and normalises the payload for scoring.

```json
{
  "Event": "charge_successful",
  "TransactionRef": "ABC123DEF456",
  "Body": {
    "amount": 5000000,
    "currency": "NGN",
    "customer_email": "user@example.com",
    "customer_name": "Amaka Okonkwo",
    "ip_address": "102.1.2.3",
    "device_id": "device-uuid-here",
    "channel": "app",
    "merchant_category": "transfer",
    "transaction_type": "debit",
    "created_at": "2025-05-01T14:30:00Z",
    "meta": {
      "receiver_account": "0123456789",
      "receiver_bank_code": "058"
    }
  }
}
```

**Fields TGT uses for fraud scoring:**

| Field | Model | Purpose |
|---|---|---|
| `amount` | Transformer, CNN-GNN | Amount anomaly detection |
| `customer_email` | Transformer | Behavioral sequence lookup |
| `ip_address` | CNN-GNN, GraphSAGE | Network cluster analysis |
| `device_id` | TSSGC, Transformer | Device fingerprint matching |
| `channel` | Ensemble | Weight adjustment per channel |
| `created_at` | Transformer | Time-of-day anomaly |
| `meta.receiver_account` | GraphSAGE | Mule network graph lookup |

**Optional fields that improve accuracy — send if available:**

```json
{
  "is_new_device": false,
  "is_new_recipient": true,
  "customer_phone": "08031234567",
  "imei": "123456789012345",
  "carrier": "MTN"
}
```

---

## Step 3 — What Happens After the Webhook

TGT runs the full detection pipeline and takes one of three actions automatically. TGT does not block the transaction in real time; it scores the transaction and escalates suspicious activity to the backend.

### Green Zone — Score < 0.65

Transaction proceeds normally. No action taken.

### Amber Zone — Score 0.65 to 0.89

TGT flags the transaction for review and may trigger step-up authentication. Your operations flow should handle manual review or customer verification when a transaction remains suspicious.

### Red Zone — Score ≥ 0.90

TGT submits a refund request to the backend:

```http
POST https://api.neobank.example/transaction/refund
{
  "transaction_ref": "ABC123DEF456",
  "gateway_transaction_ref": "GATEWAY123",
  "reason_for_refund": "TGT RED ZONE: unified score 0.97. Signals: Mule network detected, Behavioral anomaly"
}
```

This is a best-effort reverse action after settlement. The backend decides whether the refund or reversal can be processed.

---

## Step 4 — Optional Direct API Calls

If your app needs to trigger a refund or dispute manually, use TGT's direct endpoints:

### Initiate a Refund

```http
POST /dispute/refund
Content-Type: application/json

{
  "transaction_ref": "ABC123DEF456",
  "reason": "Customer reported unauthorised transaction"
}
```

**Response:**
```json
{
  "status": "refund_initiated",
  "transaction_ref": "ABC123DEF456",
  "backend_response": {
    "status": 200,
    "message": "Refund request accepted"
  }
}
```

---

### Get Transaction Details

```http
GET /transactions/transaction/{transaction_ref}
```

**Response:**
```json
{
  "transaction_ref": "ABC123DEF456",
  "data": {
    "amount": 5000000,
    "currency": "NGN",
    "status": "confirmed",
    "customer_email": "user@example.com"
  }
}
```

---

### Fetch Customer Transaction History

```http
GET /transactions/customer/{customer_identifier}
```

**Response:**
```json
{
  "customer_identifier": "customer_123",
  "count": 5,
  "transactions": [ ... ]
}
```

---

## Step 5 — Reading Audit Logs

Every Amber and Red Zone decision is logged with a full explanation. Your compliance or operations team can query recent decisions:

```http
GET /audit/recent?limit=50
```

**Response:**
```json
[
  {
    "transaction_ref": "ABC123DEF456",
    "timestamp": "2025-05-01T14:31:02Z",
    "unified_score": 0.97,
    "zone": "RED",
    "action_taken": "REFUND_INITIATED — funds escalated to backend",
    "scores_breakdown": {
      "transformer": 0.85,
      "graphsage": 0.98,
      "cnn_gnn": 0.42,
      "tssgc": 0.31,
      "gan_autoencoder": 0.22,
      "unified": 0.97
    },
    "top_signals": [
      "Mule network detected (GraphSAGE)",
      "Behavioral anomaly (Transformer)"
    ],
    "customer_email": "user@example.com",
    "amount": 50000.00,
    "channel": "app",
    "model_version": "1.0.0"
  }
]
```

---

## Step 6 — Health Check

Use this to confirm TGT is live before processing transactions:

```http
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "service": "TGT"
}
```

---

## Latency Expectations

| Stage | Expected Time |
|---|---|
| Backend fires webhook | ~50ms after transaction |
| TGT returns 200 ACK | < 50ms |
| Full pipeline (5 models) | < 200ms |
| Backend refund request | ~100–300ms |
| Total time to escalation | < 600ms from transaction |

---

## Error Handling

TGT is designed to fail safe — if the fraud pipeline errors internally, it does not block or reverse the transaction. Errors are logged but the transaction proceeds.

| Scenario | TGT Behaviour |
|---|---|
| Redis unavailable | Falls back to cache-only lookup |
| Neo4j unavailable | Uses synthetic graph snapshot, continues inference |
| Model inference timeout | Returns conservative mid-range score (0.50) |
| Backend refund API fails | Error logged, retry queued, transaction flagged for manual review |
| Invalid webhook signature | HTTP 401, pipeline does not run |
| Malformed JSON payload | HTTP 400, pipeline does not run |

---

## Integration Checklist

```
[ ] Backend webhook URL set to: https://your-TGT-domain.com/webhook
[ ] Webhook event subscription configured for charge_successful or transaction.completed
[ ] NEOBANK_WEBHOOK_SECRET matches value in backend configuration
[ ] NEOBANK_SECRET_KEY set in TGT .env for outbound API calls
[ ] NEOBANK_BASE_URL set correctly (sandbox vs production)
[ ] /health endpoint returns {"status": "ok"} before going live
[ ] Redis running and reachable at REDIS_URL
[ ] Neo4j running and reachable at NEO4J_URI
[ ] All 5 model checkpoints present in /checkpoints
[ ] Audit log directory writable at LOG_PATH
```
