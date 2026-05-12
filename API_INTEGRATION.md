# TGT API Integration Documentation

## Integrating TGT Fraud Detection into Your Payment Application

---

## Overview

TGT sits as an intelligence layer between your payment application and the Squad API. Your App does not call TGT directly during a transaction, instead TGT listens to Squad webhooks, runs fraud detection in the background, and takes autonomous action (hold, dispute, approve) before settlement completes. This means zero changes to your existing payment flow for the happy path, and automatic fraud interception on the unhappy path.

---

## Architecture Position

```
Your Payment App
      │
      │  initiates payment
      ▼
 Squad API  ────────── fires webhook ──────────► TGT
      │                                               │
      │                                    runs 5 models in parallel
      │                                               │
      │◄────── Proceed / Hold / Dispute API ──────────┘
      │
      ▼
  Settlement
```

TGT intercepts between the Squad webhook event and final settlement. Your app never waits on TGT — the 200 ACK to Squad is immediate, and fraud action happens asynchronously before funds move.

---

## Base URLs

```
Production:  https://TGT-domain.com
Sandbox:     http://localhost:8000
```

---

## Authentication

TGT does not expose a public authentication layer to your app — it authenticates inbound Squad webhooks using HMAC-SHA512 signature verification. Set your Squad webhook secret in TGT's `.env`:

```env
SQUAD_WEBHOOK_SECRET=your_squad_webhook_signing_secret
```

Squad automatically signs every webhook it sends. TGT verifies the signature on every inbound event and rejects anything that doesn't match with HTTP 401.

---

## Step 1 — Register Your Squad Webhook

In your Squad dashboard, point your webhook URL to TGT's webhook receiver:

```
POST https://TGT-domain.com/squad/webhook
```

Subscribe to these events:

```
transaction.success
transfer.initiated
payment_link.paid
```

Squad will send a signed POST request to this endpoint for every matching event. TGT returns HTTP 200 immediately and processes fraud detection in the background.

---

## Step 2 — Webhook Payload

Squad sends this payload structure. TGT reads all fields automatically — you do not need to forward or transform anything:

```json
{
  "Event": "transaction.success",
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

TGT runs the full detection pipeline and takes one of three actions automatically. Your app receives no direct callback for this — the action is taken directly via Squad's API:

### Green Zone — Score < 0.65

Transaction proceeds normally. No action taken. Settlement completes as usual. Your app sees a normal successful transaction.

### Amber Zone — Score 0.65 to 0.89

Settlement is held. TGT triggers step-up authentication. Your app should handle this by listening for the Squad settlement status — if a transaction stays in `pending` state longer than expected, it has been held for re-verification. You can optionally poll TGT's audit log endpoint to get the reason.

### Red Zone — Score ≥ 0.90

TGT calls Squad's Dispute API automatically:

```http
POST https://sandbox-api-d.squadco.com/dispute/transaction/raise-dispute
{
  "transaction_ref": "ABC123DEF456",
  "reason": "TGT RED ZONE: unified score 0.97. Signals: Mule network detected, Behavioral anomaly"
}
```

Funds are frozen in the virtual account before merchant settlement. Your customer's money is protected. The dispute reason string includes the fraud score and which models fired, giving your team immediate context for review.

---

## Step 4 — Optional Direct API Calls

If your app needs to manually trigger a dispute or reversal (for example, from a customer complaint flow), TGT exposes these endpoints directly:

### Raise a Dispute

```http
POST /squad/dispute
Content-Type: application/json

{
  "transaction_ref": "ABC123DEF456",
  "reason": "Customer reported unauthorised transaction"
}
```

**Response:**
```json
{
  "status": "dispute_raised",
  "transaction_ref": "ABC123DEF456",
  "squad_response": {
    "status": 200,
    "message": "Dispute raised successfully"
  }
}
```

---

### Reverse a Transaction

```http
POST /squad/reverse
Content-Type: application/json

{
  "transaction_ref": "ABC123DEF456"
}
```

**Response:**
```json
{
  "status": "reversed",
  "transaction_ref": "ABC123DEF456",
  "squad_response": {
    "status": 200,
    "message": "Transaction reversed"
  }
}
```

---

### Check Dispute Status

```http
GET /squad/dispute/status/{transaction_ref}
```

**Response:**
```json
{
  "status": "pending",
  "transaction_ref": "ABC123DEF456",
  "raised_at": "2025-05-01T14:31:02Z",
  "reason": "TGT RED ZONE: unified score 0.97"
}
```

---

### Get Transaction Details

```http
GET /squad/transaction/{transaction_ref}
```

**Response:**
```json
{
  "transaction_ref": "ABC123DEF456",
  "data": {
    "amount": 5000000,
    "currency": "NGN",
    "status": "disputed",
    "customer_email": "user@example.com"
  }
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
    "action_taken": "DISPUTE_RAISED — funds frozen before settlement",
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
| Squad fires webhook | ~50ms after transaction |
| TGT returns 200 ACK | < 50ms |
| Full pipeline (5 models) | < 200ms |
| Squad Dispute API call | ~100–300ms (Squad network) |
| Total time to funds frozen | < 600ms from transaction |

---

## Error Handling

TGT is designed to fail safe — if the fraud pipeline errors internally, it does not block or reverse the transaction. Errors are logged but the transaction proceeds. This ensures TGT never becomes a single point of failure for your payment flow.

| Scenario | TGT Behaviour |
|---|---|
| Redis unavailable | Falls back to Squad API for transaction history |
| Neo4j unavailable | Uses synthetic graph snapshot, continues inference |
| Model inference timeout | Returns conservative mid-range score (0.50) |
| Squad Dispute API fails | Error logged, retry queued, transaction flagged for manual review |
| Invalid webhook signature | HTTP 401, pipeline does not run |
| Malformed JSON payload | HTTP 400, pipeline does not run |

---

## Integration Checklist

```
[ ] Squad webhook URL set to: https://your-TGT-domain.com/squad/webhook
[ ] Webhook events subscribed: transaction.success, transfer.initiated, payment_link.paid
[ ] SQUAD_WEBHOOK_SECRET matches value in Squad dashboard
[ ] SQUAD_SECRET_KEY set in TGT .env for outbound API calls
[ ] SQUAD_BASE_URL set correctly (sandbox vs production)
[ ] /health endpoint returns {"status": "ok"} before going live
[ ] Redis running and reachable at REDIS_URL
[ ] Neo4j running and reachable at NEO4J_URI
[ ] All 5 model checkpoints present in /checkpoints
[ ] Audit log directory writable at LOG_PATH
```

---

## Example: Full Integration in a Node.js Payment App

This shows what your existing payment app looks like — TGT requires zero changes to this code. It simply listens on the Squad webhook side.

```javascript
// Your existing payment initiation — unchanged
const initiatePayment = async (req, res) => {
  const { amount, email, callbackUrl } = req.body;

  const response = await axios.post(
    'https://sandbox-api-d.squadco.com/transaction/initiate',
    {
      amount,
      email,
      currency: 'NGN',
      callback_url: callbackUrl,
      transaction_ref: generateRef(),
    },
    {
      headers: { Authorization: `Bearer ${process.env.SQUAD_SECRET_KEY}` }
    }
  );

  // TGT is listening on the Squad webhook side.
  // If this transaction is fraudulent, TGT will raise a dispute
  // automatically before settlement — your app code here is unchanged.
  res.json({ checkoutUrl: response.data.data.checkout_url });
};


// Your webhook receiver — forward Squad events to TGT
// OR point Squad directly at TGT's /squad/webhook endpoint
// and remove this entirely.
app.post('/squad/webhook', async (req, res) => {
  await axios.post(
    'https://TGT-domain.com/squad/webhook',
    req.body,
    { headers: { 'x-squad-encrypted-body': req.headers['x-squad-encrypted-body'] } }
  );
  res.sendStatus(200);
});
```

If you point Squad's webhook URL directly at TGT, you don't need the forwarding route above at all — TGT handles everything end to end.

---

## Example: Full Integration in a Python Payment App

```python
import httpx
import os

TGT_URL = os.getenv("TGT_URL", "http://localhost:8000")
SQUAD_KEY      = os.getenv("SQUAD_SECRET_KEY")


async def initiate_payment(amount: int, email: str, callback_url: str) -> dict:
    """Initiate a Squad payment — TGT monitors via webhook, no changes here."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://sandbox-api-d.squadco.com/transaction/initiate",
            json={
                "amount":          amount,
                "email":           email,
                "currency":        "NGN",
                "callback_url":    callback_url,
                "transaction_ref": generate_ref(),
            },
            headers={"Authorization": f"Bearer {SQUAD_KEY}"},
        )
    return response.json()


async def manual_dispute(transaction_ref: str, reason: str) -> dict:
    """Manually raise a dispute — for customer complaint flows."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{TGT_URL}/squad/dispute",
            json={"transaction_ref": transaction_ref, "reason": reason},
        )
    return response.json()


async def get_fraud_decision(transaction_ref: str) -> dict:
    """Fetch TGT's audit record for a transaction."""
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{TGT_URL}/audit/recent",
            params={"limit": 100},
        )
    records = response.json()
    return next(
        (r for r in records if r["transaction_ref"] == transaction_ref),
        None
    )
```

---

## Support & Debugging

**TGT is not receiving webhooks**
Confirm the webhook URL is reachable from Squad's servers. Test with:
```bash
curl -X POST https://TGT-domain.com/squad/webhook \
  -H "Content-Type: application/json" \
  -d '{"Event": "test", "TransactionRef": "TEST001", "Body": {}}'
# Expected: {"status": "ignored", "event": "test"}
```

**Transactions not being disputed despite high fraud signals**
Check that `SQUAD_SECRET_KEY` has dispute permissions in your Squad dashboard. Sandbox keys may require explicit dispute scope.

**All scores returning 0.5**
Models are running on random weights — checkpoints not loaded. Run:
```bash
python -c "
from app.models.serve import ModelServer
import asyncio
async def check():
    s = ModelServer()
    await s.load_all()
    for k, v in s.checkpoint_status().items():
        print(f'{k}: {v}')
asyncio.run(check())
"
```

**Inference taking > 200ms**
Redis is not caching correctly. Check `REDIS_URL` in `.env` and confirm Redis is running:
```bash
docker ps | grep redis
redis-cli ping   # should return PONG
```