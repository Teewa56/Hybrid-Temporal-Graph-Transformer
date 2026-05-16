import time
import hmac
import hashlib
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import webhooks, disputes, transactions
from app.services.cache_service import CacheService
from app.models.serve import ModelServer
from app.continual_learning.drift_detector import DriftDetector


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    app.state.cache = CacheService()
    await app.state.cache.connect()

    app.state.model_server = ModelServer()
    await app.state.model_server.load_all()

    app.state.drift_detector = DriftDetector()

    print("TGT online — all models loaded, cache connected.")
    yield

    # Shutdown
    await app.state.cache.disconnect()
    print("TGT shutting down.")


app = FastAPI(
    title="TGT — Fraud Detection Engine",
    description="Hybrid Temporal Graph Transformer fraud detection layer for African FinTechs.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_latency_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    latency_ms = (time.time() - start) * 1000
    response.headers["X-Inference-Latency-Ms"] = f"{latency_ms:.2f}"
    return response


app.include_router(webhooks.router, prefix="/webhook", tags=["Webhooks"])
app.include_router(disputes.router, prefix="/dispute", tags=["Disputes & Refunds"])
app.include_router(transactions.router, prefix="/transactions", tags=["Transactions"])


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Hybrid_Temporal_Graph_Transformer"}


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "path": str(request.url)},
    )