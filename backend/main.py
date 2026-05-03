from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.trainer_state import trainer_state
from app.routers import prediction, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
)
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Train all coins on startup, release resources on shutdown."""
    log.info("🚀 Starting up — training models for all coins…")
    await trainer_state.train_all()
    log.info("✅ Startup training complete.")
    yield
    log.info("🛑 Shutting down.")


app = FastAPI(
    title="Crypto Price Forecaster API",
    version="2.0.0",
    description="AI-powered ensemble crypto price forecasting (Prophet · SARIMA · LSTM · Transformer)",
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# ALLOWED_ORIGINS env var is a comma-separated list.
# In production: set it to your Vercel frontend URL, e.g.:
#   ALLOWED_ORIGINS=https://your-app.vercel.app
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(prediction.router, prefix="/api", tags=["Prediction"])

# ── Debug (remove after confirming CORS works) ────────────────────────────────
@app.get("/debug")
async def debug():
    return {"allowed_origins": settings.ALLOWED_ORIGINS}
