"""
main.py
=======
FastAPI entry point for the Crypto Price Forecaster.

KEY CHANGE vs original:
  Old: train ALL coins synchronously during startup -> Railway kills the process
       before it finishes (OOM or health-check timeout).
  New: server starts IMMEDIATELY and responds to /api/health right away.
       Training runs coin-by-coin in a background thread.
       The frontend polls /api/health until status == "ready".
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.trainer_state import trainer_state
from app.routers import prediction, health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
log = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Server starts instantly. Training is kicked off as a background task
    so Railway's health-check port bind succeeds immediately.
    """
    log.info("Starting server - launching background training task...")

    # Fire-and-forget: train all coins without blocking server startup
    asyncio.create_task(trainer_state.train_all_background())

    log.info("Server ready - training running in background. Poll /api/health for progress.")
    yield
    log.info("Shutting down.")


app = FastAPI(
    title="Crypto Price Forecaster API",
    version="2.0.0",
    description="AI-powered ensemble crypto price forecasting",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(prediction.router, prefix="/api", tags=["Prediction"])
