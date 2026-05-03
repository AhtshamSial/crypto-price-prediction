from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

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
# Using a manual middleware instead of FastAPI's CORSMiddleware because
# starlette's implementation returns 400 on OPTIONS preflights when the
# origin list is non-empty in certain versions.
class CORSMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        origin = request.headers.get("origin", "")
        allowed = settings.ALLOWED_ORIGINS

        cors_headers = {
            "Access-Control-Allow-Origin": origin if origin in allowed else "",
            "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, X-Request-ID",
            "Access-Control-Max-Age": "600",
        }

        # Return 200 immediately for preflight requests
        if request.method == "OPTIONS":
            if origin in allowed:
                return Response(status_code=200, headers=cors_headers)
            return Response(status_code=403)

        # For actual requests, add CORS headers to the response
        response = await call_next(request)
        if origin in allowed:
            for key, value in cors_headers.items():
                response.headers[key] = value
        return response


app.add_middleware(CORSMiddleware)

# ── Routers ──────────────────────────────────────────────────────────────────
app.include_router(health.router, prefix="/api", tags=["Health"])
app.include_router(prediction.router, prefix="/api", tags=["Prediction"])


# ── Debug (remove after confirming CORS works) ────────────────────────────────
@app.get("/debug")
async def debug():
    return {"allowed_origins": settings.ALLOWED_ORIGINS}