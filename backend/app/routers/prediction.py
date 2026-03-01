"""
app/routers/prediction.py
==========================
Prediction endpoints.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import PredictRequest, PredictResponse
from app.services.prediction_service import run_prediction

log = logging.getLogger("prediction_router")
router = APIRouter()


@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    """
    Run ensemble forecast for the requested coin.

    Returns live market data, technical indicators, and price forecasts
    at horizons: 1w, 1m, 3m, 6m, 1y — with 95% confidence intervals.
    """
    try:
        result = await run_prediction(
            coin=req.coin,
            run_backtest=req.run_backtest,
        )
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        log.exception("Unexpected error during prediction for %s: %s", req.coin, exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(exc)}",
        )
