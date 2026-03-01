"""
app/services/prediction_service.py
===================================
All prediction / forecast business logic lives here.
Routers stay thin — they only call these functions.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from app.core.trainer_state import trainer_state
from app.models.schemas import ForecastData, LiveMarket, PredictResponse

log = logging.getLogger("prediction_service")


def _safe_list(arr, length: int) -> list:
    """Convert a numpy array (or None) to a plain Python list, NaN → None."""
    if arr is None:
        return [None] * length
    out = []
    for v in arr[:length]:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append(None)
        else:
            out.append(round(float(v), 6))
    return out


def _build_forecast_data(result) -> ForecastData:
    """Convert a ForecastResult dataclass into our Pydantic ForecastData model."""
    n = len(result.horizons)
    return ForecastData(
        horizons=list(result.horizons),
        current_price=float(result.current_price) if result.current_price else None,
        ensemble=_safe_list(result.ensemble_preds, n),
        ensemble_lower=_safe_list(result.ensemble_lower, n),
        ensemble_upper=_safe_list(result.ensemble_upper, n),
        ensemble_uncertainty=_safe_list(
            getattr(result, "ensemble_uncertainty", None), n
        ),
        prophet=_safe_list(result.prophet_preds, n),
        sarima=_safe_list(result.sarima_preds, n),
        lstm=_safe_list(result.lstm_preds, n),
        transformer=_safe_list(result.transformer_preds, n),
    )


def _build_live_market(ticker: Dict) -> LiveMarket:
    def _f(k):
        v = ticker.get(k)
        return float(v) if v is not None else None

    return LiveMarket(
        price=_f("price"),
        price_change_pct_24h=_f("price_change_pct_24h"),
        volume_24h=_f("volume_24h"),
        bid=_f("bid"),
        ask=_f("ask"),
        spread=_f("spread"),
        timestamp=ticker.get("timestamp"),
    )


def _build_indicators(df: pd.DataFrame) -> Dict[str, float]:
    row = df.iloc[-1]
    return {
        k: round(float(v), 6)
        for k, v in row.items()
        if pd.notna(v) and isinstance(v, (int, float, np.floating))
    }


async def run_prediction(coin: str, run_backtest: bool = False) -> PredictResponse:
    """
    Main prediction entry point.
    Raises ValueError if coin not trained or not supported.
    """
    coin = coin.upper()

    if not trainer_state.is_ready:
        raise RuntimeError("Models are still training. Please try again in a moment.")

    data = trainer_state.get_coin_data(coin)
    if data is None:
        trained = trainer_state.trained_coins
        raise ValueError(
            f"Coin '{coin}' is not available. "
            f"Trained coins: {', '.join(trained) if trained else 'none yet'}."
        )

    trainer     = data["trainer"]
    df: pd.DataFrame = data["df"]
    X: np.ndarray    = data["X"]
    y: np.ndarray    = data["y"]
    dm               = data["dm"]
    model_training   = data["model_training"]

    # Fetch live ticker
    ticker: Dict = {}
    try:
        ticker = dm.live_ticker()
    except Exception as exc:
        log.warning("Live ticker failed for %s: %s", coin, exc)
        ticker = {}

    live_price    = ticker.get("price")
    current_price = float(df["close"].dropna().iloc[-1])
    latest_window = X[-1:]

    # Forecast
    forecast_result = trainer.forecast(
        latest_window, current_price=current_price, live_price=live_price
    )

    # Validation metrics
    val_metrics: Dict = {}
    try:
        val_metrics = trainer.validate() or {}
    except Exception as exc:
        log.warning("Validation metrics failed for %s: %s", coin, exc)

    # Backtest (optional, expensive)
    backtest_metrics: Dict = {}
    if run_backtest:
        try:
            backtest_metrics = trainer.walk_forward_backtest(
                df, X, y, n_test_steps=60, step_size=7
            ) or {}
        except Exception as exc:
            log.warning("Backtest failed for %s: %s", coin, exc)

    return PredictResponse(
        coin=coin,
        generated_at=datetime.now(timezone.utc).isoformat(),
        live_market=_build_live_market(ticker),
        indicators=_build_indicators(df),
        forecast=_build_forecast_data(forecast_result),
        validation_metrics=val_metrics,
        backtest_metrics=backtest_metrics,
    )
