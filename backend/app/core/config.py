"""
app/core/config.py
==================
Central configuration loaded from environment variables / .env file.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load .env if present
load_dotenv(Path(__file__).parent.parent.parent / ".env")


class Settings:
    # ── API Keys ────────────────────────────────────────────────────────────
    COINGECKO_API_KEY:    str = os.getenv("COINGECKO_API_KEY", "")
    CRYPTOPANIC_TOKEN:    str = os.getenv("CRYPTOPANIC_TOKEN", "")
    NEWSAPI_KEY:          str = os.getenv("NEWSAPI_KEY", "")

    # ── Model settings ──────────────────────────────────────────────────────
    WINDOW:          int  = int(os.getenv("WINDOW", "30"))
    EPOCHS:          int  = int(os.getenv("EPOCHS", "50"))
    OPTUNA_TRIALS:   int  = int(os.getenv("OPTUNA_TRIALS", "15"))
    RUN_OPTUNA:      bool = os.getenv("RUN_OPTUNA", "true").lower() == "true"
    ENABLE_PROPHET:  bool = os.getenv("ENABLE_PROPHET", "true").lower() == "true"
    ENABLE_SARIMA:   bool = os.getenv("ENABLE_SARIMA", "true").lower() == "true"
    ENABLE_LSTM:     bool = os.getenv("ENABLE_LSTM", "true").lower() == "true"
    ENABLE_TRANSFORMER: bool = os.getenv("ENABLE_TRANSFORMER", "true").lower() == "true"

    # ── Storage ─────────────────────────────────────────────────────────────
    # Path to the folder containing BTC.csv, ETH.csv, etc.
    DATA_DIR:        str  = os.getenv("DATA_DIR", str(Path(__file__).parent.parent.parent / "data"))
    CACHE_DIR:       str  = os.getenv("CACHE_DIR", ".cache")
    MODEL_CACHE_DIR: str  = os.getenv("MODEL_CACHE_DIR", ".model_cache")

    # ── CORS ────────────────────────────────────────────────────────────────
    # Comma-separated list in env, e.g. "http://localhost:5173,https://myapp.vercel.app"
    ALLOWED_ORIGINS: List[str] = [
        o.strip()
        for o in os.getenv(
            "ALLOWED_ORIGINS",
            "http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173"
        ).split(",")
        if o.strip()
    ]


settings = Settings()
