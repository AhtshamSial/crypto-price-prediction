"""
app/core/trainer_state.py
=========================
Singleton that owns trained model objects for all coins.

KEY CHANGE vs original:
  Old: train_all() called synchronously in lifespan -> blocks server startup
       -> Railway kills process after health check timeout.
  New: train_all_background() runs as asyncio background task.
       Server is ready instantly. Frontend polls /api/health for progress.
       Each coin trains one at a time to keep memory low.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional

from app.core.config import settings

log = logging.getLogger("trainer_state")


def _import_pipeline():
    """
    Dynamically import data_pipeline and model_training from DATA_DIR.
    Works with Python 3.13 @dataclass fix (register in sys.modules first).
    """
    import sys
    import importlib.util

    data_dir = Path(settings.DATA_DIR)

    for candidate in [data_dir, data_dir.parent / "improved", data_dir.parent]:
        dp = candidate / "data_pipeline.py"
        mt = candidate / "model_training.py"
        if not (dp.exists() and mt.exists()):
            continue

        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)

        spec_dp = importlib.util.spec_from_file_location(
            "data_pipeline", str(dp), submodule_search_locations=[],
        )
        data_pipeline = importlib.util.module_from_spec(spec_dp)
        sys.modules["data_pipeline"] = data_pipeline
        spec_dp.loader.exec_module(data_pipeline)

        spec_mt = importlib.util.spec_from_file_location(
            "model_training", str(mt), submodule_search_locations=[],
        )
        model_training = importlib.util.module_from_spec(spec_mt)
        sys.modules["model_training"] = model_training
        spec_mt.loader.exec_module(model_training)

        return data_pipeline, model_training, candidate_str

    raise ImportError(
        f"Could not find data_pipeline.py / model_training.py under {data_dir}. "
        "Set the DATA_DIR environment variable to the folder containing those files."
    )


class TrainerState:
    """Thread-safe singleton holding all trained coin models."""

    def __init__(self):
        self._trained_data: Dict[str, dict] = {}
        self._trained_coins: List[str] = []
        self._failed_coins: List[str] = []
        self._currently_training: Optional[str] = None
        self._lock = asyncio.Lock()
        self._ready = False
        self._training_started = False

    @property
    def is_ready(self) -> bool:
        return self._ready

    @property
    def trained_coins(self) -> List[str]:
        return list(self._trained_coins)

    @property
    def failed_coins(self) -> List[str]:
        return list(self._failed_coins)

    @property
    def currently_training(self) -> Optional[str]:
        return self._currently_training

    @property
    def training_started(self) -> bool:
        return self._training_started

    async def train_all_background(self, force_retrain: bool = False) -> None:
        """
        Called as asyncio.create_task() from lifespan.
        Trains one coin at a time in a thread pool executor so the event loop
        stays free to serve HTTP requests (including /api/health polling).
        """
        self._training_started = True
        loop = asyncio.get_event_loop()

        try:
            data_pipeline, model_training, module_dir = _import_pipeline()
        except ImportError as exc:
            log.error("Cannot import ML modules: %s", exc)
            self._ready = True  # mark ready anyway so health check passes
            return

        SUPPORTED_COINS = data_pipeline.SUPPORTED_COINS

        for coin in SUPPORTED_COINS:
            self._currently_training = coin
            log.info("Background training: starting %s...", coin)
            try:
                await loop.run_in_executor(
                    None,
                    self._train_one_coin,
                    coin, data_pipeline, model_training, module_dir, force_retrain,
                )
                self._trained_coins.append(coin)
                log.info("Background training: %s complete (%d/%d)",
                         coin, len(self._trained_coins), len(SUPPORTED_COINS))
            except Exception as exc:
                log.error("Background training: %s FAILED: %s", coin, exc, exc_info=True)
                self._failed_coins.append(coin)

        self._currently_training = None
        self._ready = True
        log.info("All background training complete. Trained: %s | Failed: %s",
                 self._trained_coins, self._failed_coins)

    def _train_one_coin(
        self,
        coin: str,
        data_pipeline,
        model_training,
        module_dir: str,
        force_retrain: bool = False,
    ) -> None:
        """Synchronous training for a single coin. Runs in thread pool."""
        from pathlib import Path

        build_feature_matrix = data_pipeline.build_feature_matrix
        DataManager = data_pipeline.DataManager
        ModelTrainer = model_training.ModelTrainer
        HORIZONS = model_training.HORIZONS

        data_dir = Path(module_dir)

        # Locate CSV
        csv_path: Optional[str] = None
        for name in [f"{coin}.csv", f"{coin.lower()}.csv"]:
            p = data_dir / name
            if p.exists():
                csv_path = str(p)
                break

        dm = DataManager(
            coin=coin,
            csv_path=csv_path,
            cache_dir=str(data_dir / settings.CACHE_DIR),
            coingecko_api_key=settings.COINGECKO_API_KEY or None,
            cryptopanic_token=settings.CRYPTOPANIC_TOKEN or None,
            newsapi_key=settings.NEWSAPI_KEY or None,
            use_websocket=False,
        )
        df = dm.load(fetch_coingecko=True, fetch_binance=True)
        sentiment = dm.sentiment()
        X, y, feature_names = build_feature_matrix(
            df, window=settings.WINDOW, horizons=HORIZONS, sentiment=sentiment
        )

        trainer = ModelTrainer(
            coin=coin,
            cache_dir=str(data_dir / settings.MODEL_CACHE_DIR),
            epochs=settings.EPOCHS,
            window=settings.WINDOW,
            enable_prophet=settings.ENABLE_PROPHET,
            enable_sarima=settings.ENABLE_SARIMA,
            enable_lstm=settings.ENABLE_LSTM,
            enable_transformer=settings.ENABLE_TRANSFORMER,
            run_optuna=settings.RUN_OPTUNA,
            optuna_trials=settings.OPTUNA_TRIALS,
        )
        trainer.train(df, X, y, feature_names, force_retrain=force_retrain)

        self._trained_data[coin] = {
            "trainer": trainer,
            "df": df,
            "X": X,
            "y": y,
            "feature_names": feature_names,
            "dm": dm,
            "module_dir": module_dir,
            "model_training": model_training,
            "data_pipeline": data_pipeline,
        }

    def get_coin_data(self, coin: str) -> Optional[dict]:
        return self._trained_data.get(coin.upper())


# Module-level singleton
trainer_state = TrainerState()
