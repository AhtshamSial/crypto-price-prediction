"""
app/routers/health.py
=====================
Health and status endpoints.

/api/health now returns rich training progress so the frontend
can show a loading state while models train in the background.
"""

from fastapi import APIRouter

from app.core.trainer_state import trainer_state
from app.models.schemas import HealthResponse

router = APIRouter()

SUPPORTED_COINS = ["BTC", "ETH", "BNB", "SOL", "XRP"]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Returns backend readiness and training progress.
    
    status values:
      "starting"  - server just booted, training not yet started
      "training"  - actively training models (currently_training shows which coin)
      "ready"     - all coins trained, predictions available
    
    The frontend should poll this endpoint every 5s until status == "ready".
    """
    if not trainer_state.training_started:
        status = "starting"
    elif not trainer_state.is_ready:
        status = "training"
    else:
        status = "ready"

    return HealthResponse(
        status=status,
        trained_coins=trainer_state.trained_coins,
        supported_coins=SUPPORTED_COINS,
        currently_training=trainer_state.currently_training,
        failed_coins=trainer_state.failed_coins,
    )
