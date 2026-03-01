# Crypto Price Forecaster — FastAPI Backend

## Quick Start (Development)

```bash
# 1. Clone / unzip into a folder
cd fastapi_backend/

# 2. Create virtual env
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy env file and fill in your keys
cp .env.example .env

# 5. Ensure your CSV files are in the data/ folder
#    BTC.csv, ETH.csv, BNB.csv, SOL.csv, XRP.csv

# 6. Run
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts training all 5 coin models on startup.
**First run takes 1–5 minutes** depending on your machine (models are cached after that).

API docs: http://localhost:8000/docs

---

## Environment Variables (.env)

| Variable           | Description                               | Default            |
|--------------------|-------------------------------------------|--------------------|
| DATA_DIR           | Folder with CSV files + ML modules        | ./data             |
| COINGECKO_API_KEY  | CoinGecko API key                         | (empty)            |
| CRYPTOPANIC_TOKEN  | CryptoPanic token                         | (empty)            |
| NEWSAPI_KEY        | NewsAPI key                               | (empty)            |
| ALLOWED_ORIGINS    | Comma-separated CORS origins              | localhost:5173     |
| WINDOW             | Lookback window for features              | 30                 |
| EPOCHS             | Training epochs for deep models           | 50                 |
| OPTUNA_TRIALS      | Bayesian HP search trials                 | 15                 |
| ENABLE_PROPHET     | Enable Prophet model                      | true               |
| ENABLE_SARIMA      | Enable SARIMA model                       | true               |
| ENABLE_LSTM        | Enable LSTM model                         | true               |
| ENABLE_TRANSFORMER | Enable Transformer model                  | true               |

---

## API Endpoints

### GET /api/health
Returns backend status and list of trained coins.

```json
{
  "status": "ready",
  "trained_coins": ["BTC", "ETH", "BNB", "SOL", "XRP"],
  "supported_coins": ["BTC", "ETH", "BNB", "SOL", "XRP"]
}
```

### POST /api/predict
Body: `{ "coin": "BTC", "run_backtest": false }`

Returns live market data, technical indicators, and multi-horizon price forecast.

```json
{
  "coin": "BTC",
  "generated_at": "2025-01-01T12:00:00+00:00",
  "live_market": {
    "price": 95000.0,
    "price_change_pct_24h": 2.3,
    "volume_24h": 28000000000.0,
    "bid": 94995.0,
    "ask": 95005.0,
    "spread": 10.0,
    "timestamp": "2025-01-01T12:00:00Z"
  },
  "indicators": {
    "rsi_14": 54.32,
    "macd": 120.45,
    "sma_20": 92000.0
  },
  "forecast": {
    "horizons": ["1w", "1m", "3m", "6m", "1y"],
    "current_price": 95000.0,
    "ensemble": [96500.0, 102000.0, 115000.0, 130000.0, 160000.0],
    "ensemble_lower": [92000.0, 94000.0, 100000.0, 110000.0, 130000.0],
    "ensemble_upper": [101000.0, 110000.0, 130000.0, 150000.0, 190000.0],
    "prophet": [...],
    "sarima":  [...],
    "lstm":    [...],
    "transformer": [...]
  }
}
```

---

## Project Structure

```
fastapi_backend/
├── main.py                          # FastAPI app + lifespan startup
├── requirements.txt
├── .env.example
├── data/
│   ├── BTC.csv                      # Historical OHLCV data
│   ├── ETH.csv
│   ├── BNB.csv
│   ├── SOL.csv
│   ├── XRP.csv
│   ├── data_pipeline.py             # DataManager, feature engineering
│   └── model_training.py            # ModelTrainer, ForecastResult
└── app/
    ├── core/
    │   ├── config.py                # Settings from env vars
    │   └── trainer_state.py         # Singleton model registry
    ├── routers/
    │   ├── health.py
    │   └── prediction.py
    ├── services/
    │   └── prediction_service.py    # Forecast business logic
    └── models/
        └── schemas.py               # Pydantic request/response models
```
