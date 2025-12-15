#!/usr/bin/env python3
"""
Advanced Crypto Price Prediction & Simulation Engine - BACKEND VERSION
- Long-term horizons: 7d, 14d, 30d, 90d, 180d, 365d
- Data: CoinGecko (1-year OHLCV)
- Sentiment: NewsAPI + NewsData.io + AlphaVantage
- Models: RF, XGB, LGBM, LSTM, GRU, TCN, Transformer
- Simulation: Backtest + Monte Carlo
- Backend-ready, no input/print
"""

import os
import time
import warnings
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any

# ML & DL
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

import tensorflow as tf
try:
    from keras.models import Model, Sequential
    from keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, GlobalAveragePooling1D, Input, MultiHeadAttention, LayerNormalization
    from keras.optimizers import Adam
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
except ImportError:
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Conv1D, GlobalAveragePooling1D, Input, MultiHeadAttention, LayerNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Sentiment
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# ================= CONFIG =================
CONFIG = {
    'NEWSAPI_KEY': '9713084c35244ec9be191c384161072f',
    'NEWSDATA_KEY': 'pub_71847093312ab5efa41844aa8ebf9276ee914',
    'ALPHAVANTAGE_KEY': '8H0G7VGR9923QXSX',
    'PREDICTION_HORIZONS': [7, 14, 30, 90, 180, 365],
    'MODEL_DIR': './models',
    'CACHE_DIR': './data_cache',
    'RANDOM_STATE': 42,
    'SIMULATION': {
        'INITIAL_CAPITAL': 10000,
        'FEE_PCT': 0.001,
        'MC_PATHS': 1000,
        'MC_DAYS': 365
    }
}

os.makedirs(CONFIG['MODEL_DIR'], exist_ok=True)
os.makedirs(CONFIG['CACHE_DIR'], exist_ok=True)
np.random.seed(CONFIG['RANDOM_STATE'])
tf.random.set_seed(CONFIG['RANDOM_STATE'])

# ================= 1. DATA FETCHER =================
class CoinGeckoDataFetcher:
    BASE_URL = "https://api.coingecko.com/api/v3"

    @staticmethod
    def get_coin_id_from_symbol(symbol: str) -> Optional[str]:
        common_coins = {
            'BTC': 'bitcoin', 'ETH': 'ethereum', 'USDT': 'tether',
            'BNB': 'binancecoin', 'XRP': 'ripple', 'ADA': 'cardano',
            'DOGE': 'dogecoin', 'SOL': 'solana', 'DOT': 'polkadot',
            'MATIC': 'matic-network'
        }
        if symbol.upper() in common_coins:
            return common_coins[symbol.upper()]
        try:
            url = f"{CoinGeckoDataFetcher.BASE_URL}/coins/list"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                coins = response.json()
                for c in coins:
                    if c['symbol'].upper() == symbol.upper():
                        return c['id']
            return None
        except:
            return None

    @staticmethod
    def fetch_ohlcvs(symbol: str, days: int = 365) -> Optional[pd.DataFrame]:
        coin_id = CoinGeckoDataFetcher.get_coin_id_from_symbol(symbol)
        if not coin_id:
            return None
        cache_file = os.path.join(CONFIG['CACHE_DIR'], f"{symbol}_market_{days}d.pkl")
        if os.path.exists(cache_file):
            if time.time() - os.path.getmtime(cache_file) < 3600:
                return pd.read_pickle(cache_file)
        try:
            url = f"{CoinGeckoDataFetcher.BASE_URL}/coins/{coin_id}/market_chart"
            params = {"vs_currency": "usd", "days": days, "interval": "daily"}
            response = requests.get(url, params=params, timeout=15)
            if response.status_code != 200:
                return None
            data = response.json()
            prices = data.get("prices", [])
            volumes = data.get("total_volumes", [])
            if len(prices) < 200:
                return None
            df = pd.DataFrame(prices, columns=["timestamp", "close"])
            df["volume"] = [v[1] for v in volumes]
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
            df["open"] = df["close"].shift(1)
            df["high"] = df["close"].rolling(2).max()
            df["low"] = df["close"].rolling(2).min()
            df = df.dropna().reset_index(drop=True)
            df.to_pickle(cache_file)
            return df
        except:
            return None

# ================= 2. TECHNICAL INDICATORS =================
class TechnicalIndicators:
    @staticmethod
    def safe_division(a, b, default=0):
        result = np.where(b != 0, a / b, default)
        return pd.Series(result, index=a.index)

    @staticmethod
    def rsi(series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window).mean()
        loss = -delta.where(delta < 0, 0).rolling(window).mean()
        rs = TechnicalIndicators.safe_division(gain, loss, 0)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window=14, d_window=3) -> Tuple[pd.Series, pd.Series]:
        lowest_low = low.rolling(k_window).min()
        highest_high = high.rolling(k_window).max()
        k = 100 * TechnicalIndicators.safe_division(close - lowest_low, highest_high - lowest_low, 0)
        d = k.rolling(d_window).mean()
        return k, d

    @staticmethod
    def macd(series: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def roc(series: pd.Series, window=10) -> pd.Series:
        return series.pct_change(periods=window) * 100

    @staticmethod
    def ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def sma(series: pd.Series, window: int) -> pd.Series:
        return series.rolling(window).mean()

    @staticmethod
    def supertrend(high, low, close, period=10, multiplier=3):
        atr = TechnicalIndicators.atr(pd.DataFrame({'high': high, 'low': low, 'close': close}), period)
        hl2 = (high + low) / 2
        upper_band = hl2 + multiplier * atr
        lower_band = hl2 - multiplier * atr
        supertrend = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(1, index=close.index)
        for i in range(1, len(close)):
            if close.iloc[i] > upper_band.iloc[i-1]:
                trend.iloc[i] = 1
            elif close.iloc[i] < lower_band.iloc[i-1]:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1]
            supertrend.iloc[i] = lower_band.iloc[i] if trend.iloc[i] == 1 else upper_band.iloc[i]
        return supertrend.fillna(0)

    @staticmethod
    def atr(df, window=14):
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift()).abs()
        low_close = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        return tr.rolling(window).mean()

    @staticmethod
    def donchian_channels(high, low, window=20):
        upper = high.rolling(window).max()
        lower = low.rolling(window).min()
        middle = (upper + lower) / 2
        return upper, middle, lower

    @staticmethod
    def historical_volatility(series, window=30):
        log_ret = np.log(series / series.shift(1))
        return log_ret.rolling(window).std() * np.sqrt(365)

    @staticmethod
    def obv(close, volume):
        sign = np.sign(close.diff()).fillna(0)
        return (sign * volume).fillna(0).cumsum()

    @staticmethod
    def pivot_points(high, low, close):
        pp = (high + low + close) / 3
        r1 = 2 * pp - low
        s1 = 2 * pp - high
        return pp, r1, s1

    @staticmethod
    def candle_patterns(df):
        body = df['close'] - df['open']
        patterns = pd.DataFrame(index=df.index)
        patterns['bullish'] = (body > 0).astype(int)
        patterns['bearish'] = (body < 0).astype(int)
        range_val = df['high'] - df['low']
        patterns['doji'] = (abs(body) < range_val * 0.1).astype(int)
        return patterns

# ================= 3. SENTIMENT =================
class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    def _safe_request(self, url, timeout=10):
        try:
            response = requests.get(url, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            return {}
        except:
            return {}

    def newsapi_sentiment(self, symbol: str) -> float:
        if not CONFIG['NEWSAPI_KEY']:
            return 0.0
        url = f"https://newsapi.org/v2/everything?q={symbol}+cryptocurrency&apiKey={CONFIG['NEWSAPI_KEY']}&pageSize=10&sortBy=publishedAt"
        data = self._safe_request(url)
        scores = [TextBlob(f"{a.get('title','')} {a.get('description','')}").sentiment.polarity for a in data.get('articles', [])[:10] if a.get('title') or a.get('description')]
        return np.mean(scores) if scores else 0.0

    def newsdata_sentiment(self, symbol: str) -> float:
        if not CONFIG['NEWSDATA_KEY']:
            return 0.0
        url = f"https://newsdata.io/api/1/news?apikey={CONFIG['NEWSDATA_KEY']}&q={symbol}+crypto&language=en"
        data = self._safe_request(url)
        scores = [TextBlob(f"{a.get('title','')} {a.get('description','')}").sentiment.polarity for a in data.get('results', [])[:10] if a.get('title') or a.get('description')]
        return np.mean(scores) if scores else 0.0

    def alpha_vantage_sentiment(self, symbol: str) -> float:
        if not CONFIG['ALPHAVANTAGE_KEY']:
            return 0.0
        url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=CRYPTO:{symbol}&apikey={CONFIG['ALPHAVANTAGE_KEY']}"
        data = self._safe_request(url)
        scores = []
        for item in data.get('feed', [])[:10]:
            for sentiment in item.get('ticker_sentiment', []):
                if 'ticker_sentiment_score' in sentiment:
                    try:
                        scores.append(float(sentiment['ticker_sentiment_score']))
                    except:
                        pass
        return np.mean(scores) if scores else 0.0

    def get_combined_sentiment(self, symbol: str) -> Dict[str, float]:
        s1 = self.newsapi_sentiment(symbol)
        time.sleep(0.5)
        s2 = self.newsdata_sentiment(symbol)
        time.sleep(0.5)
        s3 = self.alpha_vantage_sentiment(symbol)
        weights = [0.4, 0.35, 0.25]
        combined = (weights[0]*s1 + weights[1]*s2 + weights[2]*s3)
        return {"polarity": combined, "subjectivity": abs(combined), "volatility": np.std([s1, s2, s3]) if any([s1,s2,s3]) else 0.0}

# ================= 4. FEATURE ENGINEER =================
class FeatureEngineer:
    def __init__(self):
        self.ti = TechnicalIndicators()

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['log_return'] = np.log(df['close']/df['close'].shift(1))
        df['rsi'] = self.ti.rsi(df['close'])
        stoch_k, stoch_d = self.ti.stochastic(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch_k
        df['stoch_d'] = stoch_d
        macd, macd_sig, macd_hist = self.ti.macd(df['close'])
        df['macd'] = macd
        df['macd_signal'] = macd_sig
        df['macd_hist'] = macd_hist
        df['roc'] = self.ti.roc(df['close'])
        for span in [10,20,50]:
            df[f'ema_{span}'] = self.ti.ema(df['close'], span)
        for window in [20,50]:
            df[f'sma_{window}'] = self.ti.sma(df['close'], window)
        df['supertrend'] = self.ti.supertrend(df['high'], df['low'], df['close'])
        df = df.dropna().reset_index(drop=True)
        return df

# ================= 5. MODEL TRAINER =================
class ModelTrainer:
    def __init__(self, horizon_days: int, symbol: str):
        self.horizon = horizon_days
        self.symbol = symbol
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        self.models = {}
        self.model_path = os.path.join(CONFIG['MODEL_DIR'], f"{symbol}_{horizon_days}d")
        
        self.is_trained = False 

    def prepare_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df[f'target_{self.horizon}'] = df['close'].shift(-self.horizon)
        return df.dropna()

    def build_lstm(self, input_dim: int) -> tf.keras.Model:
        model = Sequential([LSTM(32, return_sequences=False, input_shape=(1, input_dim)), Dropout(0.2), Dense(16, activation='relu'), Dense(1)])
        model.compile(optimizer=Adam(1e-3), loss='mse')
        return model

    def build_gru(self, input_dim: int) -> tf.keras.Model:
        model = Sequential([GRU(32, return_sequences=False, input_shape=(1, input_dim)), Dropout(0.2), Dense(16, activation='relu'), Dense(1)])
        model.compile(optimizer=Adam(1e-3), loss='mse')
        return model

    def train(self, df: pd.DataFrame):
        df = self.prepare_targets(df)

        feature_cols = [
            col for col in df.columns
            if col not in ['timestamp', f'target_{self.horizon}', 'open', 'high', 'low', 'close', 'volume']
        ]

        X = df[feature_cols].values
        y = df[f'target_{self.horizon}'].values

        if len(X) < 50:
            return  # ❌ do NOT mark trained

        # ===== SCALE =====
        X_scaled = self.scaler_X.fit_transform(X)
        y_scaled = self.scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

        # ===== ML MODEL =====
        self.models['ml'] = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbosity=0
        )
        self.models['ml'].fit(X_scaled, y_scaled)

        # ===== DL MODELS =====
        if len(X_scaled) > 100:
            X_seq = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            y_seq = y_scaled

            dl_models = {
                'lstm': self.build_lstm(X_scaled.shape[1]),
                'gru': self.build_gru(X_scaled.shape[1])
            }

            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True, verbose=0),
                ReduceLROnPlateau(factor=0.5, patience=3, verbose=0)
            ]

            for name, model in dl_models.items():
                model.fit(
                    X_seq, y_seq,
                    epochs=30,
                    batch_size=16,
                    validation_split=0.1,
                    callbacks=callbacks,
                    verbose=0
                )
                self.models[name] = model

        # ===== SAVE SCALERS =====
        pd.to_pickle(self.scaler_X, self.model_path + '_scaler_X.pkl')
        pd.to_pickle(self.scaler_y, self.model_path + '_scaler_y.pkl')

        self.is_trained = True  # ✅ VERY IMPORTANT

    def predict(self, features: np.ndarray) -> float:
        if not self.is_trained:
            raise RuntimeError("ModelTrainer not trained yet")

        X_scaled = self.scaler_X.transform(features.reshape(1, -1))

        ml_pred = self.models['ml'].predict(X_scaled)[0]
        dl_preds = []

        if 'lstm' in self.models:
            X_seq = X_scaled.reshape(1, 1, X_scaled.shape[1])
            dl_preds.append(self.models['lstm'].predict(X_seq, verbose=0)[0][0])

        if 'gru' in self.models:
            X_seq = X_scaled.reshape(1, 1, X_scaled.shape[1])
            dl_preds.append(self.models['gru'].predict(X_seq, verbose=0)[0][0])

        all_preds = [ml_pred] + dl_preds if dl_preds else [ml_pred]

        return float(
            self.scaler_y.inverse_transform(
                np.array(all_preds).reshape(-1, 1)
            ).mean()
        )

# ================= 6. MAIN PREDICTOR =================
class CryptoPredictor:
    def __init__(self):
        self.fetcher = CoinGeckoDataFetcher()
        self.sentiment = SentimentAnalyzer()
        self.feature_engineer = FeatureEngineer()

    def run(self, symbol: str) -> Dict[str, Any]:
        # Fetch OHLCV data
        df = self.fetcher.fetch_ohlcvs(symbol)
        if df is None or df.empty:
            return {"error": "Data fetch failed or insufficient data"}

        # Create technical features
        features_df = self.feature_engineer.create_features(df)

        # Get sentiment scores
        sent = self.sentiment.get_combined_sentiment(symbol)

        # Current price
        current_price = float(df['close'].iloc[-1])

        results = {}

        for horizon in CONFIG['PREDICTION_HORIZONS']:
            trainer = ModelTrainer(horizon, symbol)
            trainer.train(features_df)

            # Skip horizon if model could not be trained
            if not trainer.is_trained:
                continue

            # Extract latest features for prediction
            latest_features = (
                features_df.iloc[-1]
                .drop(['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                .values
            )

            # Predict future price
            pred = trainer.predict(latest_features)

            # Calculate percentage change and signal
            change_pct = ((pred - current_price) / current_price) * 100
            signal = "BUY" if change_pct > 2 else ("SELL" if change_pct < -2 else "HOLD")

            results[horizon] = {
                "predicted_price": float(pred),
                "change_pct": float(change_pct),
                "signal": signal
            }

        # Return error if no predictions could be made
        if not results:
            return {"error": "Model could not be trained for any horizon with current data"}

        return {
            "symbol": symbol,
            "current_price": current_price,
            "sentiment": sent,
            "predictions": results
        }
