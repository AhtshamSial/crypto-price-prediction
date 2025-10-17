# backend/trading_engine.py
import numpy as np
import pandas as pd
import ccxt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
# Note: transformers pipeline is imported if you use it for advanced NLP
# from transformers import pipeline
import requests
from textblob import TextBlob
import time
from pytrends.request import TrendReq
import warnings
import matplotlib.pyplot as plt
from scipy.stats import norm

warnings.filterwarnings('ignore')

# ========== Configuration ==========
# IMPORTANT: Replace API keys with environment variables in production.
EXCHANGE = ccxt.binance({
    'apiKey': 'RKVR7UHxYVOP6dXlrQukYnAlKmdCgVDE1SRyeaSH1VVanrWx87mLPt7NmeCQIMYr',  # replace or load from env
    'secret': 'Cfp1iOncFUpjWujV51hSTRl8QeVn4e9aYux17wY87DW80DSDZqQYM8lZHC1tP9Z2',  # replace or load from env
    'enableRateLimit': True,
    'options': {'defaultType': 'swap'}   # Use swap for futures trading
})
CRYPTO_PANIC_API = "b381e0ea5ba2395e197795ff599087744667eb52"  # replace or load from env
NEWSAPI_KEY = "9713084c35244ec9be191c384161072f"  # replace or load from env

MAX_DRAWDOWN_LIMIT = 0.2
RISK_PER_TRADE = 0.01
CONFIDENCE_THRESHOLD = 0.7

# ========== Feature Engineering & Data Processing ==========
def calculate_ema(series, window):
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=14): 
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df, window=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def calculate_bollinger_bands(df, window=20, num_std=2):
    sma = df['close'].rolling(window).mean()
    std = df['close'].rolling(window).std()
    return sma + (std * num_std), sma - (std * num_std)

def calculate_obv(df):
    obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    return obv

def calculate_macd(df, fast=12, slow=26, signal=9):
    ema_fast = calculate_ema(df['close'], fast)
    ema_slow = calculate_ema(df['close'], slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    macd_hist = macd_line - signal_line
    return macd_line, signal_line, macd_hist

def calculate_stochastic_oscillator(df, window=14):
    low_min = df['low'].rolling(window).min()
    high_max = df['high'].rolling(window).max()
    stoch_k = 100 * (df['close'] - low_min) / (high_max - low_min)
    stoch_d = stoch_k.rolling(3).mean()
    return stoch_k, stoch_d

def calculate_adx(df, window=14):
    plus_dm = df['high'].diff().clip(lower=0)
    minus_dm = -df['low'].diff().clip(lower=0)
    tr = pd.concat([df['high'] - df['low'], 
                    np.abs(df['high'] - df['close'].shift()), 
                    np.abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    plus_di = 100 * (plus_dm.rolling(window).mean() / atr)
    minus_di = 100 * (minus_dm.rolling(window).mean() / atr)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window).mean()
    return adx

def calculate_vwap(df):
    df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
    return df['vwap']

def calculate_cmf(df, window=20):
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0) * df['volume']
    cmf = mfv.rolling(window).sum() / df['volume'].rolling(window).sum()
    return cmf

def detect_candlestick_patterns(df):
    df['doji'] = (abs(df['open'] - df['close']) <= 0.1 * (df['high'] - df['low'])).astype(int)
    df['engulfing'] = ((df['close'] > df['open']) & (df['close'].shift() < df['open'].shift())).astype(int)
    df['morning_star'] = ((df['close'].shift(2) < df['open'].shift(2)) & 
                            (df['close'].shift(1) < df['open'].shift(1)) & 
                            (df['close'] > df['open'])).astype(int)
    return df

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['ema_20'] = calculate_ema(df['close'], 20)
    df['ema_50'] = calculate_ema(df['close'], 50)
    df['atr'] = calculate_atr(df)
    upper_bb, lower_bb = calculate_bollinger_bands(df)
    df['bb_width'] = (upper_bb - lower_bb) / df['close']
    df['rsi'] = calculate_rsi(df['close'])
    macd_line, signal_line, macd_hist = calculate_macd(df)
    df['macd_line'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_hist
    stoch_k, stoch_d = calculate_stochastic_oscillator(df)
    df['stoch_k'] = stoch_k
    df['stoch_d'] = stoch_d
    df['adx'] = calculate_adx(df)
    df['obv'] = calculate_obv(df)
    df['vwap'] = calculate_vwap(df)
    df['cmf'] = calculate_cmf(df)
    df = detect_candlestick_patterns(df)
    for lag in [1, 3, 5]:
        df[f'ret_lag_{lag}'] = df['log_ret'].shift(lag)
        df[f'volume_lag_{lag}'] = np.log(df['volume'].shift(lag))
    return df.dropna()

def prepare_data_for_training(df: pd.DataFrame):
    features = create_features(df)
    y = (features['close'].shift(-3) > features['close']).astype(int).shift(3).dropna()
    X = features.drop(columns=['open', 'high', 'low', 'close', 'volume'])
    X = X.loc[y.index]
    selector = SelectKBest(score_func=f_classif, k=20)
    X_selected = selector.fit_transform(X, y)
    return pd.DataFrame(X_selected, index=X.index), y

# Sentiment analysis helpers
def get_reddit_sentiment(coin: str) -> float:
    try:
        return np.random.uniform(-1, 1)
    except Exception as e:
        print(f"Reddit sentiment error: {str(e)}")
        return 0.0

def get_news_sentiment(coin: str) -> float:
    try:
        url = f"https://newsapi.org/v2/everything?q={coin}&apiKey={NEWSAPI_KEY}"
        response = requests.get(url)
        articles = response.json().get("articles", [])
        sentiments = []
        for article in articles[:5]:
            text = article.get("title", "") + " " + article.get("description", "")
            blob = TextBlob(text)
            sentiments.append(blob.sentiment.polarity)
        return float(np.mean(sentiments)) if sentiments else 0.0
    except Exception as e:
        print(f"News sentiment error: {str(e)}")
        return 0.0

def get_google_trends_interest(coin: str) -> float:
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload([coin], cat=0, timeframe='today 1-m', geo='', gprop='')
        interest_over_time = pytrends.interest_over_time()
        if not interest_over_time.empty:
            return float(interest_over_time[coin].mean() / 100)
        return 0.0
    except Exception as e:
        print(f"Google Trends error: {str(e)}")
        return 0.0
    finally:
        time.sleep(1)

def get_cryptopanic_sentiment(coin: str) -> float:
    try:
        url = f"https://cryptopanic.com/api/v1/posts/?auth_token={CRYPTO_PANIC_API}&currencies={coin}"
        response = requests.get(url)
        posts = response.json().get("results", [])
        sentiments = []
        for post in posts[:5]:
            title = post.get("title", "")
            blob = TextBlob(title)
            sentiments.append(blob.sentiment.polarity)
        return float(np.mean(sentiments)) if sentiments else 0.0
    except Exception as e:
        print(f"CryptoPanic sentiment error: {str(e)}")
        return 0.0

def get_combined_sentiment(coin: str) -> float:
    reddit_sentiment = get_reddit_sentiment(coin)
    news_sentiment = get_news_sentiment(coin)
    google_trends = get_google_trends_interest(coin)
    cryptopanic_sentiment = get_cryptopanic_sentiment(coin)
    combined_sentiment = (
        0.3 * reddit_sentiment +
        0.3 * news_sentiment +
        0.2 * google_trends +
        0.2 * cryptopanic_sentiment
    )
    return float(combined_sentiment)

# Ensemble + Backtester classes (as provided)
class EnsembleModel:
    def __init__(self):
        self.gbc = GradientBoostingClassifier(n_estimators=500, learning_rate=0.05, max_depth=5, subsample=0.8)
        self.rf = RandomForestClassifier(n_estimators=100, max_depth=10)
        self.lstm_model = None

    def train(self, X: pd.DataFrame, y: pd.Series):
        print("Training models...")
        param_grid = {'n_estimators': [100], 'max_depth': [5], 'learning_rate': [0.05]}
        gbc = GradientBoostingClassifier()
        grid_search = GridSearchCV(gbc, param_grid, cv=TimeSeriesSplit(n_splits=3), scoring='accuracy')
        grid_search.fit(X, y)
        self.gbc = grid_search.best_estimator_
        self.rf.fit(X, y)
        X_lstm = np.array(X).reshape((X.shape[0], 1, X.shape[1]))
        y_lstm = np.array(y)
        self.lstm_model = Sequential([
            LSTM(50, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
            Dense(1, activation='sigmoid')
        ])
        self.lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        self.lstm_model.fit(X_lstm, y_lstm, epochs=50, batch_size=32, validation_split=0.2, callbacks=[early_stopping], verbose=0)
        print("Models trained.")

    def predict(self, X: pd.DataFrame):
        X = np.array(X)
        gbc_proba = self.gbc.predict_proba(X)[:, 1]
        rf_proba = self.rf.predict_proba(X)[:, 1]
        X_lstm = X.reshape((X.shape[0], 1, X.shape[1]))
        lstm_proba = self.lstm_model.predict(X_lstm).flatten()
        final_proba = 0.4 * gbc_proba + 0.3 * rf_proba + 0.3 * lstm_proba
        return (final_proba > CONFIDENCE_THRESHOLD).astype(int), final_proba

class Backtester:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital
        self.portfolio_value = initial_capital
        self.positions = {}
        self.trades = []
        self.max_drawdown = 0
        self.cumulative_returns = []
        self.prediction_history = []

    def execute_trade(self, symbol, direction, price, atr, confidence):
        stop_loss_pct = 1.5 * atr
        take_profit_pct = 3 * atr
        position_size = (self.portfolio_value * RISK_PER_TRADE) / (stop_loss_pct if stop_loss_pct != 0 else 1)
        stop_loss = price - stop_loss_pct if direction == 'LONG' else price + stop_loss_pct
        take_profit = price + take_profit_pct if direction == 'LONG' else price - take_profit_pct
        self.positions[symbol] = {
            'direction': direction,
            'entry_price': price,
            'position_size': position_size,
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }
        self.trades.append({'symbol': symbol, 'direction': direction, 'entry_price': price, 'position_size': position_size, 'confidence': float(confidence)})

    def update_portfolio(self, symbol, current_price):
        if symbol in self.positions:
            position = self.positions[symbol]
            direction = position['direction']
            entry_price = position['entry_price']
            position_size = position['position_size']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            if (direction == 'LONG' and current_price <= stop_loss) or (direction == 'SHORT' and current_price >= stop_loss):
                pnl = position_size * (current_price - entry_price) if direction == 'LONG' else position_size * (entry_price - current_price)
                self.portfolio_value += pnl
                del self.positions[symbol]
                self.prediction_history.append(False)
                return False
            elif (direction == 'LONG' and current_price >= take_profit) or (direction == 'SHORT' and current_price <= take_profit):
                pnl = position_size * (current_price - entry_price) if direction == 'LONG' else position_size * (entry_price - current_price)
                self.portfolio_value += pnl
                del self.positions[symbol]
                self.prediction_history.append(True)
                return True
            return None

    def calculate_metrics(self):
        returns = np.diff(self.cumulative_returns) if len(self.cumulative_returns) > 1 else np.array([0.0])
        sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) != 0 else 0
        sortino_ratio = np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252) if np.std(returns[returns < 0]) != 0 else 0
        win_rate = np.mean(self.prediction_history) if len(self.prediction_history) > 0 else 0
        max_drawdown = self.max_drawdown
        cumulative_returns = (self.portfolio_value - self.initial_capital) / self.initial_capital
        return {'cumulative_returns': cumulative_returns, 'sharpe_ratio': sharpe_ratio, 'sortino_ratio': sortino_ratio, 'max_drawdown': max_drawdown, 'win_rate': win_rate}

class TradingEngine:
    def __init__(self):
        self.backtester = Backtester(initial_capital=10000)
        self.model = EnsembleModel()
        self.last_trained = None
        self.prediction_count = 0

    def run_cycle(self, symbol: str, investment: float):
        try:
            timeframe = '1h'
            limit = 500
            ohlcv = EXCHANGE.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = df[col].astype(float)
            X, y = prepare_data_for_training(df)
            # Train ensemble model if first run or retrain condition
            if not hasattr(self.model, 'gbc') or self.should_retrain():
                # NOTE: training can be slow — for a demo you might want to train once offline and save the models
                self.model.train(X, y)
                self.last_trained = pd.Timestamp.now()
            latest_data = X.iloc[-1:]
            prediction, confidence = self.model.predict(latest_data)
            current_price = float(df['close'].iloc[-1])
            atr = float(calculate_atr(df).iloc[-1])
            sentiment = float(get_combined_sentiment(symbol.split('/')[0]))
            leverage = int(self.suggest_leverage(atr, sentiment, investment))
            stop_loss_pct = 1.5 * atr
            take_profit_pct = 3 * atr
            stop_loss = current_price - stop_loss_pct if prediction == 1 else current_price + stop_loss_pct
            take_profit = current_price + take_profit_pct if prediction == 1 else current_price - take_profit_pct
            direction = "LONG" if int(prediction) == 1 else "SHORT"
            # Simulate execution & update backtester
            self.backtester.execute_trade(symbol, direction, current_price, atr, float(confidence[0] if hasattr(confidence, '__iter__') else confidence))
            outcome = self.backtester.update_portfolio(symbol, current_price)
            return {
                'symbol': symbol,
                'price': current_price,
                'direction': direction,
                'confidence': float(confidence[0]) if hasattr(confidence, '__iter__') else float(confidence),
                'sentiment': sentiment,
                'atr': atr,
                'stop_loss': float(stop_loss),
                'take_profit': float(take_profit),
                'leverage': leverage,
                'timestamp': pd.Timestamp.now().isoformat()
            }
        except Exception as e:
            print(f"Error during trading cycle for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'price': None,
                'direction': None,
                'confidence': None,
                'sentiment': None,
                'atr': None,
                'stop_loss': None,
                'take_profit': None,
                'leverage': None,
                'timestamp': pd.Timestamp.now().isoformat()
            }

    def should_retrain(self) -> bool:
        if not self.last_trained:
            return True
        if (pd.Timestamp.now() - self.last_trained).total_seconds() > 21600:
            return True
        if len(self.backtester.prediction_history) >= 10:
            recent_accuracy = np.mean(self.backtester.prediction_history[-10:])
            if recent_accuracy < 0.6:
                print("Recent prediction accuracy is low. Triggering retraining...")
                return True
        return False

    def suggest_leverage(self, atr, sentiment, investment):
        base_leverage = 10
        if atr > 10:
            base_leverage -= 2
        if sentiment > 0.5:
            base_leverage += 2
        return max(1, min(base_leverage, 20))

# Expose shared engine and helper function
engine = TradingEngine()

def get_prediction(symbol="BTC/USDT", investment=100):
    """Call this from the Flask app. Returns a dict ready for JSON response."""
    return engine.run_cycle(symbol, investment)
