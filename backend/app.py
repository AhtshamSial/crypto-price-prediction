# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from trading_engine import get_prediction, engine

app = Flask(__name__)
CORS(app)  # allow frontend calls

@app.route('/')
def home():
    return jsonify({"status": "Crypto AI backend running"})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    symbol = data.get('symbol', 'BTC/USDT')
    investment = float(data.get('investment', 100))
    # call the engine (may take time if model trains). For demo, will run straight.
    result = get_prediction(symbol, investment)
    return jsonify(result)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
