# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from trading_engine import get_prediction

app = Flask(__name__)

# Allow both localhost and 127.0.0.1 frontend URLs
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5173", "http://localhost:5173"]}})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": True, "message": "No input data provided"}), 400

        symbol = data.get("symbol", "BTC/USDT")
        investment = float(data.get("investment", 100))

        result = get_prediction(symbol, investment)

        # Validate backend response
        if not result or "price" not in result or result["price"] is None:
            return jsonify({
                "error": True,
                "message": "Trading engine failed to generate prediction."
            }), 500

        return jsonify(result)

    except Exception as e:
        print(f"[Server Error] {str(e)}")
        return jsonify({
            "error": True,
            "message": f"Server error: {str(e)}"
        }), 500


if __name__ == "__main__":
    # Run on 127.0.0.1:5000 so frontend can access it
    app.run(host="127.0.0.1", port=5000, debug=True)
