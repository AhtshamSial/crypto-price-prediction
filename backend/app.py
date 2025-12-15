# backend/app.py
from flask import Flask, request, jsonify
from flask_cors import CORS
from crypto_engine import CryptoPredictor  # Your full backend code with ML/DL models

app = Flask(__name__)

# Allow CORS for frontend running on localhost
CORS(app, resources={r"/*": {"origins": ["http://127.0.0.1:5173", "http://localhost:5173","https://crypto-price-prediction-ten.vercel.app/"]}})

# Initialize predictor once
predictor = CryptoPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": True, "message": "No input data provided"}), 400

        symbol = data.get("symbol", "BTC").upper()

        # Run your backend predictor
        result = predictor.run(symbol)

        # Validate result
        if not result or "predictions" not in result:
            return jsonify({
                "error": True,
                "message": "Prediction engine failed to generate results."
            }), 500

        return jsonify({
            "error": False,
            "symbol": symbol,
            "current_price": result.get("current_price"),
            "predictions": result.get("predictions"),
            "sentiment": result.get("sentiment", {})
        })

    except Exception as e:
        print(f"[Server Error] {str(e)}")
        return jsonify({
            "error": True,
            "message": f"Server error: {str(e)}"
        }), 500


if __name__ == "__main__":
    # Run Flask backend on port 5000
    app.run(host="127.0.0.1", port=5000, debug=True)
