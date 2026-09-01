"""
app.py

Flask web app for the Phantom Failure Detector demo.

Two things it does:
1. Serves a dashboard (index.html) where you can pick transaction details
   and see live whether the model thinks it's a phantom failure or a true failure.
2. Serves a small JSON API (/predict) that the frontend calls.

Run with: python3 app.py
Then open: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import os

app = Flask(__name__)

MODEL_PATH = "models/phantom_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"

model = None
encoders = None


def load_model():
    global model, encoders
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "Model not found. Run: python3 src/data_generator.py && python3 src/train_model.py"
        )
    model = joblib.load(MODEL_PATH)
    encoders = joblib.load(ENCODERS_PATH)


@app.route("/")
def index():
    return render_template(
        "index.html",
        banks=list(encoders["bank"].classes_),
        decline_codes=list(encoders["decline_code"].classes_),
        payment_methods=list(encoders["payment_method"].classes_),
    )


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    try:
        bank_enc = encoders["bank"].transform([data["bank"]])[0]
        decline_enc = encoders["decline_code"].transform([data["decline_code"]])[0]
        method_enc = encoders["payment_method"].transform([data["payment_method"]])[0]
    except ValueError as e:
        return jsonify({"error": f"Unknown category: {e}"}), 400

    row = pd.DataFrame([{
        "amount": float(data["amount"]),
        "response_time_ms": float(data["response_time_ms"]),
        "bank_enc": bank_enc,
        "decline_code_enc": decline_enc,
        "payment_method_enc": method_enc,
    }])

    proba = model.predict_proba(row)[0]  # [prob_true_failure, prob_phantom]
    is_phantom = bool(model.predict(row)[0])
    phantom_confidence = float(proba[1])

    if is_phantom:
        recommendation = "Hold. Do not ask customer to retry — likely to double-charge. Auto-confirm order once ledger settles."
    else:
        recommendation = "Safe to let customer retry payment. Transaction genuinely did not go through."

    return jsonify({
        "is_phantom_failure": is_phantom,
        "phantom_confidence": round(phantom_confidence * 100, 1),
        "recommendation": recommendation,
    })


if __name__ == "__main__":
    load_model()
    app.run(host="0.0.0.0", port=5000, debug=True)
