"""
train_model.py

Trains a classifier that predicts: given a "failed" transaction's features,
is it likely a PHANTOM failure (money actually went through) or a TRUE failure?

We use a Random Forest -- it's simple, trains in seconds, handles mixed
categorical + numeric features well, and (importantly for your pitch video)
you can show *feature importance*, i.e. "the model learned that gateway
timeouts and slow response times are the biggest predictors of phantom
failures" -- which sounds smart and is also just true.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

DATA_PATH = "data/transactions.csv"
MODEL_PATH = "models/phantom_model.pkl"
ENCODERS_PATH = "models/encoders.pkl"


def train():
    df = pd.read_csv(DATA_PATH)

    # Encode categorical columns (bank, decline_code, payment_method) into numbers,
    # since the model needs numeric input. We save these encoders so the live
    # app can use the exact same encoding later.
    encoders = {}
    for col in ["bank", "decline_code", "payment_method"]:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le

    feature_cols = ["amount", "response_time_ms", "bank_enc", "decline_code_enc", "payment_method_enc"]
    X = df[feature_cols]
    y = df["is_phantom_failure"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"Test accuracy: {acc*100:.1f}%\n")
    print(classification_report(y_test, preds, target_names=["True Failure", "Phantom Failure"]))

    print("Feature importance (what the model actually pays attention to):")
    for name, importance in sorted(zip(feature_cols, model.feature_importances_), key=lambda x: -x[1]):
        print(f"  {name:20s} {importance:.3f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoders, ENCODERS_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")
    print(f"Saved encoders -> {ENCODERS_PATH}")


if __name__ == "__main__":
    train()
