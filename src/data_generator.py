"""
data_generator.py

Generates a SYNTHETIC dataset of "failed" payment transactions.

Why synthetic? In a real hackathon you won't have access to Razorpay's actual
bank ledger / settlement data. So we simulate it: for every failed transaction,
we invent a "ground truth" label of whether the money actually went through
later (a phantom failure) or truly failed.

The features we generate are the same signals a real system WOULD have access
to (decline code, response time, bank, payment method, amount) -- we're just
faking the labels using realistic probability rules instead of real ledger data.
"""

import random
import csv
import os

random.seed(42)

# Decline/failure codes and their real-world tendency to be phantom (recoverable)
# vs true failures. These probabilities are based on how these errors behave in
# real payment systems (network/timeout issues are usually phantom; validation
# issues like insufficient funds or wrong CVV are usually genuinely failed).
DECLINE_CODES = {
    "GATEWAY_TIMEOUT":       0.85,  # very likely phantom
    "NETWORK_ERROR":         0.80,
    "BANK_SERVER_ERROR":     0.75,
    "WEBHOOK_DELAY":         0.90,
    "INSUFFICIENT_FUNDS":    0.05,  # very unlikely phantom (money genuinely wasn't there)
    "CARD_BLOCKED":          0.02,
    "INVALID_CVV":           0.01,
    "INVALID_OTP":           0.03,
    "CARD_EXPIRED":          0.01,
    "USER_CANCELLED":        0.01,
}

BANKS = ["HDFC", "SBI", "ICICI", "AXIS", "KOTAK", "PNB", "YES_BANK"]

# Some banks are known (in this simulation) to have flakier webhook delivery,
# which increases the phantom probability slightly.
BANK_FLAKINESS = {
    "HDFC": 0.02, "SBI": 0.08, "ICICI": 0.03, "AXIS": 0.04,
    "KOTAK": 0.05, "PNB": 0.10, "YES_BANK": 0.12,
}

PAYMENT_METHODS = ["UPI", "CARD", "NETBANKING", "WALLET"]


def generate_transaction(txn_id):
    decline_code = random.choices(
        list(DECLINE_CODES.keys()),
        weights=[1] * len(DECLINE_CODES),  # equally likely to occur, for variety
    )[0]

    base_phantom_prob = DECLINE_CODES[decline_code]
    bank = random.choice(BANKS)
    bank_boost = BANK_FLAKINESS[bank]

    # Response time: phantom failures tend to happen when the response takes
    # unusually long (near or past typical gateway timeout windows, ~15-30s)
    if base_phantom_prob > 0.5:
        response_time_ms = random.randint(12000, 35000)  # slow response
    else:
        response_time_ms = random.randint(200, 5000)     # fast, clean rejection

    # final probability this txn is a phantom failure (capped at 0.97)
    phantom_prob = min(0.97, base_phantom_prob + bank_boost)

    is_phantom = random.random() < phantom_prob  # this is our "ground truth" label

    amount = round(random.uniform(20, 5000), 2)
    payment_method = random.choice(PAYMENT_METHODS)

    return {
        "txn_id": txn_id,
        "amount": amount,
        "payment_method": payment_method,
        "bank": bank,
        "decline_code": decline_code,
        "response_time_ms": response_time_ms,
        "is_phantom_failure": int(is_phantom),  # 1 = money actually went through, 0 = true failure
    }


def generate_dataset(n=3000, out_path="data/transactions.csv"):
    rows = [generate_transaction(f"txn_{i:06d}") for i in range(n)]

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    phantom_count = sum(r["is_phantom_failure"] for r in rows)
    print(f"Generated {n} transactions -> {out_path}")
    print(f"Phantom failures: {phantom_count} ({phantom_count/n*100:.1f}%)")
    print(f"True failures:    {n - phantom_count} ({(n-phantom_count)/n*100:.1f}%)")


if __name__ == "__main__":
    generate_dataset()
