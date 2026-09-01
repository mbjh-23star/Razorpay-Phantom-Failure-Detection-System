# Phantom Failure Detector
**AI Revenue Recovery: Razorpay Buildathon**

## The problem

Sometimes a payment shows **"failed"** on the customer's screen, but the bank
ledger later confirms the money **did** go through. This is a *phantom failure*:
a gap between what the gateway shows in the moment and what actually happened
at the bank.

Two real examples that inspired this project:
- An online booking payment showed "failed" on the customer's side, while the
  money was already debited and the merchant's system showed "order confirmed."
- A vending machine payment showed "failed," the money was debited, but the
  order didn't dispense. the transaction was completed but had to go under "my orders." to fetch items instead of being redirected to the fetch page.

This costs everyone:
- **Customers** get anxious, think they've been scammed, or retry and get
  double-charged.
- **Merchants** lose the sale anyway, because most customers just walk away
  when they see "failed", they don't go digging through order history.
- **Razorpay** loses trust and recoverable revenue that a smarter system
  could have saved.

## The idea

A classifier that looks at a "failed" transaction's metadata- decline code,
how long the gateway took to respond, which bank, payment method, amount and predicts whether it's likely a **phantom failure** (hold and auto-confirm)
or a **true failure** (safe to let the customer retry).

This is deliberately built the way real reconciliation/fraud systems are
built in production: a small, explainable model (Random Forest) over
structured features, not a black-box deep model. You can point at *why* it
made a call- e.g. "decline code = gateway timeout + response time = 24s"
is the classic phantom-failure signature.

## How it works

```
data/data_generator.py   -> generates a synthetic dataset of failed
                             transactions with a "ground truth" label
                             (phantom vs true failure), mimicking the
                             signals a real system would eventually see
                             from a settlement/ledger file.

src/train_model.py        -> trains a Random Forest classifier on that
                             dataset, saves the model + label encoders.

app.py + templates/       -> a small Flask web app: enter a transaction's
                             details, get a live phantom/true failure
                             verdict and a recommended action.
```

**Note on data:** this uses a synthetic dataset, since real Razorpay ledger/
settlement data wasn't available for this project. The features and their
relationships (e.g. gateway timeouts → likely phantom; insufficient funds →
likely true failure) are modeled on how these failures behave in real
payment systems.

## Running it locally

```bash
pip install -r requirements.txt

# 1. Generate the synthetic dataset
python3 src/data_generator.py

# 2. Train the model
python3 src/train_model.py

# 3. Run the demo app
python3 app.py
```

Then open **http://localhost:5000** and try a transaction. For example:
- Bank: SBI, Decline code: `GATEWAY_TIMEOUT`, Response time: `24000` ms →
  flagged as a likely phantom failure.
- Bank: HDFC, Decline code: `INSUFFICIENT_FUNDS`, Response time: `800` ms →
  flagged as a true failure.

## Results

- ~86% test accuracy distinguishing phantom vs true failures on the
  synthetic dataset.
- The model's top learned feature is **response time**, followed by
  **decline code** — matching the real-world intuition that phantom
  failures come from slow/delayed confirmations, not from clean rejections.

## What this could become (with real data)

- Plugged into an actual settlement/webhook feed, so "phantom" predictions
  get auto-verified within minutes instead of guessed.
- A customer-facing message layer: "Don't retry- we're confirming your
  payment now," instead of leaving the customer to guess.
- A merchant-side auto-confirm trigger, so orders aren't lost over a
  false "failed" status.

## Why this matters for revenue recovery

Every phantom failure that goes unflagged is a transaction that already
succeeded but gets treated as lost, either through customer abandonment,
duplicate charges, or manual support overhead. Catching these earlier
recovers real revenue without needing a single additional sale.
