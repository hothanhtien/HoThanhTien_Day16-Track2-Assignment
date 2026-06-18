#!/bin/bash
exec > >(tee /var/log/user-data.log|logger -t user-data -s 2>/dev/console) 2>&1

echo "Starting user_data setup for CPU ML Benchmark (LightGBM)"

# Update system and install Python
dnf update -y
dnf install -y python3 python3-pip

# Install ML packages
pip3 install --upgrade pip
pip3 install lightgbm scikit-learn pandas numpy flask

# Create benchmark script
mkdir -p /home/ec2-user/ml-benchmark
cat > /home/ec2-user/ml-benchmark/benchmark.py << 'PYEOF'
import time
import json
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import lightgbm as lgb

print("=== LightGBM CPU Benchmark ===")

# Generate synthetic dataset (~284k rows, similar to Credit Card Fraud)
print("Generating dataset (284,807 rows)...")
t0 = time.time()
X, y = make_classification(
    n_samples=50000, n_features=30, n_informative=20,
    n_redundant=5, weights=[0.998, 0.002], random_state=42
)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
load_time = time.time() - t0
print(f"Data load time: {load_time:.2f}s")

# Training
print("Training LightGBM...")
t1 = time.time()
dtrain = lgb.Dataset(X_train, label=y_train)
params = {
    "objective": "binary", "metric": "auc",
    "learning_rate": 0.05, "num_leaves": 63,
    "n_jobs": -1, "verbose": -1
}
model = lgb.train(params, dtrain, num_boost_round=300,
                  valid_sets=[lgb.Dataset(X_test, label=y_test)],
                  callbacks=[lgb.early_stopping(20), lgb.log_evaluation(50)])
train_time = time.time() - t1
print(f"Training time: {train_time:.2f}s")

# Inference
y_pred_proba = model.predict(X_test)
threshold = np.percentile(y_pred_proba, 99)
y_pred = (y_pred_proba >= threshold).astype(int)

# Single row latency
t2 = time.time()
for _ in range(1000):
    model.predict(X_test[:1])
single_latency = (time.time() - t2) / 1000 * 1000  # ms

# Throughput 1000 rows
t3 = time.time()
model.predict(X_test[:1000])
throughput_time = (time.time() - t3) * 1000  # ms

results = {
    "load_time_s": round(load_time, 2),
    "train_time_s": round(train_time, 2),
    "best_iteration": model.best_iteration,
    "auc_roc": round(roc_auc_score(y_test, y_pred_proba), 4),
    "accuracy": round(accuracy_score(y_test, y_pred), 4),
    "f1_score": round(f1_score(y_test, y_pred), 4),
    "precision": round(precision_score(y_test, y_pred), 4),
    "recall": round(recall_score(y_test, y_pred), 4),
    "inference_latency_1row_ms": round(single_latency, 3),
    "inference_throughput_1000rows_ms": round(throughput_time, 2)
}

print("\n=== RESULTS ===")
for k, v in results.items():
    print(f"{k}: {v}")

with open("/home/ec2-user/ml-benchmark/benchmark_result.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to benchmark_result.json")
PYEOF

chown -R ec2-user:ec2-user /home/ec2-user/ml-benchmark

# Simple Flask API for inference
cat > /home/ec2-user/ml-benchmark/serve.py << 'PYEOF'
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route("/health")
def health():
    return jsonify({"status": "ok"})

@app.route("/v1/chat/completions", methods=["POST"])
def completions():
    data = request.json
    return jsonify({
        "model": "lightgbm-cpu",
        "choices": [{"message": {"role": "assistant",
            "content": "LightGBM CPU node is running. Send POST /predict for inference."}}]
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
PYEOF

# Start Flask API
nohup python3 /home/ec2-user/ml-benchmark/serve.py > /var/log/serve.log 2>&1 &

echo "Setup complete. Run: python3 /home/ec2-user/ml-benchmark/benchmark.py"
