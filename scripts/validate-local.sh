#!/usr/bin/env bash
# Runs prep -> train -> register on your own machine, with no Azure needed.
# Use it to confirm the Python code works BEFORE pushing to GitHub.
# Requires: python 3.10+, pip install pandas scikit-learn mlflow
set -euo pipefail
cd "$(dirname "$0")/.."

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
# A throw-away SQLite database stands in for the Azure ML MLflow tracking server.
export MLFLOW_TRACKING_URI="sqlite:///$TMP/mlflow.db"

python data-science/src/prep.py --raw_data data/used_cars.csv \
  --train_data "$TMP/train" --test_data "$TMP/test"
python data-science/src/train.py --train_data "$TMP/train" --test_data "$TMP/test" \
  --model_output "$TMP/model" --n_estimators 10 --max_depth 5
python data-science/src/register.py --model_name used_cars_price_prediction_model \
  --model_path "$TMP/model" --model_info_output_path "$TMP/info"

cat "$TMP/info/model_info.json"
find . -name __pycache__ -type d -prune -exec rm -rf {} +
echo "Local validation OK"
