"""Pipeline step 2 - model training (component: ``train_model``).

What it does
------------
1. Reads ``train.csv`` and ``test.csv`` produced by ``prep.py``.
2. Trains a ``RandomForestRegressor`` that predicts ``price`` (in lakhs,
   1 lakh = 100,000) from the other columns.
3. Evaluates on the test set and logs to MLflow:
   * parameters: ``n_estimators``, ``max_depth``
   * metrics:    ``MSE`` (used by the sweep to pick the best trial),
                 ``MAE`` and ``R2`` (extra diagnostics)
4. Saves the model in MLflow format to ``--model_output``.

How it is used
--------------
``mlops/azureml/train/newpipeline.yml`` runs this component inside a *sweep* job.
The sweep launches many trials with different ``n_estimators`` / ``max_depth`` and
keeps the trial with the lowest ``MSE``. The metric name ``MSE`` here must match
``objective.primary_metric`` in ``newpipeline.yml``.

Nothing in this file is tenant- or account-specific, so it does not need to change
when you move to a new GitHub account or Azure tenant.
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def parse_args():
    """Parse the command-line arguments that Azure ML passes to the job."""
    parser = argparse.ArgumentParser("train")
    parser.add_argument("--train_data", type=str, required=True, help="Training data directory")
    parser.add_argument("--test_data", type=str, required=True, help="Testing data directory")
    parser.add_argument("--model_output", type=str, required=True, help="Model output directory")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees")
    parser.add_argument("--max_depth", type=int, default=10, help="Maximum tree depth")
    return parser.parse_args()


def read_dataset(path, filename):
    """Read a named CSV from an Azure ML folder input, failing clearly if absent."""
    csv_path = Path(path) / filename
    if not csv_path.is_file():
        raise FileNotFoundError(f"Expected dataset file was not found: {csv_path}")
    return pd.read_csv(csv_path)


def main(args):
    """Fit, evaluate, log, and save the price prediction model."""
    train_data = read_dataset(args.train_data, "train.csv")
    test_data = read_dataset(args.test_data, "test.csv")

    target_column = "price"
    if target_column not in train_data or target_column not in test_data:
        raise ValueError(f"Both datasets must contain the '{target_column}' target column")

    # Features (X) are every column except the target (y).
    x_train, y_train = train_data.drop(columns=target_column), train_data[target_column]
    x_test, y_test = test_data.drop(columns=target_column), test_data[target_column]

    model = RandomForestRegressor(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    # Inside an Azure ML job these calls are recorded against the job automatically.
    mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
    mlflow.log_metrics({"MSE": mse, "MAE": mae, "R2": r2})

    model_path = Path(args.model_output)
    model_path.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(model, model_path)


if __name__ == "__main__":
    args = parse_args()
    print(f"Train dataset input path: {args.train_data}")
    print(f"Test dataset input path: {args.test_data}")
    print(f"Model output path: {args.model_output}")
    print(f"Number of estimators: {args.n_estimators}")
    print(f"Max depth: {args.max_depth}")
    with mlflow.start_run():
        main(args)
