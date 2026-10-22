"""Pipeline step 1 - data preparation (component: ``prep_data``).

What it does
------------
1. Reads the raw used-car CSV (``--raw_data``).
2. Validates that the expected columns exist and contain no missing values.
3. Label-encodes the categorical ``Segment`` column
   (``luxury segment`` -> 0, ``non-luxury segment`` -> 1, because
   ``LabelEncoder`` sorts the labels alphabetically).
4. Splits the data into train/test sets with a fixed seed (reproducible).
5. Writes ``train.csv`` and ``test.csv`` into the two output folders that Azure ML
   mounts for the job (``--train_data`` / ``--test_data``).
6. Logs the row counts to MLflow so they appear in Azure ML Studio.

Run by: ``mlops/azureml/train/prep.yml`` (registered as component ``prep_data``).

Nothing in this file is tenant- or account-specific, so it does not need to change
when you move to a new GitHub account or Azure tenant. Change it only if your
dataset's columns differ.
"""

import argparse
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Columns the rest of the pipeline relies on. If your CSV uses different names,
# update this set AND the column list documented in train.py.
REQUIRED_COLUMNS = {
    "Segment",
    "Kilometers_Driven",
    "Mileage",
    "Engine",
    "Power",
    "Seats",
    "price",
}


def parse_args():
    """Parse the command-line arguments that Azure ML passes to the job."""
    parser = argparse.ArgumentParser("prep")
    parser.add_argument("--raw_data", type=str, required=True, help="Path to raw data")
    parser.add_argument("--train_data", type=str, required=True, help="Path to train dataset")
    parser.add_argument("--test_data", type=str, required=True, help="Path to test dataset")
    parser.add_argument(
        "--test_train_ratio",
        type=float,
        default=0.2,
        help="Test-train ratio",
    )
    return parser.parse_args()


def main(args):
    """Validate, encode, split, and save the datasets."""
    if not 0 < args.test_train_ratio < 1:
        raise ValueError("--test_train_ratio must be greater than 0 and less than 1")

    data = pd.read_csv(args.raw_data)

    # Fail early with a clear message instead of a confusing error later on.
    missing_columns = REQUIRED_COLUMNS.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Raw data is missing required columns: {sorted(missing_columns)}")
    if data[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Raw data contains missing values in required columns")

    # Convert the text category into numbers so the model can use it.
    data = data.copy()
    data["Segment"] = LabelEncoder().fit_transform(data["Segment"])

    # random_state=42 keeps the split identical on every run.
    train_data, test_data = train_test_split(
        data, test_size=args.test_train_ratio, random_state=42
    )

    # The output paths are folders managed by Azure ML (type: uri_folder).
    train_path = Path(args.train_data)
    test_path = Path(args.test_data)
    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_path / "train.csv", index=False)
    test_data.to_csv(test_path / "test.csv", index=False)

    mlflow.log_metrics(
        {"train_row_count": len(train_data), "test_row_count": len(test_data)}
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"Raw data path: {args.raw_data}")
    print(f"Train dataset output path: {args.train_data}")
    print(f"Test dataset path: {args.test_data}")
    print(f"Test-train ratio: {args.test_train_ratio}")
    with mlflow.start_run():
        main(args)
