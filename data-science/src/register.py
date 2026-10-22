"""Pipeline step 3 - model registration (component: ``register_model``).

What it does
------------
1. Loads the best model chosen by the sweep step (``--model_path``).
2. Logs it to the current MLflow run and registers it in the Azure ML workspace's
   model registry under ``--model_name``. Each run creates a new version
   (v1, v2, ...).
3. Writes ``model_info.json`` (name, version, URI) to ``--model_info_output_path``
   so later steps or people can see exactly what was registered.

The registered model name is set in ``mlops/azureml/train/newpipeline.yml``
(``register_model.inputs.model_name``). It is a free choice and is not tied to
any tenant, but you can rename it there.

Nothing in this file is tenant- or account-specific.
"""

import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn


def parse_args():
    """Parse the command-line arguments that Azure ML passes to the job."""
    parser = argparse.ArgumentParser("register")
    parser.add_argument("--model_name", type=str, required=True, help="Registered model name")
    parser.add_argument("--model_path", type=str, required=True, help="Model directory")
    parser.add_argument(
        "--model_info_output_path",
        type=str,
        required=True,
        help="Directory for model registration metadata",
    )
    return parser.parse_args()


def main(args):
    """Load, log, register, and describe the selected model."""
    print(f"Registering {args.model_name}")
    model = mlflow.sklearn.load_model(args.model_path)

    # Log first so the registered version points at an artifact owned by this run.
    logged_model = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="random_forest_price_regressor",
    )
    registered_model = mlflow.register_model(
        model_uri=logged_model.model_uri,
        name=args.model_name,
    )

    output_path = Path(args.model_info_output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    (output_path / "model_info.json").write_text(
        json.dumps(
            {
                "model_name": registered_model.name,
                "model_version": registered_model.version,
                "model_uri": logged_model.model_uri,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    args = parse_args()
    print(f"Model name: {args.model_name}")
    print(f"Model path: {args.model_path}")
    print(f"Model info output path: {args.model_info_output_path}")
    with mlflow.start_run():
        main(args)
