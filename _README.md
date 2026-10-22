# Used-Car Price Prediction MLOps Pipeline

Azure ML pipeline (prep → hyper-parameter sweep → register) driven by GitHub Actions.

**Start with [`RUNBOOK.md`](RUNBOOK.md)** — it lists every value to change for a new
GitHub account / Azure tenant (marked `>>> CHANGE ... <<<` in the files).

Quick start:
1. `bash scripts/validate-local.sh` — check the code works locally.
2. Edit and run `scripts/setup-azure.sh` — creates Azure resources and the `AZURE_CREDENTIALS` GitHub secret.
3. Set `resource_group` / `aml_workspace` in `config-infra-prod.yml`.
4. Push to `main` — the workflow trains, tunes and registers the model.
