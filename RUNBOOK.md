# Runbook: Used-Car Price Prediction MLOps Pipeline (GitHub Actions + Azure ML)

Rebuild the whole project from scratch in a **new GitHub account** and a **new
Azure tenant**. Everything here was done and verified in the original project
except where marked **(untested)**.

> **Legend** — 🔧 = a value you MUST change for the new GitHub account / Azure
> tenant. Every such place in the code is also tagged with
> `>>> CHANGE FOR NEW ... <<<` (find them all with the command in section 2).

---

## 1. What you are building

```
git push to main
   └─ GitHub Actions: deploy-model-training-pipeline
        get-config ─► create-compute ───────────────┐
                  ├─► register-dataset ─────────────┤
                  └─► register-environment ─► register-components ─► run-pipeline
                                                                          │
                                         Azure ML pipeline job            ▼
                                  prep_data ─► sweep_step (16 trials) ─► register_model
```

| Stage | Component / file | Result in Azure ML |
|---|---|---|
| Data prep | `prep.py` / `prep.yml` | train/test CSVs, row-count metrics |
| Train + tune | `train.py` / `train.yml` + sweep in `newpipeline.yml` | 16 trials; MSE, MAE, R² logged with MLflow |
| Register | `register.py` / `register.yml` | model `used_cars_price_prediction_model` v1, v2… |
| CI/CD | `.github/workflows/*.yml` | every push to `main` re-runs everything |

Layout (must be kept — YAML paths are relative):

```
.github/workflows/   5 reusable + 1 main workflow
config-infra-prod.yml  🔧 resource group + workspace names
data/used_cars.csv
data-science/src/    prep.py, train.py, register.py
data-science/environment/train-conda.yml
mlops/azureml/train/ data.yml, train-env.yml, prep.yml, train.yml, register.yml, newpipeline.yml
mlops/azureml/endpoint/  (optional) endpoint.yml, deployment.yml
scripts/             setup-azure.sh, validate-local.sh
```

---

## 2. Everything you must change (checklist)

| # | 🔧 What | Where | Original value | New value |
|---|---|---|---|---|
| 1 | Azure resource group name | `config-infra-prod.yml` → `resource_group` | `gl-pro3rg` | your new RG |
| 2 | Azure ML workspace name | `config-infra-prod.yml` → `aml_workspace` | `gl-pro3ws` | your new workspace |
| 3 | Azure subscription / tenant | `az login`, `az account set`, `scripts/setup-azure.sh` | old tenant's subscription | new subscription ID |
| 4 | GitHub owner/repo | `scripts/setup-azure.sh` → `GITHUB_REPO`, `git remote` | `pia-ntsada/caronsale` | new `owner/repo` |
| 5 | GitHub secret | repo → Settings → Secrets → Actions | `AZURE_CREDENTIALS` (old SP JSON) | **new** SP JSON, same name |
| 6 | Azure region | `scripts/setup-azure.sh` → `LOCATION` | `eastus` | a region where you have quota |
| 7 | VM size (if no quota) | main workflow `create-compute.size` | `Standard_DS11_v2` | a size you have quota for |
| 8 | Endpoint name (optional) | `endpoint.yml`, `deployment.yml` | `used-cars-price-endpoint` | a regionally unique name |

**Must stay consistent with each other (change together or not at all):**

| Value | Appears in |
|---|---|
| Compute `cpu-cluster` | main workflow `cluster_name`, `newpipeline.yml` `default_compute` |
| Dataset `used-cars-data` | `data.yml` `name`, main workflow `register-dataset.name`, `newpipeline.yml` `azureml:used-cars-data@latest` |
| Environment `used-cars-train-env` | `train-env.yml` `name`, `prep.yml`, `train.yml`, `register.yml` |
| Component names `prep_data`, `train_model`, `register_model` | the three component YAMLs, `newpipeline.yml` |
| Metric `MSE` | `train.py`, `newpipeline.yml` `primary_metric` |
| Secret `AZURE_CREDENTIALS` | the GitHub secret, every `secrets:` block in the main workflow |

Find every tagged place:

```bash
grep -rn ">>> CHANGE\|<your-\|<new-" --include="*.yml" --include="*.sh" --include="*.md" .
```

**No change needed** for `prep.py`, `train.py`, `register.py`, `custom-*.yml`
workflows (they take all inputs from the main workflow).

**Do not copy** from the old project: the old Azure subscription ID, the old
service-principal JSON, or any notebook that hard-codes a subscription ID.

---

## 3. Prerequisites

**Accounts and permissions (new tenant)**
- Azure subscription where you can create a resource group and an Azure ML workspace.
- Permission to assign roles on the resource group (**Owner** or **User Access
  Administrator** on it) AND permission in Microsoft Entra ID to create app
  registrations. Without both, `az ad sp create-for-rbac` fails; ask the tenant
  admin to create the service principal for you.
- vCPU quota for the VM size used (default `Standard_DS11_v2`, 2 vCPUs, DSv2 family).
  Check: Azure Portal → Subscriptions → Usage + quotas. New/free subscriptions
  often have 0 for some families; request an increase or pick another size.
- New GitHub account with permission to create a repository and Actions secrets.

**Tools on your machine**
- `git`, `python` 3.10+, [`az` CLI](https://learn.microsoft.com/cli/azure/install-azure-cli), [`gh` CLI](https://cli.github.com/)
- `pip install pandas scikit-learn mlflow` (for local validation only)

---

## 4. Step-by-step

### Step 1 — Get the files and validate the code locally
```bash
cd used-cars-mlops-runbook
bash scripts/validate-local.sh        # expect: model_info.json printed, "Local validation OK"
```
If MLflow complains about a file store, the script already uses SQLite; make sure
you did not set `MLFLOW_TRACKING_URI` yourself.

### Step 2 — Sign in to the NEW accounts
```bash
az login --tenant <new-tenant-id-or-domain>     # 🔧 new tenant
az account list -o table                         # note the new subscription ID
az account set --subscription "<new-subscription-id>"   # 🔧

gh auth login -h github.com -p https -s repo,workflow -w   # 🔧 sign in as the NEW GitHub user
gh auth status                                    # confirm the account shown is the new one
```
Lesson learned: the repo push failed originally because `gh` was logged in as a
different account (HTTP 403 / `push: false`). Always check `gh auth status`.
The `workflow` scope is required to push files under `.github/workflows/`.

### Step 3 — Create the new GitHub repository and push the files
```bash
gh repo create <new-owner>/<new-repo> --public --source . --remote origin   # 🔧 (or --private)
git init -b main                       # skip if already a git repo
git add . && git commit -m "Initial MLOps pipeline"
```
Do **not** push yet: the workflow would fail until Azure and the secret exist.
(If it does run early, just re-run it later.)

### Step 4 — Create the Azure resources and the service principal
**Fast path:** edit the variables at the top of `scripts/setup-azure.sh`
(🔧 subscription ID, location, resource group, workspace, `GITHUB_REPO`), then:
```bash
bash scripts/setup-azure.sh
```
It registers resource providers, creates the resource group and workspace, creates
a service principal scoped to the resource group (role Contributor), and stores the
JSON directly as the GitHub secret `AZURE_CREDENTIALS` (never printed or saved).

**Manual path** (same result):
```bash
export SUB="<new-subscription-id>" RG="<new-rg>" WS="<new-workspace>" LOC="eastus"   # 🔧
az account set --subscription "$SUB"
az extension add -n ml -y
for ns in Microsoft.MachineLearningServices Microsoft.ContainerRegistry Microsoft.Storage \
          Microsoft.KeyVault Microsoft.Insights Microsoft.OperationalInsights Microsoft.Cache; do
  az provider register --namespace $ns --wait; done
az group create -n "$RG" -l "$LOC"
az ml workspace create -n "$WS" -g "$RG" -l "$LOC"
az ad sp create-for-rbac --name used-cars-mlops-github --role contributor \
  --scopes /subscriptions/$SUB/resourceGroups/$RG --sdk-auth | gh secret set AZURE_CREDENTIALS -R <new-owner>/<new-repo>
```
(Or copy the JSON into **Settings → Secrets and variables → Actions → New
repository secret**, name `AZURE_CREDENTIALS`. Treat that JSON as a password;
never commit it or paste it into chats.)

Verify: `gh secret list -R <new-owner>/<new-repo>` shows `AZURE_CREDENTIALS`.

### Step 5 — Point the config at the new workspace
Edit `config-infra-prod.yml` 🔧:
```yaml
resource_group: <new-rg>
aml_workspace: <new-workspace>
```
Optional: if you lack quota for `Standard_DS11_v2`, change `size:` in the
`create-compute` job of the main workflow 🔧.

### Step 6 — Push and run the workflow
```bash
git add -A && git commit -m "Configure for new tenant" && git push -u origin main
gh run list -R <new-owner>/<new-repo> --limit 3
gh run watch <run-id> -R <new-owner>/<new-repo>      # or use the Actions tab
```
You can also start it manually: Actions → *deploy-model-training-pipeline* → **Run workflow**.

Expected timeline (original run): get-config seconds; create-compute ~1 min;
register-dataset/environment ~1 min; register-components ~1 min; **run-pipeline
~20–25 min** (first run adds the environment image build, ~10–15 min). All six
jobs should finish green.

### Step 7 — Verify in Azure
```bash
az ml job list -g "$RG" -w "$WS" --max-results 3 -o table
az ml component list -g "$RG" -w "$WS" -o table          # prep_data, train_model, register_model
az ml model list -n used_cars_price_prediction_model -g "$RG" -w "$WS" -o table
```
In Azure ML Studio: **Jobs → used-cars-price-experiment** → open the pipeline →
the `sweep-train` step → **Trials** (compare MSE / MAE / R²).

### Step 8 — Prove CI/CD works (the "update a script and commit" task)
Make a small change to `data-science/src/train.py` (the original project added
`MAE` and `R2` metrics next to `MSE`), then:
```bash
git add -A && git commit -m "Update train.py" && git push
```
A new workflow run starts automatically and should succeed, producing a new model
version. Screenshot the commit and the green run.

### Step 9 — (Optional, untested) deploy an online endpoint
Only needed if something must call the model over REST. Costs money while it runs.
```bash
az ml online-endpoint create   -f mlops/azureml/endpoint/endpoint.yml   -g "$RG" -w "$WS"
az ml online-deployment create -f mlops/azureml/endpoint/deployment.yml --all-traffic -g "$RG" -w "$WS"
```
Test input columns (in order): `Segment` (0=luxury, 1=non-luxury), `Kilometers_Driven`,
`Mileage`, `Engine`, `Power`, `Seats`.

### Step 10 — Clean up (avoid cost) and sign out
```bash
az ml online-endpoint delete -n used-cars-price-endpoint -g "$RG" -w "$WS" --yes   # if created
az group delete -n "$RG" --yes --no-wait        # deletes EVERYTHING in the RG
gh auth logout                                   # if this is a shared machine
```
The compute cluster scales to zero when idle, so it costs nothing at rest.

---

## 5. Mapping to the project criteria

| Criterion | Where it is satisfied |
|---|---|
| Components for preprocessing, training, tuning + MLflow logging, registration; combined in one pipeline; executed | `prep.yml`, `train.yml`, `register.yml`, sweep in `newpipeline.yml`; Step 6–7 |
| Credentials connecting GitHub Actions to Azure; workflow executed | `AZURE_CREDENTIALS` secret (Step 4); Step 6 |
| Update `prep.py` or `train.py`, commit, validate CI/CD | Step 8 |
| Business takeaways | Section 7 |

---

## 6. Troubleshooting (all hit in the original project unless marked)

| Symptom | Cause | Fix |
|---|---|---|
| `push` rejected / 403, `permissions.push=false` | `gh` logged in as the wrong account | `gh auth login` as the repo owner; check `gh auth status` |
| Push rejected for workflow files | token lacks `workflow` scope | `gh auth login -s repo,workflow` |
| Pipeline stuck in **Preparing** for > 30 min | old environment (python 3.7, old pins, `git+https` pip deps, ubuntu18.04 image) | use the provided `train-conda.yml` and `train-env.yml`; cancel the run (`gh run cancel <id>` **and** `az ml job cancel -n <job>`), fix, push |
| Local: "filesystem tracking backend is in maintenance mode" | newer MLflow rejects `./mlruns` | use SQLite URI (done in `validate-local.sh`) |
| `az ml ...` → "extension ml is not installed" | CLI extension missing | `az extension add -n ml -y` |
| `register-components` fails, component not found | environment not registered yet | the job already `needs: register-environment`; re-run |
| Job stays **Queued** | cluster has 0 nodes and is scaling up, or no quota | wait ~3–5 min; if it persists, check vCPU quota (section 3) |
| Quota / `InsufficientQuota` | new subscription has no vCPU quota for the family | request an increase or change VM `size` |
| `AuthorizationFailed` in a job | SP lacks role on RG, or wrong RG in config | confirm Contributor on the RG; confirm `config-infra-prod.yml` names |
| `Please run 'az login'` / login step fails | `AZURE_CREDENTIALS` missing, wrong name, or malformed JSON | recreate the secret (Step 4) |
| `az ad sp create-for-rbac` fails with insufficient privileges **(untested)** | no Entra permission to create apps | ask tenant admin; or use OIDC federated credentials with `azure/login` `client-id`/`tenant-id`/`subscription-id` |
| `--sdk-auth` unsupported **(untested)** | removed from newer Azure CLI | switch the `azure/login` step in each `custom-*.yml` to OIDC |
| Dataset/environment name errors | names out of sync | see "must stay consistent" table in section 2 |

---

## 7. Business takeaways (from the original data — re-run to confirm)

Prices are in **lakhs (1 lakh = 100,000)**; the data does not state a currency.
Dataset: 200 cars (86 luxury, 114 non-luxury).

- **Accuracy:** on the pipeline's 80/20 split the best trial (20 trees, depth 10)
  gave R² 0.95, MAE ≈ 4.5 lakhs. A stricter 5-fold cross-validation gave
  **R² 0.84, MAE ≈ 6.4 lakhs (~19% of price); 43% of predictions within 10% and
  65% within 20% of the real price.** Treat the model as a pricing guide, not an
  automatic price.
- **Segment dominates price:** luxury median ≈ 54.8 vs non-luxury ≈ 10.2 lakhs.
  Luxury costs ~3× more per BHP (0.34 vs 0.12 lakhs).
- **Drivers:** engine size and power matter most (correlation 0.67 / 0.79).
  Within non-luxury, specs explain price well (engine 0.86); within luxury they
  explain much less (power 0.44) → brand/model/age are missing.
- **Kilometers driven and seats have little effect** in this data, which signals
  missing variables (manufacturing year, brand, condition, owners).
- **Data quality:** no nulls/duplicates, but 3 cars have mileage 0 (impossible).
- **Recommendations:** have appraisers review predictions (especially luxury);
  collect year/brand/condition; add range checks in `prep.py`; consider separate
  luxury and non-luxury models; add an accuracy gate before registering a model;
  monitor against actual sales and retrain (the pipeline already automates it);
  deploy an endpoint only once validated; delete idle endpoints.
- **Caveats:** small sample, single split in the pipeline, correlations are not causation.

---

## 8. Security notes
- Never commit the service-principal JSON, subscription IDs, or tokens.
- Scope the service principal to the single resource group (as done here).
- Rotate or delete the service principal when the project ends:
  `az ad sp delete --id <appId>`.
- If you used `gh auth login` on a shared machine, run `gh auth logout` afterwards.
