# FairLens — Complete Setup & Run Guide

This guide takes you from zero to a fully running FairLens instance,
locally and on Google Cloud. Follow each section in order.

---

## Prerequisites — Install These First

| Tool | Version | Install |
|------|---------|---------|
| Git | any | https://git-scm.com |
| Python | 3.11+ | https://python.org |
| Node.js | 20+ | https://nodejs.org |
| Docker Desktop | any | https://docker.com |
| Google Cloud CLI | any | https://cloud.google.com/sdk/docs/install |
| GitHub CLI | any | https://cli.github.com |
| Terraform | 1.5+ | https://developer.hashicorp.com/terraform/install |

Check everything is installed:
```bash
git --version && python3 --version && node --version && docker --version && gcloud --version && gh --version && terraform --version
```

---

## PART 1 — Push to GitHub (5 minutes)

### 1.1  Unzip the project

```bash
unzip fairlens-repo.zip
cd fairlens
```

### 1.2  Initialise git and push

```bash
# Initialise git
git init
git add .
git commit -m "feat: FairLens initial implementation — Google Solution Challenge 2026"

# Create GitHub repo and push in one command (GitHub CLI)
gh repo create fairlens \
  --public \
  --description "AI Bias Detection & Remediation Platform — Google Solution Challenge 2026" \
  --push \
  --source=.
```

Your repo is now live at: `https://github.com/YOUR_USERNAME/fairlens`

> **If you don't have GitHub CLI:**
> 1. Go to https://github.com/new
> 2. Create a repo named `fairlens` (public, no README)
> 3. Run:
>    ```bash
>    git remote add origin https://github.com/YOUR_USERNAME/fairlens.git
>    git branch -M main
>    git push -u origin main
>    ```

---

## PART 2 — Run Locally Without GCP (10 minutes)

This lets you see the full UI and test the pipeline with mock responses.
You don't need a GCP account for this part.

### 2.1  Backend

```bash
cd backend

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create .env (mock mode — no real GCP needed)
cp .env.example .env
```

Edit `.env` and set:
```
GOOGLE_CLOUD_PROJECT=local-dev
ENVIRONMENT=development
DEBUG=true
```

Start the backend:
```bash
uvicorn app.main:app --reload --port 8000
```

You should see:
```
INFO:     Uvicorn running on http://127.0.0.1:8000
INFO:     Application startup complete.
```

Open http://localhost:8000/docs — you'll see the full interactive API docs.

### 2.2  Frontend

Open a **new terminal**:

```bash
cd frontend
npm install
npm run dev
```

You should see:
```
  VITE v5.x.x  ready in 800ms
  ➜  Local:   http://localhost:5173/
```

Open http://localhost:5173 — FairLens is running.

### 2.3  Test with the demo dataset

To test the full pipeline locally, you need a sample sklearn model and dataset.
Run this in your backend virtual environment:

```bash
python3 - << 'EOF'
import pickle, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Create synthetic hiring dataset with injected gender bias
np.random.seed(42)
n = 1000
gender   = np.random.choice(["male", "female"], n)
race     = np.random.choice(["white", "black", "hispanic", "asian"], n)
age      = np.random.randint(22, 55, n)
exp      = np.random.randint(1, 20, n)
edu      = np.random.choice(["bachelors", "masters", "phd"], n)

# Inject bias: males 35% more likely to be hired
bias = np.where(gender == "male", 0.35, 0.0)
prob = np.clip(0.25 + 0.03 * exp + bias, 0, 1)
hired = (np.random.rand(n) < prob).astype(int)

df = pd.DataFrame({
    "gender": gender, "race": race, "age": age,
    "experience": exp, "education": edu, "hired": hired
})

# Save dataset
df.to_csv("sample_dataset.csv", index=False)
print(f"Dataset saved: {len(df)} rows, {hired.mean():.1%} hired")

# Train model
le = LabelEncoder()
Xe = df.drop(columns=["hired"]).copy()
for col in ["gender", "race", "education"]:
    Xe[col] = le.fit_transform(Xe[col])

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(Xe, hired)

# Save model
with open("sample_model.pkl", "wb") as f:
    pickle.dump(clf, f)
print("Model saved: sample_model.pkl")
print("Upload these two files in the FairLens UI.")
print("Protected columns: gender,race,age")
print("Target column: hired")
EOF
```

This creates `sample_model.pkl` and `sample_dataset.csv` in your backend directory.

### 2.4  Run through the UI

1. Go to http://localhost:5173
2. Click **Launch FairLens →**
3. **Dataset source:** select **Upload CSV** and drop in `sample_dataset.csv`
4. **Model type (optional):** select **scikit-learn / XGBoost** and drop in `sample_model.pkl`
   - You can skip the model — stages 1–3 work with a dataset alone
5. Click **Detect Biases Automatically →**

FairLens auto-detects protected attributes (`gender`, `race`, `age`) and the target column (`hired`) from the dataset — no manual column configuration needed. The three analysis stages run sequentially. Once complete, you'll land on the Results page showing the bias topology map, counterfactual constitution, and proxy graph.

> **Note:** Without a real GCP project, the Counterfactual Constitution stage (which calls Gemini 2.5 Flash) will return an error. Bias Cartography and Proxy Hunter run fully offline. If a model file is provided, the Constitution stage also skips Gemini and produces a dataset-level summary instead. To enable full Gemini-powered analysis, complete Part 3 below.

---

## PART 3 — Connect to Google Cloud (20 minutes)

### 3.1  Create a GCP Project

```bash
# Log in
gcloud auth login

# Create project
gcloud projects create fairlens-gsc2026 --name="FairLens GSC2026"
gcloud config set project fairlens-gsc2026

# Link billing (replace with your billing account ID)
gcloud billing accounts list                         # find your billing account ID
gcloud billing projects link fairlens-gsc2026 \
  --billing-account=XXXXXXXX-XXXXXX-XXXXXX
```

### 3.2  Provision infrastructure with Terraform

```bash
cd infrastructure/terraform

# Create the Terraform state bucket first
gsutil mb -p fairlens-gsc2026 gs://fairlens-terraform-state

# Authenticate Terraform
gcloud auth application-default login

# Init and apply
terraform init
terraform plan  -var="project_id=fairlens-gsc2026"
terraform apply -var="project_id=fairlens-gsc2026"
# Type 'yes' when prompted
```

Terraform will provision:
- Artifact Registry (Docker images)
- Cloud Storage bucket (model uploads)
- BigQuery dataset + table (audit trail)
- Service account with correct IAM roles
- Cloud Run service skeleton

### 3.3  Store your Gemini API key

```bash
# Get a key at: https://aistudio.google.com/app/apikey
echo -n "YOUR_GEMINI_API_KEY" | \
  gcloud secrets create fairlens-gemini-key \
    --data-file=- \
    --replication-policy="automatic"
```

### 3.4  Update your local .env

```bash
cd ../../backend
```

Edit `.env`:
```
GOOGLE_CLOUD_PROJECT=fairlens-gsc2026
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
```

To get a local service account key:
```bash
gcloud iam service-accounts keys create ./service-account-key.json \
  --iam-account=fairlens-backend@fairlens-gsc2026.iam.gserviceaccount.com
```

Restart the backend — now the Constitution stage will call Gemini 2.5 Flash for real.

---

## PART 4 — Set Up GitHub Actions for Auto-Deploy

Every push to `main` will automatically build and deploy to Cloud Run.

### 4.1  Set up Workload Identity Federation (keyless auth)

```bash
export PROJECT_ID=fairlens-gsc2026
export GITHUB_USER=YOUR_GITHUB_USERNAME

# Get project number
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format='value(projectNumber)')

# Create Workload Identity Pool
gcloud iam workload-identity-pools create "github-pool" \
  --project=$PROJECT_ID \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Create provider
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project=$PROJECT_ID \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Allow GitHub repo to impersonate the service account
gcloud iam service-accounts add-iam-policy-binding \
  fairlens-backend@$PROJECT_ID.iam.gserviceaccount.com \
  --project=$PROJECT_ID \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/github-pool/attribute.repository/${GITHUB_USER}/fairlens"

# Print the provider resource name (you'll need this for secrets)
gcloud iam workload-identity-pools providers describe github-provider \
  --workload-identity-pool=github-pool \
  --location=global \
  --project=$PROJECT_ID \
  --format='value(name)'
```

### 4.2  Add GitHub Actions secrets

Go to your repo → **Settings → Secrets and variables → Actions → New repository secret**

Add these three secrets:

| Secret name | Value |
|-------------|-------|
| `GCP_PROJECT_ID` | `fairlens-gsc2026` |
| `GCP_SERVICE_ACCOUNT` | `fairlens-backend@fairlens-gsc2026.iam.gserviceaccount.com` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | (the full resource name from the command above) |

Or add them via GitHub CLI:

```bash
gh secret set GCP_PROJECT_ID        --body "fairlens-gsc2026"
gh secret set GCP_SERVICE_ACCOUNT   --body "fairlens-backend@fairlens-gsc2026.iam.gserviceaccount.com"
gh secret set GCP_WORKLOAD_IDENTITY_PROVIDER --body "projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/providers/github-provider"
```

### 4.3  Trigger deployment

```bash
git add .
git commit -m "chore: add GCP secrets, trigger first deploy"
git push origin main
```

Watch the deployment:
```bash
gh run watch   # shows live GitHub Actions progress
```

Or go to: `https://github.com/YOUR_USERNAME/fairlens/actions`

After ~3-4 minutes, you'll see:
```
✅ Deploy Backend → Cloud Run
✅ Deploy Frontend → Cloud Run
✅ FairLens deployed at: https://fairlens-frontend-xxxx-uc.a.run.app
```

---

## PART 5 — Verify Everything Works

### Check backend health

```bash
# Get backend URL
BACKEND=$(gcloud run services describe fairlens-api --region us-central1 --format='value(status.url)')
echo "Backend: $BACKEND"

# Health check
curl $BACKEND/health/
# Expected: {"status":"ok","timestamp":"..."}

# API docs
open $BACKEND/docs
```

### Check frontend

```bash
FRONTEND=$(gcloud run services describe fairlens-frontend --region us-central1 --format='value(status.url)')
echo "Frontend: $FRONTEND"
open $FRONTEND
```

### Check BigQuery audit logs

```bash
bq query --use_legacy_sql=false \
  'SELECT * FROM `fairlens-gsc2026.fairlens_audit.bias_audits` ORDER BY timestamp DESC LIMIT 5'
```

---

## PART 6 — Run a Full Audit on Cloud

1. Open your Cloud Run frontend URL
2. **Dataset source:** select **Upload CSV** and upload `sample_dataset.csv`
3. **Model type (optional):** select **scikit-learn / XGBoost** and upload `sample_model.pkl`
   - Stages 1–3 run without a model; Stage 4 (Red-Team) requires one
4. Click **Detect Biases Automatically →**
   - Protected attributes and target column are auto-detected — no manual input needed

All four stages now run on Cloud Run with:
- Bias Cartography → Gemini-powered bias analysis (dataset-only, no SHAP/UMAP required)
- Counterfactual Constitution → calls **Gemini 2.5 Flash** via Vertex AI (full counterfactuals when model provided; dataset-level analysis otherwise)
- Proxy Variable Hunter → calls **Vertex AI Embeddings** (text-embedding-004)
- Audit logs written to **BigQuery**
- Red-Team Agent → LangGraph + **Gemini 2.5 Flash**, streamed live via SSE *(model file required)*

---

## Troubleshooting

**Backend fails to start locally:**
```bash
# Check Python version (needs 3.11+)
python3 --version

# Reinstall deps
pip install -r requirements.txt --upgrade
```

**Gemini API errors:**
```bash
# Verify secret exists
gcloud secrets versions access latest --secret=fairlens-gemini-key

# Check Vertex AI API is enabled
gcloud services list --enabled | grep aiplatform
# If not: gcloud services enable aiplatform.googleapis.com
```

**Cloud Run deployment fails:**
```bash
# Check build logs
gcloud builds list --limit=5
gcloud builds log $(gcloud builds list --limit=1 --format='value(id)')
```

**GitHub Actions failing on auth:**
- Double-check `GCP_WORKLOAD_IDENTITY_PROVIDER` includes the full resource path
- Make sure `PROJECT_NUMBER` (not project ID) is in the provider path
- Re-run: `gcloud projects describe $PROJECT_ID --format='value(projectNumber)'`

**CORS errors in frontend:**
```bash
# Add your frontend URL to ALLOWED_ORIGINS in backend .env
# Or update config.py and redeploy
gcloud run services update fairlens-api \
  --set-env-vars "ALLOWED_ORIGINS=[\"https://your-frontend-url.run.app\"]" \
  --region us-central1
```

---

## Quick Reference — All URLs

| Resource | URL |
|----------|-----|
| GitHub repo | `https://github.com/YOUR_USERNAME/fairlens` |
| GitHub Actions | `https://github.com/YOUR_USERNAME/fairlens/actions` |
| Backend API docs | `https://fairlens-api-xxxx-uc.a.run.app/docs` |
| Frontend | `https://fairlens-frontend-xxxx-uc.a.run.app` |
| GCP Console | `https://console.cloud.google.com/run?project=fairlens-gsc2026` |
| BigQuery audit logs | `https://console.cloud.google.com/bigquery?project=fairlens-gsc2026` |
| Vertex AI | `https://console.cloud.google.com/vertex-ai?project=fairlens-gsc2026` |

---

*Total time from unzip to fully deployed on GCP: ~35 minutes*
