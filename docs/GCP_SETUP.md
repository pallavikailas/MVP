# GCP Setup Guide for FairLens

## Prerequisites
- Google Cloud account with billing enabled
- `gcloud` CLI installed
- `terraform` >= 1.5 installed
- Docker installed

---

## Step 1: Create GCP Project

```bash
gcloud projects create fairlens-gsc2026 --name="FairLens GSC2026"
gcloud config set project fairlens-gsc2026
gcloud billing projects link fairlens-gsc2026 --billing-account=YOUR_BILLING_ACCOUNT_ID
```

---

## Step 2: Provision Infrastructure with Terraform

```bash
cd infrastructure/terraform

# Initialise Terraform (creates GCS backend bucket first)
gsutil mb -p fairlens-gsc2026 gs://fairlens-terraform-state

terraform init
terraform plan -var="project_id=fairlens-gsc2026"
terraform apply -var="project_id=fairlens-gsc2026"
```

This provisions:
- Artifact Registry repository
- Cloud Storage bucket for model uploads
- BigQuery dataset + tables for audit trail
- Service account with correct IAM roles
- Cloud Run service (backend)

---

## Step 3: Configure GitHub Actions Secrets

In your GitHub repository → Settings → Secrets → Actions, add:

| Secret | Value |
|--------|-------|
| `GCP_PROJECT_ID` | `fairlens-gsc2026` |
| `GCP_SERVICE_ACCOUNT` | `fairlens-backend@fairlens-gsc2026.iam.gserviceaccount.com` |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | From Terraform output or manual setup below |

### Set up Workload Identity Federation (keyless auth):

```bash
# Create Workload Identity Pool
gcloud iam workload-identity-pools create "github-pool" \
  --project="fairlens-gsc2026" \
  --location="global" \
  --display-name="GitHub Actions Pool"

# Create provider
gcloud iam workload-identity-pools providers create-oidc "github-provider" \
  --project="fairlens-gsc2026" \
  --location="global" \
  --workload-identity-pool="github-pool" \
  --display-name="GitHub Actions Provider" \
  --attribute-mapping="google.subject=assertion.sub,attribute.repository=assertion.repository" \
  --issuer-uri="https://token.actions.githubusercontent.com"

# Bind service account to GitHub repo
gcloud iam service-accounts add-iam-policy-binding \
  fairlens-backend@fairlens-gsc2026.iam.gserviceaccount.com \
  --project="fairlens-gsc2026" \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/PROJECT_NUMBER/locations/global/workloadIdentityPools/github-pool/attribute.repository/YOUR_GITHUB_USERNAME/fairlens"
```

---

## Step 4: Store Gemini API Key in Secret Manager

```bash
echo -n "YOUR_GEMINI_API_KEY" | \
  gcloud secrets create fairlens-gemini-key \
    --data-file=- \
    --replication-policy="automatic"
```

---

## Step 5: First Deployment

Push to `main` branch — GitHub Actions will:
1. Run CI (lint + test)
2. Build Docker images
3. Push to Artifact Registry
4. Deploy backend to Cloud Run
5. Build frontend with backend URL
6. Deploy frontend to Cloud Run

Or deploy manually:

```bash
# Backend
cd backend
gcloud run deploy fairlens-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --set-env-vars GOOGLE_CLOUD_PROJECT=fairlens-gsc2026

# Get backend URL
BACKEND_URL=$(gcloud run services describe fairlens-api --region us-central1 --format='value(status.url)')

# Frontend
cd ../frontend
VITE_API_BASE_URL=$BACKEND_URL npm run build
gcloud run deploy fairlens-frontend \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

---

## BigQuery Schema Verification

```sql
-- Check audit logs are flowing
SELECT * FROM `fairlens-gsc2026.fairlens_audit.bias_audits`
ORDER BY timestamp DESC
LIMIT 10;
```

---

## Vertex AI Model Access

The backend uses Gemini 1.5 Pro via Vertex AI. Ensure the service account has:
- `roles/aiplatform.user`

And the Vertex AI API is enabled:
```bash
gcloud services enable aiplatform.googleapis.com
```

---

## Cost Estimate (per hackathon demo day)

| Service | Est. Cost |
|---------|-----------|
| Cloud Run (backend) | ~$0.50/day (0 requests when idle) |
| Cloud Run (frontend) | ~$0.10/day |
| Vertex AI (Gemini 1.5 Pro) | ~$0.02-0.10 per audit |
| BigQuery | ~$0.00 (free tier) |
| Cloud Storage | ~$0.02/month |
| **Total demo day** | **< $5** |
