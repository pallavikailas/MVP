terraform {
  required_version = ">= 1.5"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
  backend "gcs" {
    bucket = "fairlens-terraform-state"
    prefix = "terraform/state"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

variable "project_id" { type = string }
variable "region"     { default = "us-central1" }

# ── Enable APIs ──────────────────────────────────────────────────────────────

resource "google_project_service" "apis" {
  for_each = toset([
    "run.googleapis.com",
    "artifactregistry.googleapis.com",
    "bigquery.googleapis.com",
    "storage.googleapis.com",
    "aiplatform.googleapis.com",
    "cloudbuild.googleapis.com",
    "secretmanager.googleapis.com",
    "iam.googleapis.com",
  ])
  service            = each.key
  disable_on_destroy = false
}

# ── Artifact Registry ────────────────────────────────────────────────────────

resource "google_artifact_registry_repository" "fairlens" {
  location      = var.region
  repository_id = "fairlens"
  format        = "DOCKER"
  depends_on    = [google_project_service.apis]
}

# ── Cloud Storage bucket for model uploads ───────────────────────────────────

resource "google_storage_bucket" "models" {
  name          = "${var.project_id}-fairlens-models"
  location      = "US"
  force_destroy = false

  uniform_bucket_level_access = true

  lifecycle_rule {
    action { type = "Delete" }
    condition { age = 90 }
  }
}

# ── BigQuery dataset for audit trail ─────────────────────────────────────────

resource "google_bigquery_dataset" "fairlens_audit" {
  dataset_id  = "fairlens_audit"
  location    = "US"
  description = "FairLens bias audit logs and decision history"

  depends_on = [google_project_service.apis]
}

resource "google_bigquery_table" "bias_audits" {
  dataset_id = google_bigquery_dataset.fairlens_audit.dataset_id
  table_id   = "bias_audits"

  schema = jsonencode([
    { name = "audit_id",             type = "STRING",    mode = "REQUIRED" },
    { name = "stage",                type = "STRING",    mode = "NULLABLE" },
    { name = "hotspot_count",        type = "INTEGER",   mode = "NULLABLE" },
    { name = "flagged_slice_count",  type = "INTEGER",   mode = "NULLABLE" },
    { name = "timestamp",            type = "TIMESTAMP", mode = "REQUIRED" },
  ])

  deletion_protection = false
}

# ── Service Account for backend ──────────────────────────────────────────────

resource "google_service_account" "fairlens_backend" {
  account_id   = "fairlens-backend"
  display_name = "FairLens Backend Service Account"
}

resource "google_project_iam_member" "backend_roles" {
  for_each = toset([
    "roles/bigquery.dataEditor",
    "roles/storage.objectAdmin",
    "roles/aiplatform.user",
    "roles/secretmanager.secretAccessor",
  ])
  project = var.project_id
  role    = each.key
  member  = "serviceAccount:${google_service_account.fairlens_backend.email}"
}

# ── Cloud Run backend ────────────────────────────────────────────────────────

resource "google_cloud_run_v2_service" "backend" {
  name     = "fairlens-api"
  location = var.region

  template {
    service_account = google_service_account.fairlens_backend.email
    timeout         = "600s"

    scaling {
      min_instance_count = 0
      max_instance_count = 10
    }

    containers {
      image = "${var.region}-docker.pkg.dev/${var.project_id}/fairlens/backend:latest"

      resources {
        limits = { cpu = "2", memory = "4Gi" }
      }

      env {
        name  = "GOOGLE_CLOUD_PROJECT"
        value = var.project_id
      }
      env {
        name  = "ENVIRONMENT"
        value = "production"
      }
      env {
        name  = "GCS_BUCKET_NAME"
        value = google_storage_bucket.models.name
      }
    }
  }

  depends_on = [google_project_service.apis]
}

resource "google_cloud_run_v2_service_iam_member" "backend_public" {
  project  = var.project_id
  location = var.region
  name     = google_cloud_run_v2_service.backend.name
  role     = "roles/run.invoker"
  member   = "allUsers"
}

# ── Outputs ──────────────────────────────────────────────────────────────────

output "backend_url" {
  value = google_cloud_run_v2_service.backend.uri
}
output "artifact_registry" {
  value = google_artifact_registry_repository.fairlens.name
}
output "models_bucket" {
  value = google_storage_bucket.models.name
}
