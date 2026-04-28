# FairLens — MVP

Bias detection for AI systems. Enter a HuggingFace model and/or dataset, get a bias report.

**Live MVP:** https://pallavikailas.github.io/FairLens/
**Full Prototype:** https://fairlens-frontend-nrk2z2yadq-uc.a.run.app

---

## What it does

1. **Model Probe** — loads your HuggingFace model against a neutral reference dataset to surface intrinsic biases across demographic slices (gender, race, age, etc.)
2. **Dataset Analysis** — scans your HuggingFace dataset for structural bias: demographic imbalances, proxy chains (ZIP code → race, job title → gender), and representation gaps
3. **Results** — bias cartography map, hotspot list, proxy chains, and compliance flags (80% Rule, Equal Opportunity)

Analysis is powered by **Gemini 2.5 Flash** running on Google Cloud.

---

## Try it

| Input | Example values |
|---|---|
| HuggingFace Model | `unitary/toxic-bert` |
| HuggingFace Token | `hf_...` (required for gated models) |
| HuggingFace Dataset | `mstz/adult` or `LabHC/bias_in_bios` |

At least one of model or dataset is required. Whichever is missing, that phase is skipped automatically.

**Good end-to-end test:** Model → `unitary/toxic-bert` · Dataset → `LabHC/bias_in_bios`

---

## Flow

```
HuggingFace Model ID  ──┐
HuggingFace Token     ──┤──▶ Phase 1: Model Probe ──┐
                        │                            │
HuggingFace Dataset  ───┘──▶ Phase 2: Dataset Probe─┴──▶ Results
```

---

## Run locally

```bash
# Backend
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000

# Frontend (new terminal)
cd frontend
npm install
VITE_API_BASE_URL=http://localhost:8000 npm run dev
```

Open http://localhost:5173/FairLens/

---

## Deployment

This branch auto-deploys to **GitHub Pages** on every push via `.github/workflows/deploy-mvp.yml`.

To enable: repo → **Settings → Pages → Source → GitHub Actions**.

The `main` branch deploys the full prototype to Cloud Run independently — the two are completely separate deployments.
