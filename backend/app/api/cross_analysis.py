"""
Cross-Analysis API
==================
Phase 3 — group model biases (Phase 1) + dataset biases (Phase 2) and run
all three analysis stages (cartography + constitution + proxy) on the combined
model × user-dataset view to surface interaction biases.

Also runs the cross_analyzer service to synthesize aligned, proxy-amplification,
and blind-spot findings into a single risk matrix.
"""

import io
import json
import logging
import pickle
import uuid
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.api._utils import resolve_feature_cols
from app.services.auto_detect import auto_detect_columns
from app.services.cartography import cartography_service
from app.services.compliance_mapper import check_compliance
from app.services.constitution import constitution_service
from app.services.cross_analyzer import cross_analyzer_service
from app.services.dataset_loader import load_dataset_csv
from app.services.model_adapter import FairLensAdapter
from app.services.proxy_hunter import ProxyVariableHunter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/run")
async def run_cross_analysis(
    # Phase 1 + 2 results (JSON strings) for synthesis
    model_probe_results:   str = Form(..., description="JSON of Phase 1 model-probe results"),
    dataset_probe_results: str = Form(..., description="JSON of Phase 2 dataset-probe results"),
    # Model (needed to run cartography + constitution on user dataset)
    model_file:    Optional[UploadFile] = File(default=None),
    model_type:    str  = Form(default="sklearn"),
    api_endpoint:  str  = Form(default=""),
    llm_api_key:   str  = Form(default=""),
    hf_token:      str  = Form(default=""),
    # Dataset
    dataset_file:   Optional[UploadFile] = File(default=None),
    dataset_source: str  = Form(default="upload"),
    dataset_url:    str  = Form(default=""),
    # Columns (can be inferred from dataset_probe_results)
    protected_cols: str  = Form(default="auto"),
    target_col:     str  = Form(default="auto"),
):
    """
    Phase 3 cross-analysis:
      a) Re-run cartography + constitution + proxy on (model × user dataset)
      b) Synthesize Phase 1 + Phase 2 findings into aligned/proxy-amp/blind-spot report
    """
    audit_id = str(uuid.uuid4())[:8]

    try:
        model_probe   = json.loads(model_probe_results)
        dataset_probe = json.loads(dataset_probe_results)
    except Exception as e:
        raise HTTPException(400, f"Invalid JSON in probe results: {e}")

    # ── Infer columns from dataset probe when not provided ────────────────────
    if protected_cols in ("auto", "", "['auto']") or "auto" in protected_cols.split(","):
        protected = dataset_probe.get("detected_protected_cols") or dataset_probe.get("protected_cols", [])
        target    = dataset_probe.get("detected_target_col") or dataset_probe.get("target_col", "auto")
    else:
        protected = [c.strip() for c in protected_cols.split(",") if c.strip() and c.strip() != "auto"]
        target    = target_col

    # ── Load dataset ─────────────────────────────────────────────────────────
    try:
        dataset_csv = await load_dataset_csv(dataset_file, dataset_source, dataset_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to load dataset: {e}")

    if not target or target == "auto":
        df_check = pd.read_csv(io.StringIO(dataset_csv))
        for col in df_check.columns:
            if df_check[col].nunique() == 2:
                target = col
                break
        else:
            target = df_check.columns[-1]

    # ── Load model ────────────────────────────────────────────────────────────
    model = None
    if model_file is not None:
        model_bytes = await model_file.read()
        if model_bytes:
            try:
                try:
                    raw = pickle.loads(model_bytes)
                except Exception:
                    import joblib
                    raw = joblib.load(io.BytesIO(model_bytes))
                model = FairLensAdapter.auto_detect(raw)
            except Exception as e:
                raise HTTPException(400, f"Failed to load model: {e}")

    elif model_type == "huggingface" and api_endpoint:
        model = FairLensAdapter.from_huggingface_auto(api_endpoint, hf_token=hf_token)
    elif model_type == "openai" and api_endpoint:
        model = FairLensAdapter.from_openai(model_name=api_endpoint, api_key=llm_api_key)
    elif model_type == "gemini_llm" and api_endpoint:
        model = FairLensAdapter.from_gemini(model_name=api_endpoint, api_key=llm_api_key)
    elif model_type == "api" and api_endpoint:
        model = FairLensAdapter.from_api(api_endpoint)

    if model is None:
        raise HTTPException(400, "No model provided for cross-analysis.")

    # ── Generate model predictions on user dataset ────────────────────────────
    df = pd.read_csv(io.StringIO(dataset_csv))
    raw_model = getattr(model, "_model", model)
    model_mt = (getattr(model, "get_model_type", lambda: "")() or "").lower()

    if "huggingface" in model_mt or model_type == "huggingface":
        # HF text classifiers need a text column synthesised from string columns
        if "text" not in df.columns:
            str_cols = df.select_dtypes(include=["object"]).columns.tolist()
            df["text"] = (
                df[str_cols].fillna("").astype(str).agg(" ".join, axis=1)
                if str_cols else df.astype(str).agg(" ".join, axis=1)
            )
        sample_size = min(300, len(df))
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        dataset_csv = df.to_csv(index=False)
        try:
            model_predictions = [int(p) for p in model.predict(df)]
        except Exception as e:
            logger.warning(f"[{audit_id}] HF cross-analysis predictions failed: {e}")
            raise HTTPException(422, f"Model could not predict on user dataset: {e}")
        feature_cols = [c for c in df.columns if c != target]
        X = df[feature_cols]

    elif model_type in ("openai", "gemini_llm"):
        feature_cols = [c for c in df.columns if c != target]
        sample_size = min(100, len(df))
        df = df.sample(sample_size, random_state=42).reset_index(drop=True)
        dataset_csv = df.to_csv(index=False)
        X = df[feature_cols]
        try:
            preds = model.predict(X)
            model_predictions = [int(round(float(p))) for p in preds]
        except Exception as e:
            raise HTTPException(422, f"Model could not predict on user dataset: {e}")

    else:
        feature_cols = resolve_feature_cols(raw_model, df, target)
        X = df[feature_cols]
        try:
            preds = model.predict(X)
            model_predictions = [int(p) for p in preds]
        except Exception as e:
            logger.warning(f"[{audit_id}] Cross-analysis model predictions failed: {e}")
            raise HTTPException(422, f"Model could not predict on user dataset: {e}")

    # ── 3a. Cartography on model × user dataset ───────────────────────────────
    carto_cross = await cartography_service.run_cartography(
        dataset_csv=dataset_csv,
        protected_cols=protected,
        target_col=target,
        model_predictions=model_predictions,
        audit_id=audit_id,
    )
    carto_cross["compliance_tags"] = check_compliance(carto_cross.get("slice_metrics", []))

    # ── 3b. Constitution on model × user dataset ──────────────────────────────
    try:
        y_pred_arr = np.array(model_predictions)
        constitution_cross = await constitution_service.generate_constitution(
            model=model,
            X=X,
            y_pred=y_pred_arr,
            protected_cols=protected,
            feature_names=feature_cols,
            cartography_results=carto_cross,
            audit_id=audit_id,
        )
    except Exception as e:
        logger.warning(f"[{audit_id}] Cross constitution failed: {e}")
        constitution_cross = {"error": str(e)}

    # ── 3c. Proxy hunter on user dataset ─────────────────────────────────────
    hunter = ProxyVariableHunter()
    y_series = (
        pd.to_numeric(df[target], errors="coerce").fillna(0)
        if target in df.columns else None
    )
    proxy_cross = await hunter.run_hunt(
        X=X, y=y_series, protected_cols=protected, audit_id=audit_id
    )

    # ── 3d. Cross-analysis synthesis (Phase 1 × Phase 2) ─────────────────────
    cross_synthesis = await cross_analyzer_service.analyze(
        model_probe_results=model_probe,
        dataset_probe_results=dataset_probe,
        audit_id=audit_id,
    )

    return JSONResponse(content={
        "audit_id":           audit_id,
        "analysis_type":      "cross_analysis",
        "cartography":        carto_cross,
        "constitution":       constitution_cross,
        "proxy":              proxy_cross,
        "cross_synthesis":    cross_synthesis,
        "detected_protected_cols": protected,
        "detected_target_col":    target,
        "summary": {
            "fair_score":             carto_cross.get("fair_score", {}),
            "model_type":             model_type,
            "compounded_risks":       cross_synthesis.get("summary", {}).get("total_compounded_risks", 0),
            "aligned_biases":         cross_synthesis.get("summary", {}).get("aligned_count", 0),
            "proxy_amplifications":   cross_synthesis.get("summary", {}).get("proxy_amplification_count", 0),
        },
    })
