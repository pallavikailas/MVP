"""
Dataset Probe API
=================
Phase 2 — analyze the user's uploaded dataset for structural biases
WITHOUT any model involvement (ground-truth labels only).
"""

import io
import logging
import uuid
from typing import Optional

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.auto_detect import auto_detect_columns
from app.services.dataset_loader import load_dataset_csv
from app.services.dataset_probe import dataset_probe_service
from app.services.compliance_mapper import check_compliance

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/run")
async def run_dataset_probe(
    dataset_file:   Optional[UploadFile] = File(default=None),
    protected_cols: str  = Form(default="auto"),
    target_col:     str  = Form(default="auto"),
    dataset_source: str  = Form(default="upload"),
    dataset_url:    str  = Form(default=""),
    hf_token:       str  = Form(default=""),
):
    """
    Analyze the user's dataset for structural biases (no model needed).

    Returns cartography + proxy-hunter results using ground-truth labels,
    plus a structured `dataset_biases` list for downstream cross-analysis.
    """
    audit_id = str(uuid.uuid4())[:8]

    try:
        dataset_csv = await load_dataset_csv(dataset_file, dataset_source, dataset_url)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Failed to load dataset: {e}")

    # ── Resolve protected columns ─────────────────────────────────────────────
    is_auto = protected_cols in ("auto", "", "['auto']") or "auto" in protected_cols.split(",")
    if is_auto:
        try:
            detected = await auto_detect_columns(dataset_csv, audit_id)
            protected = detected["protected_cols"]
            target    = detected["target_col"]
        except Exception as e:
            raise HTTPException(500, f"Auto-detect columns failed: {e}")
    else:
        protected = [c.strip() for c in protected_cols.split(",") if c.strip() and c.strip() != "auto"]
        target    = target_col

    # Fallback target detection
    if not target or target == "auto":
        df_check = pd.read_csv(io.StringIO(dataset_csv))
        for col in df_check.columns:
            if df_check[col].nunique() == 2:
                target = col
                break
        else:
            target = df_check.columns[-1]

    try:
        result = await dataset_probe_service.probe(
            dataset_csv=dataset_csv,
            protected_cols=protected,
            target_col=target,
            audit_id=audit_id,
        )
        result["detected_protected_cols"] = protected
        result["detected_target_col"]     = target
        # Attach compliance tags to cartography sub-result
        if "cartography" in result:
            result["cartography"]["compliance_tags"] = check_compliance(
                result["cartography"].get("slice_metrics", [])
            )
        return JSONResponse(content=result)
    except Exception as e:
        logger.exception(f"[{audit_id}] Dataset probe failed")
        raise HTTPException(500, f"Dataset probe failed: {e}")
