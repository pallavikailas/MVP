"""
Model Probe API
===============
Phase 1 — probe the model against the embedded reference dataset to detect
hidden biases intrinsic to the model, independent of the user's dataset.
"""

import io
import logging
import pickle
import uuid
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.model_adapter import FairLensAdapter
from app.services.model_probe import model_probe_service
from app.services.compliance_mapper import check_compliance

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/run")
async def run_model_probe(
    model_file:    Optional[UploadFile] = File(default=None),
    model_type:    str  = Form(default="sklearn"),
    api_endpoint:  str  = Form(default=""),
    llm_api_key:   str  = Form(default=""),
    hf_token:      str  = Form(default=""),
    protected_cols: str = Form(default=""),
):
    """
    Probe the model on the embedded reference dataset.

    Returns cartography + constitution results on that reference probe,
    plus a structured `model_biases` list for downstream cross-analysis.
    """
    audit_id = str(uuid.uuid4())[:8]
    model = None

    # ── Load model ────────────────────────────────────────────────────────────
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
                raise HTTPException(400, f"Failed to load model file: {e}")

    elif model_type == "huggingface" and api_endpoint:
        try:
            model = FairLensAdapter.from_huggingface_auto(api_endpoint, hf_token=hf_token)
        except Exception as e:
            raise HTTPException(400, f"HuggingFace model load failed: {e}")

    elif model_type == "openai" and api_endpoint:
        try:
            model = FairLensAdapter.from_openai(model_name=api_endpoint, api_key=llm_api_key)
        except Exception as e:
            raise HTTPException(400, f"OpenAI model config failed: {e}")

    elif model_type == "gemini_llm" and api_endpoint:
        try:
            model = FairLensAdapter.from_gemini(model_name=api_endpoint, api_key=llm_api_key)
        except Exception as e:
            raise HTTPException(400, f"Gemini model config failed: {e}")

    elif model_type == "api" and api_endpoint:
        try:
            model = FairLensAdapter.from_api(api_endpoint)
        except Exception as e:
            raise HTTPException(400, f"REST API adapter failed: {e}")

    if model is None:
        raise HTTPException(400, "No model provided. Upload a .pkl file or specify a model endpoint.")

    user_protected = [c.strip() for c in protected_cols.split(",") if c.strip() and c.strip() != "auto"]

    try:
        result = await model_probe_service.probe(
            model=model,
            model_type=model_type,
            audit_id=audit_id,
            user_protected_cols=user_protected or None,
        )
        # Attach compliance tags to cartography sub-result
        if "cartography" in result and not result.get("degenerate"):
            result["cartography"]["compliance_tags"] = check_compliance(
                result["cartography"].get("slice_metrics", [])
            )
        return JSONResponse(content=result)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        logger.exception(f"[{audit_id}] Model probe failed")
        raise HTTPException(500, f"Model probe failed: {e}")
