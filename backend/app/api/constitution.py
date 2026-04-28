"""API routes for Counterfactual Constitution."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd
import numpy as np
import pickle, io, uuid, json, asyncio, logging

from app.services.constitution import constitution_service
from app.services.auto_detect import auto_detect_columns
from app.services.dataset_loader import load_dataset_csv
from app.api._utils import resolve_feature_cols

logger = logging.getLogger(__name__)
router = APIRouter()



@router.post("/generate")
async def generate_constitution(
    dataset_file: Optional[UploadFile] = File(default=None),
    model_file: Optional[UploadFile] = File(default=None),
    protected_cols: str = Form(...),
    target_col: str = Form(...),
    cartography_results: str = Form(..., description="JSON string from cartography stage"),
    dataset_source: str = Form(default="upload"),
    dataset_url: str = Form(default=""),
    model_type: str = Form(default="sklearn"),
    api_endpoint: str = Form(default=""),
    llm_api_key: str = Form(default=""),
    hf_token: str = Form(default=""),
):
    audit_id = str(uuid.uuid4())[:8]
    try:
        # Load dataset
        dataset_csv = await load_dataset_csv(dataset_file, dataset_source, dataset_url)
        df = pd.read_csv(io.StringIO(dataset_csv))

        # Resolve protected_cols — handle 'auto' sentinel
        is_auto = protected_cols in ("auto", "", "['auto']") or "auto" in protected_cols.split(",")
        if is_auto:
            detected = await auto_detect_columns(dataset_csv, audit_id)
            protected = detected["protected_cols"]
            tgt = detected["target_col"]
        else:
            protected = [c.strip() for c in protected_cols.split(",") if c.strip() and c.strip() != "auto"]
            tgt = target_col if target_col and target_col != "auto" else None

        # Fallback target column detection
        if not tgt or tgt not in df.columns:
            for col in df.columns:
                if df[col].nunique() == 2:
                    tgt = col
                    break
            else:
                tgt = df.columns[-1]

        # Load model if provided
        model = None
        if model_file is not None:
            model_bytes = await model_file.read()
            if model_bytes:
                try:
                    raw = pickle.loads(model_bytes)
                except Exception:
                    try:
                        import joblib
                        raw = joblib.load(io.BytesIO(model_bytes))
                    except Exception as e:
                        raise HTTPException(400, f"Failed to load model file: {e}")
                # Always wrap raw models so predict auto-encodes categoricals consistently
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.auto_detect(raw)

        elif model_type == "huggingface" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_huggingface_auto(api_endpoint, hf_token=hf_token)
                # HuggingFaceAdapter._to_text auto-serialises tabular rows when no 'text' column present
            except Exception as e:
                err = str(e)
                if "No module named 'transformers'" in err:
                    raise HTTPException(500, "transformers library not installed on the server. Contact the administrator.")
                if any(kw in err.lower() for kw in ["text-generation", "causal", "generative", "seq2seq", "not supported for the pipeline"]):
                    raise HTTPException(400, f"'{api_endpoint}' is a generative/text-generation model. FairLens requires a text-classification model. Try one of: unitary/toxic-bert, cardiffnlp/twitter-roberta-base-sentiment, valurank/distilroberta-base-offensive-language-identification")
                raise HTTPException(400, f"Failed to load HuggingFace model '{api_endpoint}': {e}")

        elif model_type == "api" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_api(api_endpoint)
            except Exception as e:
                raise HTTPException(400, f"Failed to connect to API model '{api_endpoint}': {e}")

        elif model_type == "llm_hf" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_generative_huggingface(api_endpoint, hf_token=hf_token)
            except Exception as e:
                raise HTTPException(400, f"Failed to load HuggingFace generative model '{api_endpoint}': {e}")

        elif model_type == "openai" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_openai(model_name=api_endpoint, api_key=llm_api_key)
            except Exception as e:
                raise HTTPException(400, f"Failed to initialise OpenAI model '{api_endpoint}': {e}")

        elif model_type == "gemini_llm" and api_endpoint:
            try:
                from app.services.model_adapter import FairLensAdapter
                model = FairLensAdapter.from_gemini(model_name=api_endpoint, api_key=llm_api_key)
            except Exception as e:
                raise HTTPException(400, f"Failed to initialise Gemini model '{api_endpoint}': {e}")

        feature_cols = resolve_feature_cols(model, df, tgt) if model is not None else [c for c in df.columns if c != tgt]
        X = df[feature_cols]

        # Require a model — reject if none was provided
        if model is None:
            raise HTTPException(
                400,
                "No model provided. Upload a .pkl file or specify a model endpoint "
                "(HuggingFace, OpenAI, Gemini, REST API). "
                "FairLens generates counterfactual constitutions from your model's decisions."
            )

        # Generate predictions
        # For HF/API models, sample to keep latency under ~90s (HF Inference API ~0.5s/call)
        model_type_str = (model.get_model_type() if model is not None and hasattr(model, "get_model_type") else "") or ""
        is_api_model = any(t in model_type_str for t in ("HuggingFace", "REST:", "GenerativeLLM"))
        pred_sample_size = 150 if is_api_model else len(X)

        X_pred = X.iloc[:pred_sample_size] if pred_sample_size < len(X) else X
        try:
            y_pred_sample = model.predict(X_pred)
            if pred_sample_size < len(X):
                import scipy.stats as _stats
                fill_val = int(_stats.mode(y_pred_sample, keepdims=True).mode[0])
                y_pred = np.full(len(X), fill_val, dtype=int)
                y_pred[:pred_sample_size] = y_pred_sample
            else:
                y_pred = y_pred_sample
        except Exception as e:
            raise HTTPException(400, f"Model prediction failed: {e}. Check that the model is compatible with your dataset.")

        carto = json.loads(cartography_results)

        try:
            result = await asyncio.wait_for(
                constitution_service.generate_constitution(
                    model=model,
                    X=X,
                    y_pred=y_pred,
                    protected_cols=protected,
                    feature_names=feature_cols,
                    cartography_results=carto,
                    audit_id=audit_id,
                ),
                timeout=540,  # 9 min — well under Cloud Run's 1hr, avoids gateway timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                504,
                "Constitution timed out (>9 min). The model API is too slow for full counterfactual analysis. "
                "Try a faster model or reduce dataset size."
            )
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
