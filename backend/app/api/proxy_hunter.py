"""API routes for Proxy Variable Hunter."""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from typing import Optional
import pandas as pd, io, uuid

from app.services.dataset_loader import load_dataset_csv

router = APIRouter()


@router.post("/hunt")
async def hunt_proxies(
    dataset_file: Optional[UploadFile] = File(default=None),
    protected_cols: str = Form(...),
    target_col: str = Form(...),
    dataset_source: str = Form(default="upload"),
    dataset_url: str = Form(default=""),
):
    from app.services.proxy_hunter import proxy_hunter_service
    from app.services.auto_detect import auto_detect_columns

    audit_id = str(uuid.uuid4())[:8]
    try:
        dataset_csv = await load_dataset_csv(dataset_file, dataset_source, dataset_url)
        df = pd.read_csv(io.StringIO(dataset_csv))

        is_auto = protected_cols in ("auto", "", "['auto']") or "auto" in protected_cols.split(",")
        if is_auto:
            detected = await auto_detect_columns(dataset_csv, audit_id)
            protected = detected["protected_cols"]
            tgt = detected["target_col"]
        else:
            protected = [c.strip() for c in protected_cols.split(",") if c.strip() and c.strip() != "auto"]
            tgt = target_col if target_col and target_col != "auto" else None

        if not tgt or tgt not in df.columns:
            for col in df.columns:
                if df[col].nunique() == 2:
                    tgt = col
                    break
            else:
                tgt = df.columns[-1]

        X = df[[c for c in df.columns if c != tgt]]
        y = pd.to_numeric(df[tgt], errors="coerce").fillna(0) if tgt in df.columns else None

        result = await proxy_hunter_service.run_hunt(X, y, protected, audit_id)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
