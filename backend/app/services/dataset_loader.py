"""
Shared dataset loading utility.

Supports: file upload, direct URL, HuggingFace datasets (with multi-fallback + 429 retry).
Returns CSV string in all cases so callers can pd.read_csv(io.StringIO(csv)).
"""
import asyncio
import io
import logging
import time
from typing import Optional, Tuple

import httpx
import pandas as pd
from fastapi import HTTPException, UploadFile

logger = logging.getLogger(__name__)

_HF_SERVER = "https://datasets-server.huggingface.co"
_HF_HUB = "https://huggingface.co"
_MAX_FILE_BYTES = 100 * 1024 * 1024  # 100 MB

# In-memory cache for HuggingFace datasets.
# Keyed by dataset name; value is (csv_string, loaded_at_timestamp).
# Entries expire after 10 minutes so a session's 4+ pipeline stages share one HF fetch.
_HF_CACHE: dict[str, Tuple[str, float]] = {}
_HF_CACHE_TTL = 600  # seconds


async def _get_with_retry(client: httpx.AsyncClient, url: str, max_retries: int = 3, **kwargs) -> httpx.Response:
    """GET with exponential back-off on 429 responses, honouring Retry-After."""
    for attempt in range(max_retries):
        resp = await client.get(url, **kwargs)
        if resp.status_code != 429:
            return resp
        # Honour server-supplied Retry-After (seconds or HTTP-date int)
        try:
            wait = int(resp.headers.get("Retry-After", 0)) or 2 ** attempt
        except (ValueError, TypeError):
            wait = 2 ** attempt
        logger.info(f"[HF] 429 rate-limited on {url} — retrying in {wait}s (attempt {attempt+1}/{max_retries})")
        await asyncio.sleep(wait)
    return resp  # return last response even if still 429


async def load_dataset_csv(
    dataset_file: Optional[UploadFile],
    dataset_source: str,
    dataset_url: str,
) -> str:
    """Load a dataset from any supported source and return it as a CSV string."""
    if dataset_source == "upload" or not dataset_url:
        if not dataset_file:
            raise HTTPException(400, "dataset_file is required when dataset_source is 'upload'")
        raw = await dataset_file.read()
        return raw.decode("utf-8", errors="replace")

    if dataset_source == "url":
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            resp = await client.get(dataset_url)
            resp.raise_for_status()
            return resp.text

    if dataset_source == "huggingface":
        name = dataset_url.strip()
        # Serve from cache if the same dataset was fetched recently in this process.
        cached = _HF_CACHE.get(name)
        if cached and (time.time() - cached[1]) < _HF_CACHE_TTL:
            logger.info(f"[HF cache hit] Returning cached CSV for '{name}'")
            return cached[0]
        csv = await _load_huggingface(name)
        _HF_CACHE[name] = (csv, time.time())
        return csv

    if dataset_source == "kaggle":
        raise HTTPException(
            400,
            "Kaggle requires authentication. Download the CSV from Kaggle and upload it directly."
        )

    raise HTTPException(400, f"Unknown dataset_source: '{dataset_source}'")


async def _load_huggingface(name: str) -> str:
    """
    Load a HuggingFace dataset by trying three strategies in order:
      1. datasets-server /rows  (instant, works for indexed datasets)
      2. Hub file listing → download first small CSV / JSON / JSONL file
      3. datasets-server /parquet → download first parquet shard < 100 MB
    All HTTP requests retry on 429 with exponential back-off.
    """
    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:

        # --- Strategy 1: datasets-server /rows ---
        # Use /splits (not /info) — it directly returns valid (config, split) pairs,
        # avoiding the /info dataset_info parsing that silently falls back to "default"
        # when the real config name differs (e.g. "bias_in_bios" for LabHC/bias_in_bios).
        config, split = "default", "train"
        splits_resp = await _get_with_retry(client, f"{_HF_SERVER}/splits?dataset={name}")
        if splits_resp.status_code == 401:
            raise HTTPException(
                400,
                f"HuggingFace dataset '{name}' is gated or private. "
                f"Download the CSV manually from huggingface.co/datasets/{name} and upload it directly."
            )
        if splits_resp.status_code == 404:
            raise HTTPException(
                400,
                f"HuggingFace dataset '{name}' was not found. "
                f"Check the dataset name at huggingface.co/datasets."
            )
        if splits_resp.status_code == 200:
            split_entries = splits_resp.json().get("splits", [])
            if split_entries:
                # Prefer a "train" split; otherwise take the first available
                train_entry = next((s for s in split_entries if s.get("split") == "train"), split_entries[0])
                config = train_entry.get("config", config)
                split = train_entry.get("split", split)
                logger.info(f"[HF] /splits resolved config='{config}' split='{split}' for {name}")
        # Non-200 other than 401/404: fall through to parquet/file strategies.

        rows_resp = await _get_with_retry(
            client,
            f"{_HF_SERVER}/rows?dataset={name}&config={config}&split={split}&offset=0&length=500"
        )
        if rows_resp.status_code == 200:
            rows = rows_resp.json().get("rows", [])
            if rows:
                logger.info(f"[HF] Loaded {len(rows)} rows via datasets-server for {name}")
                return pd.DataFrame([r["row"] for r in rows]).to_csv(index=False)
            logger.info(f"[HF] datasets-server returned 0 rows for {name} — falling through to file listing")
        else:
            logger.info(f"[HF] datasets-server /rows returned {rows_resp.status_code} for {name} — falling through")

        # --- Strategy 2: Hub file listing → CSV/JSON ---
        try:
            tree_resp = await _get_with_retry(client, f"{_HF_HUB}/api/datasets/{name}/tree/main")
            if tree_resp.status_code == 200:
                tree = tree_resp.json()
                candidates = [
                    f for f in tree if isinstance(f, dict)
                    and f.get("path", "").lower().endswith((".csv", ".tsv", ".jsonl", ".json"))
                    and f.get("size", _MAX_FILE_BYTES + 1) < _MAX_FILE_BYTES
                ]
                candidates.sort(key=lambda f: f.get("size", 0))
                for f in candidates[:3]:
                    path = f["path"]
                    dl = await _get_with_retry(client, f"{_HF_HUB}/datasets/{name}/resolve/main/{path}")
                    if dl.status_code != 200:
                        continue
                    try:
                        if path.lower().endswith(".tsv"):
                            df = pd.read_csv(io.StringIO(dl.text), sep="\t", nrows=500)
                        elif path.lower().endswith(".jsonl"):
                            df = pd.read_json(io.StringIO(dl.text), lines=True, nrows=500)
                        elif path.lower().endswith(".json"):
                            df = pd.read_json(io.StringIO(dl.text), nrows=500)
                        else:
                            df = pd.read_csv(io.StringIO(dl.text), nrows=500)
                        logger.info(f"[HF] Loaded {len(df)} rows from {path} for {name}")
                        return df.to_csv(index=False)
                    except Exception as e:
                        logger.debug(f"[HF] Failed to parse {path}: {e}")
                        continue
        except Exception as e:
            logger.debug(f"[HF] Hub tree listing failed: {e}")

        # --- Strategy 3: parquet shards (with aggressive 429 retry) ---
        # Some datasets (e.g. LabHC/bias_in_bios) are parquet-only and HuggingFace
        # rate-limits parquet CDN downloads heavily.  We retry up to 6 times per shard
        # (1+2+4+8+16+32 = 63 s max wait) and honour Retry-After headers.
        try:
            parquet_resp = await _get_with_retry(client, f"{_HF_SERVER}/parquet?dataset={name}")
            if parquet_resp.status_code == 200:
                pfiles = sorted(
                    parquet_resp.json().get("parquet_files", []),
                    key=lambda x: x.get("size", _MAX_FILE_BYTES + 1),
                )
                for pf in pfiles[:4]:  # fewer shards, more retries per shard
                    if pf.get("size", _MAX_FILE_BYTES + 1) > _MAX_FILE_BYTES:
                        continue
                    dl = await _get_with_retry(client, pf["url"], max_retries=6)
                    if dl.status_code != 200:
                        logger.debug(f"[HF] Parquet shard {pf.get('url','')[:60]} returned {dl.status_code}")
                        continue
                    try:
                        import pyarrow.parquet as pq
                        table = pq.read_table(io.BytesIO(dl.content))
                        df = table.slice(0, min(500, table.num_rows)).to_pandas()
                        logger.info(f"[HF] Loaded {len(df)} rows from parquet shard for {name}")
                        return df.to_csv(index=False)
                    except Exception as e:
                        logger.debug(f"[HF] Failed to read parquet shard: {e}")
                        continue
        except Exception as e:
            logger.debug(f"[HF] Parquet fallback failed: {e}")

    raise HTTPException(
        400,
        f"Could not load HuggingFace dataset '{name}'. "
        f"The dataset may be gated, too large, or not indexed. "
        f"Download the CSV from huggingface.co/datasets/{name} and upload it directly."
    )
