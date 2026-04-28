"""Shared helpers for API route handlers."""
from typing import List
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def resolve_feature_cols(model, df: pd.DataFrame, fallback_target: str) -> List[str]:
    """Return the feature columns the model expects, falling back to all non-target columns."""
    if hasattr(model, "feature_names_in_"):
        names = list(model.feature_names_in_)
        if names and all(n in df.columns for n in names):
            return names
    if hasattr(model, "feature_names") and model.feature_names:
        names = model.feature_names
        if all(n in df.columns for n in names):
            return list(names)
    if hasattr(model, "feature_name_"):
        try:
            names = model.feature_name_()
            if names and all(n in df.columns for n in names):
                return list(names)
        except Exception:
            pass
    return [c for c in df.columns if c != fallback_target]
