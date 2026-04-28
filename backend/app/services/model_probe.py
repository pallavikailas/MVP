"""
Model Bias Probe Service
========================
Probes the provided model against a neutral EMBEDDED reference dataset to reveal
hidden biases intrinsic to the model itself — completely independent of whatever
dataset the user has uploaded.

Pipeline
--------
1. Generate reference probe dataset (fixed 300-row dataset OR model-specific probe)
2. Run model predictions on reference dataset
3. Run Bias Cartography on (reference data + model predictions) → demographic disparity map
4. Run Counterfactual Constitution on (reference data + model) → implicit decision rules
5. Extract structured bias list for red-team targeting
"""

import io
import logging
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional

from app.services.reference_dataset import (
    generate_reference_dataset,
    generate_text_reference_dataset,
    generate_model_specific_probe,
    REFERENCE_PROTECTED_COLS,
    REFERENCE_TARGET_COL,
)
from app.services.cartography import cartography_service
from app.services.constitution import constitution_service

logger = logging.getLogger(__name__)

_LLM_TYPES = {"GenerativeLLM", "OpenAI", "Gemini", "HuggingFace"}


def _is_llm(model) -> bool:
    mt = (getattr(model, "get_model_type", lambda: "")() or "")
    return any(t in mt for t in _LLM_TYPES)


def _uses_text_reference_probe(model) -> bool:
    mt = (getattr(model, "get_model_type", lambda: "")() or "")
    return mt == "HuggingFace"


class ModelBiasProbe:

    async def probe(
        self,
        model: Any,
        model_type: str,
        audit_id: str,
        user_protected_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Probe *model* on the embedded reference dataset and return
        a structured dict of discovered hidden biases.
        """
        logger.info(f"[{audit_id}] Starting model bias probe with embedded reference dataset")

        # ── 1. Build probe dataset ────────────────────────────────────────────
        if _uses_text_reference_probe(model):
            ref_df, ref_csv = generate_text_reference_dataset()
            probe_protected = REFERENCE_PROTECTED_COLS
            probe_target = REFERENCE_TARGET_COL
        elif _is_llm(model):
            # LLM models handle any text; use the fixed standard reference dataset
            ref_df, ref_csv = generate_reference_dataset()
            probe_protected = REFERENCE_PROTECTED_COLS
            probe_target    = REFERENCE_TARGET_COL
        else:
            # Structured (sklearn / API) models: generate a dataset matching model features
            feature_names = self._get_feature_names(model)
            if feature_names:
                ref_df, ref_csv, probe_protected, probe_target, model_predict_cols = (
                    generate_model_specific_probe(feature_names, protected_cols=user_protected_cols)
                )
            else:
                # Fallback to standard reference dataset
                ref_df, ref_csv = generate_reference_dataset()
                probe_protected    = REFERENCE_PROTECTED_COLS
                probe_target       = REFERENCE_TARGET_COL
                model_predict_cols = None  # will use all non-target cols

        logger.info(
            f"[{audit_id}] Probe dataset: {len(ref_df)} rows, "
            f"protected={probe_protected}, target='{probe_target}'"
        )

        # ── 2. Get model predictions on reference dataset ────────────────────
        # model_predict_cols may be a subset when demographics were injected
        if _is_llm(model) or model_predict_cols is None:
            feature_cols = [c for c in ref_df.columns if c != probe_target]
        else:
            feature_cols = model_predict_cols
        X_ref = ref_df[feature_cols]

        try:
            raw_preds = model.predict(X_ref)
            model_predictions = [int(p) for p in raw_preds]
            logger.info(f"[{audit_id}] Model generated {len(model_predictions)} predictions on reference dataset")
        except Exception as first_err:
            # Fallback: label-encode any categorical columns and retry
            logger.warning(f"[{audit_id}] Direct prediction failed ({first_err}), retrying with label encoding")
            try:
                from sklearn.preprocessing import LabelEncoder
                X_enc = X_ref.copy()
                for col in X_enc.select_dtypes(include=["object", "category"]).columns:
                    try:
                        X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
                    except Exception:
                        X_enc[col] = 0
                X_enc = X_enc.fillna(0)
                raw_preds = model.predict(X_enc)
                model_predictions = [int(p) for p in raw_preds]
                logger.info(f"[{audit_id}] Label-encoded prediction: {len(model_predictions)} predictions")
            except Exception as e:
                logger.error(f"[{audit_id}] Model prediction on reference dataset failed: {e}")
                raise ValueError(f"Model could not be probed on reference dataset: {e}")

        diagnostics = self._prediction_diagnostics(
            ref_df=ref_df,
            predictions=model_predictions,
            target_col=probe_target,
        )

        # ── 3. Bias Cartography on reference data + model predictions ────────
        carto_results = await cartography_service.run_cartography(
            dataset_csv=ref_csv,
            protected_cols=probe_protected,
            target_col=probe_target,
            model_predictions=model_predictions,
            audit_id=audit_id,
        )
        if diagnostics["collapsed_output"] or diagnostics["near_constant_output"]:
            carto_results["fair_score"] = {
                "score": 0,
                "label": "Invalid",
                "color": "red",
                "reason": diagnostics["reason"],
            }

        # ── 4. Counterfactual Constitution on reference data + model ─────────
        # Pass the FULL ref_df (including injected demographics) so constitution can
        # flip demographic columns.  Use only model feature cols for re-prediction.
        X_full = ref_df[[c for c in ref_df.columns if c != probe_target]]
        try:
            y_pred_arr = np.array(model_predictions)
            constitution_results = await constitution_service.generate_constitution(
                model=model,
                X=X_full,
                y_pred=y_pred_arr,
                protected_cols=probe_protected,
                feature_names=list(X_full.columns),
                cartography_results=carto_results,
                audit_id=audit_id,
            )
        except Exception as e:
            logger.warning(f"[{audit_id}] Constitution on reference dataset failed: {e}")
            constitution_results = {"error": str(e), "patterns": [], "sections": []}

        # ── 5. Extract structured bias list ──────────────────────────────────
        model_biases = self._extract_model_biases(carto_results, constitution_results)
        if diagnostics["collapsed_output"] or diagnostics["near_constant_output"]:
            model_biases.insert(0, {
                "attribute": "model_output_distribution",
                "value": diagnostics["reason"],
                "type": "model_failure",
                "severity": "critical",
                "magnitude": 1.0,
                "source": "model_probe_diagnostics",
                "positive_rate": diagnostics["positive_rate"],
                "accuracy_vs_reference": diagnostics["accuracy_vs_reference"],
            })

        return {
            "audit_id":              audit_id,
            "analysis_type":         "model_probe",
            "reference_dataset_size": len(ref_df),
            "reference_protected_cols": probe_protected,
            "reference_target_col":  probe_target,
            "cartography":           carto_results,
            "constitution":          constitution_results,
            "prediction_diagnostics": diagnostics,
            "model_biases":          model_biases,
            "summary": {
                "fair_score":             carto_results.get("fair_score", {}),
                "bias_count":             len(model_biases),
                "most_biased_attribute":  model_biases[0]["attribute"] if model_biases else None,
                "analysis_source":        "embedded_reference_dataset",
                "model_type":             model_type,
                "prediction_diagnostics": diagnostics,
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _get_feature_names(model) -> Optional[List[str]]:
        """Extract feature names from sklearn-style model, including ensembles and pipelines."""
        # Unwrap adapter layers: try _raw_model() first (BaseModelAdapter interface),
        # then common attribute names used by different adapter implementations.
        if hasattr(model, "_raw_model"):
            try:
                raw = model._raw_model()
            except Exception:
                raw = getattr(model, "model", getattr(model, "_model", model))
        else:
            raw = getattr(model, "model", getattr(model, "_model", model))

        def _from_estimator(est) -> Optional[List[str]]:
            for attr in ("feature_names_in_", "feature_names"):
                val = getattr(est, attr, None)
                if val is not None:
                    return list(val)
            try:
                val = est.feature_name_()
                if val:
                    return list(val)
            except Exception:
                pass
            return None

        # Direct attributes
        found = _from_estimator(raw)
        if found:
            return found

        # Pipeline: check steps in reverse (last fitted step most likely has feature_names_in_)
        if hasattr(raw, "steps"):
            for _, step in reversed(raw.steps):
                found = _from_estimator(step)
                if found:
                    return found

        # VotingClassifier / StackingClassifier / BaggingClassifier
        for attr in ("estimators_", "estimators"):
            estimators = getattr(raw, attr, None)
            if not estimators:
                continue
            for est in estimators:
                est_raw = est[1] if isinstance(est, tuple) else est
                found = _from_estimator(est_raw)
                if found:
                    return found
                # Sub-estimator may also be a Pipeline
                if hasattr(est_raw, "steps"):
                    for _, step in reversed(est_raw.steps):
                        found = _from_estimator(step)
                        if found:
                            return found

        return None

    @staticmethod
    def _calibrate_probe(model, feature_cols, probe_protected, original_ref_df, probe_target):
        """
        Try progressively different input distributions until predictions are mixed (5–95%).
        Returns (ref_df, ref_csv, X_ref, predictions) on success, or None on failure.
        """
        rng = np.random.default_rng(0)
        n = 240

        # ── Feature-specific heuristic ranges ────────────────────────────────
        def _heuristic_range(feat):
            fl = feat.lower()
            if any(k in fl for k in ("temp", "celsius", "fahrenheit")):   return (0.0, 45.0)
            if any(k in fl for k in ("humid", "moisture", "saturation")): return (0.0, 100.0)
            if any(k in fl for k in ("rain", "precipitation")):           return (0.0, 400.0)
            if any(k in fl for k in ("ph", "acidity")):                   return (3.5, 9.5)
            if any(k in fl for k in ("wind", "speed", "velocity")):       return (0.0, 120.0)
            if any(k in fl for k in ("sun", "light", "solar", "hour")):   return (0.0, 14.0)
            if any(k in fl for k in ("diversity", "index", "ratio")):     return (0.0, 1.0)
            if any(k in fl for k in ("stage", "growth", "phase")):        return (1.0, 6.0)
            if any(k in fl for k in ("type", "class", "category")):       return (0.0, 8.0)
            if any(k in fl for k in ("score", "credit", "fico")):         return (300.0, 850.0)
            if any(k in fl for k in ("income", "salary", "wage")):        return (20_000.0, 200_000.0)
            if any(k in fl for k in ("age",)):                            return (18.0, 80.0)
            return None  # no heuristic — fall back to distribution-level ranges

        # Distributions to try in order
        DISTS: List[Dict[str, tuple]] = [
            # 1. Feature-specific heuristic
            {f: (_heuristic_range(f) or (0.0, 100.0)) for f in feature_cols},
            # 2. Normalised 0-1
            {f: (0.0, 1.0)    for f in feature_cols},
            # 3. Small integers
            {f: (0.0, 10.0)   for f in feature_cols},
            # 4. Wide
            {f: (0.0, 1000.0) for f in feature_cols},
            # 5. Signed normalised
            {f: (-1.0, 1.0)   for f in feature_cols},
            # 6. Very wide
            {f: (0.0, 10_000.0) for f in feature_cols},
        ]

        best_X, best_full, best_preds, best_dist = None, None, None, 1.0

        for ranges in DISTS:
            data = {f: rng.uniform(lo, hi, n) for f, (lo, hi) in ranges.items()}
            X = pd.DataFrame(data)
            try:
                preds = [int(p) for p in model.predict(X)]
                pr = sum(preds) / len(preds)
                if abs(pr - 0.5) < best_dist:
                    best_dist = abs(pr - 0.5)
                    best_X = X
                    best_preds = preds
                    if best_dist < 0.45:      # pos_rate 5–95% → good enough
                        break
            except Exception:
                continue

        # ── Mixing strategy: half min-values, half max-values ─────────────────
        if best_dist >= 0.45:
            for lo_scale, hi_scale in [(0.0, 1e4), (0.0, 1e6), (-1e4, 1e4)]:
                n2 = n // 2
                Xlo = pd.DataFrame({f: rng.uniform(lo_scale, lo_scale + 0.01 * abs(hi_scale - lo_scale), n2) for f in feature_cols})
                Xhi = pd.DataFrame({f: rng.uniform(hi_scale * 0.9, hi_scale, n2)                             for f in feature_cols})
                try:
                    pl = [int(p) for p in model.predict(Xlo)]
                    ph = [int(p) for p in model.predict(Xhi)]
                    combined = pl + ph
                    pr = sum(combined) / len(combined)
                    if abs(pr - 0.5) < best_dist:
                        best_dist  = abs(pr - 0.5)
                        best_X     = pd.concat([Xlo, Xhi], ignore_index=True)
                        best_preds = combined
                        if best_dist < 0.45:
                            break
                except Exception:
                    continue

        if best_X is None or best_dist >= 0.47:
            return None  # give up

        # Re-attach demographic cols and probe target to build full ref_df
        full = best_X.copy()
        for col in probe_protected:
            if col in original_ref_df.columns and col not in full.columns:
                full[col] = original_ref_df[col].values[:len(full)]
        # Inject standard demographics if none present
        demo_needed = [c for c in ("gender", "race", "age_group") if c not in full.columns]
        if demo_needed:
            full["gender"]    = rng.choice(["Male", "Female", "Non-binary"], len(full))
            full["race"]      = rng.choice(["White", "Black", "Hispanic", "Asian", "Other"], len(full))
            full["age_group"] = rng.choice(["18-30", "31-45", "46-60", "60+"], len(full))

        # Use model predictions as the probe target
        full[probe_target] = best_preds
        ref_csv = full.to_csv(index=False)
        return full, ref_csv, best_X, best_preds

    @staticmethod
    def _extract_model_biases(carto: Dict, constitution: Dict) -> List[Dict]:
        """Collect structured bias objects from cartography hotspots + constitution patterns."""
        biases: Dict[str, Dict] = {}

        for hotspot in carto.get("hotspots", []):
            attr = str(hotspot.get("dominant_slice", "")).split("=")[0].strip()
            if not attr:
                continue
            entry = {
                "attribute": attr,
                "value":     hotspot.get("dominant_slice", ""),
                "type":      "statistical_disparity",
                "severity":  hotspot.get("severity", "medium"),
                "magnitude": float(hotspot.get("mean_bias_magnitude", 0)),
                "source":    "model_probe_cartography",
                "spd":       float(hotspot.get("statistical_parity_diff", 0)),
            }
            if attr not in biases or entry["magnitude"] > biases[attr]["magnitude"]:
                biases[attr] = entry

        for pattern in constitution.get("patterns", []):
            attr = pattern.get("attribute", "")
            if not attr or pattern.get("flip_rate", 0) <= 0.05:
                continue
            entry = {
                "attribute": attr,
                "value":     pattern.get("bias_direction", ""),
                "type":      "counterfactual_flip",
                "severity":  pattern.get("severity", "medium"),
                "magnitude": float(pattern.get("flip_rate", 0)),
                "source":    "model_probe_constitution",
                "flip_rate": float(pattern.get("flip_rate", 0)),
            }
            if attr not in biases or entry["magnitude"] > biases[attr]["magnitude"]:
                biases[attr] = entry

        return sorted(biases.values(), key=lambda x: x["magnitude"], reverse=True)

    @staticmethod
    def _prediction_diagnostics(ref_df: pd.DataFrame, predictions: List[int], target_col: str) -> Dict[str, Any]:
        pred = np.array(predictions, dtype=int)
        unique_values = sorted(np.unique(pred).tolist())
        positive_rate = float(pred.mean()) if len(pred) else 0.0
        collapsed_output = len(unique_values) <= 1
        near_constant_output = positive_rate <= 0.01 or positive_rate >= 0.99

        accuracy_vs_reference = None
        if target_col in ref_df.columns:
            y_true = pd.to_numeric(ref_df[target_col], errors="coerce").fillna(0).astype(int).values
            if len(y_true) == len(pred):
                accuracy_vs_reference = round(float((y_true == pred).mean()), 4)

        reason = ""
        if collapsed_output:
            label = unique_values[0] if unique_values else 0
            reason = f"Model predicted a single class ({label}) for every reference sample."
        elif near_constant_output:
            reason = f"Model output was near-constant across the reference probe (positive rate {positive_rate:.3f})."

        return {
            "unique_prediction_count": len(unique_values),
            "unique_prediction_values": unique_values,
            "positive_rate": round(positive_rate, 4),
            "collapsed_output": collapsed_output,
            "near_constant_output": near_constant_output,
            "accuracy_vs_reference": accuracy_vs_reference,
            "reason": reason,
        }


model_probe_service = ModelBiasProbe()
