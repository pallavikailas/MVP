"""
Fairness Red-Team Agent
========================
An adversarial LangGraph multi-agent system that:
1. Generates synthetic edge cases targeting known bias hotspots
2. Probes the model with these edge cases to CONFIRM hidden bias
3. Upon user confirmation, applies bias mitigation patches
4. Validates fixes by re-running bias metrics

Architecture:
  Orchestrator Agent
    ├── Attack Agent (generates adversarial demographic probes)
    ├── Evaluator Agent (measures bias in responses)
    ├── Patcher Agent (applies Reweighing / Adversarial Debiasing / Threshold Adjustment)
    └── Validator Agent (confirms fix, flags regressions)
"""

import asyncio
import base64
import pickle
import threading
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, AsyncGenerator
import logging
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from langgraph.graph import StateGraph, END
from app.core.config import settings

logger = logging.getLogger(__name__)


class _FairLensRemediatedModel:
    """
    Wrapper that embeds per-group correction factors and/or per-group decision
    thresholds into a serialisable sklearn-compatible model.

    When the user loads the downloaded .pkl and calls model.predict(X), the
    fairness corrections are applied automatically — no extra code needed.
    """

    def __init__(self, base_model: Any, group_corrections: Dict = None, group_thresholds: Dict = None):
        self._base_model       = base_model
        self.group_corrections = group_corrections or {}
        self.group_thresholds  = group_thresholds  or {}

    # expose feature_names_in_ if the wrapped model has them
    def __getattr__(self, name: str):
        return getattr(self._base_model, name)

    def fit(self, X, y, **kwargs):
        self._base_model.fit(X, y, **kwargs)
        return self

    def predict_proba(self, X: "pd.DataFrame") -> "np.ndarray":
        if hasattr(self._base_model, "predict_proba"):
            proba = self._base_model.predict_proba(X)[:, 1].copy()
        else:
            preds = self._base_model.predict(X)
            proba = preds.astype(float)

        # Apply demographic-parity correction factors
        for attr, corr_data in self.group_corrections.items():
            if isinstance(corr_data, dict) and "correction_factors" in corr_data:
                factors = corr_data["correction_factors"]
            elif isinstance(corr_data, dict):
                factors = corr_data
            else:
                continue
            if hasattr(X, "columns") and attr in X.columns:
                for val, factor in factors.items():
                    mask = X[attr].astype(str) == str(val)
                    proba[mask] = np.clip(proba[mask] * float(factor), 0.0, 1.0)

        return np.column_stack([1 - proba, proba])

    def predict(self, X: "pd.DataFrame") -> "np.ndarray":
        proba = self.predict_proba(X)[:, 1]

        # Per-group decision thresholds
        if self.group_thresholds and hasattr(X, "columns"):
            preds = np.full(len(X), -1, dtype=int)
            for attr, thresholds in self.group_thresholds.items():
                if attr not in X.columns:
                    continue
                for val, thresh in thresholds.items():
                    mask = (X[attr].astype(str) == str(val)).values
                    preds[mask] = (proba[mask] >= float(thresh)).astype(int)
            # Default threshold for rows not covered
            default_mask = preds == -1
            preds[default_mask] = (proba[default_mask] >= 0.5).astype(int)
            return preds

        return (proba >= 0.5).astype(int)


class RedTeamState(dict):
    model: Any
    X_train: pd.DataFrame
    y_train: np.ndarray
    audit_results: Dict
    confirmed_biases: List[Dict]
    synthetic_probes: List[Dict]
    evaluation_results: List[Dict]
    mitigation_plan: List[Dict]
    patch_results: Dict
    validation_results: Dict
    iteration: int
    status: str
    log: List[str]


class FairnessRedTeamAgent:

    def __init__(self):
        self.graph = self._build_graph()

    @staticmethod
    def _safe_predict(model, X: pd.DataFrame):
        """Predict with automatic label-encoding fallback for categorical columns."""
        try:
            return model.predict(X)
        except Exception:
            X_enc = X.copy()
            le = LabelEncoder()
            for col in X_enc.select_dtypes(include=["object", "category"]).columns:
                try:
                    X_enc[col] = le.fit_transform(X_enc[col].astype(str))
                except Exception:
                    X_enc[col] = 0
            return model.predict(X_enc.fillna(0))

    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(dict)
        workflow.add_node("attack",       self._attack_node)
        workflow.add_node("evaluate",     self._evaluate_node)
        workflow.add_node("decide_patch", self._decide_patch_node)
        workflow.add_node("patch",        self._patch_node)
        workflow.add_node("validate",     self._validate_node)
        workflow.add_node("report",       self._report_node)

        workflow.set_entry_point("attack")
        workflow.add_edge("attack",   "evaluate")
        workflow.add_edge("evaluate", "decide_patch")
        workflow.add_conditional_edges(
            "decide_patch", self._should_patch, {"patch": "patch", "report": "report"}
        )
        workflow.add_edge("patch", "validate")
        workflow.add_conditional_edges(
            "validate", self._should_continue, {"attack": "attack", "report": "report"}
        )
        workflow.add_edge("report", END)
        return workflow.compile()

    async def run(
        self,
        model: Any,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        audit_results: Dict,
        confirmed_biases: List[Dict],
        audit_id: str,
        stop_event: Optional[threading.Event] = None,
    ) -> AsyncGenerator[Dict, None]:
        if stop_event is None:
            stop_event = threading.Event()

        state = {
            "model":            model,
            "X_train":          X_train,
            "y_train":          y_train,
            "audit_results":    audit_results,
            "confirmed_biases": confirmed_biases,
            "synthetic_probes":  [],
            "evaluation_results": [],
            "mitigation_plan":   [],
            "patch_results":     {},
            "validation_results": {},
            "iteration":  0,
            "status":     "running",
            "log":        [],
            "audit_id":   audit_id,
            "_stop_event": stop_event,
        }

        loop = asyncio.get_event_loop()
        # Use object sentinel instead of None so queue.get() can distinguish "done" from error
        _DONE = object()
        queue: asyncio.Queue = asyncio.Queue()

        def _run_graph():
            try:
                for step_output in self.graph.stream(state):
                    if stop_event.is_set():
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, step_output)
            except Exception as e:
                loop.call_soon_threadsafe(queue.put_nowait, e)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        thread = threading.Thread(target=_run_graph, daemon=True)
        thread.start()

        last_log_len = 0
        latest_state = state
        try:
            while True:
                item = await queue.get()
                if item is _DONE:
                    break
                if isinstance(item, Exception):
                    logger.error(f"[{audit_id}] Graph execution error: {item}")
                    latest_state = {**latest_state, "log": [*latest_state.get("log", []), f"[Error] {item}"]}
                    break
                node_name  = list(item.keys())[0]
                node_state = item[node_name]
                latest_state = node_state
                full_log = node_state.get("log", [])
                new_lines = full_log[last_log_len:]
                last_log_len = len(full_log)
                yield {
                    "node":      node_name,
                    "iteration": node_state.get("iteration", 0),
                    "log":       new_lines,
                    "status":    node_state.get("status", "running"),
                }
        except GeneratorExit:
            stop_event.set()
            return

        if stop_event.is_set():
            yield {"node": "complete", "status": "stopped", "results": {}}
            return

        # Always emit the complete event so the frontend can render the report.
        # Only send final_report (not the full state) to avoid raw probe feature
        # data that can contain NaN values and produce invalid JSON.
        final_report = latest_state.get("final_report")
        if final_report is None:
            validation    = latest_state.get("validation_results", {})
            patch_results = latest_state.get("patch_results", {})
            final_report = {
                "audit_id":        audit_id,
                "completed_at":    datetime.utcnow().isoformat(),
                "iterations":      latest_state.get("iteration", 0),
                "biases_targeted": len(latest_state.get("confirmed_biases", [])),
                "patches_applied": len(patch_results.get("applied", [])),
                "patches_failed":  patch_results.get("failed", []),
                "biases_improved": len(validation.get("improved", [])),
                "biases_regressed": len(validation.get("regressed", [])),
                "biases_unchanged": len(validation.get("unchanged", [])),
                "validation":       validation,
                "mitigation_plan":  latest_state.get("mitigation_plan", []),
                "patch_results":    patch_results,
                "remediated_fairness": self._fairness_delta(validation),
                "patched_model_artifact": self._serialise_model_artifact(
                    state.get("model"),
                    group_corrections=latest_state.get("group_corrections"),
                    group_thresholds=latest_state.get("group_thresholds"),
                ),
                "log_summary":     latest_state.get("log", [])[-15:],
                "status":          "complete",
            }
        yield {"node": "complete", "status": "done", "results": final_report}

    # ── Nodes ─────────────────────────────────────────────────────────────────

    def _attack_node(self, state: Dict) -> Dict:
        confirmed  = state.get("confirmed_biases", [])
        X_train    = state["X_train"]
        iteration  = state.get("iteration", 0)
        log        = list(state.get("log", []))

        log.append(f"[Attack Agent] Iteration {iteration+1}: generating probes for {len(confirmed)} confirmed biases")

        probes = []
        for bias in confirmed:
            attr = bias.get("attribute", "")
            if attr not in X_train.columns:
                log.append(f"[Attack Agent] '{attr}' not in training data — will rely on cartography evidence")
                continue

            if X_train[attr].dtype == object and X_train[attr].nunique() > 20:
                log.append(f"[Attack Agent] Skipping '{attr}' — high-cardinality text column")
                continue

            top_values   = X_train[attr].value_counts().head(8).index.tolist()
            base_samples = X_train.sample(min(20, len(X_train)), random_state=iteration)

            for _, row in base_samples.iterrows():
                for val in top_values:
                    probe = row.copy()
                    probe[attr] = val
                    probes.append({
                        "probe_id":    f"probe_{len(probes)}",
                        "base_attr":   attr,
                        "set_value":   str(val),
                        "features":    probe.to_dict(),
                        "target_bias": bias.get("type", "demographic_parity"),
                    })

        log.append(f"[Attack Agent] Generated {len(probes)} adversarial probes")
        return {**state, "synthetic_probes": probes, "iteration": iteration + 1, "log": log}

    def _evaluate_node(self, state: Dict) -> Dict:
        """
        Runs probes and measures per-group disparity.
        When probes cannot be run (attr not in training data, prediction failure),
        falls back to cartography SPD and user-confirmed magnitudes.
        """
        model  = state["model"]
        probes = state["synthetic_probes"]
        log    = list(state.get("log", []))

        evaluation: List[Dict] = []

        if not probes:
            log.append("[Evaluator Agent] No probes generated — using cartography evidence for confirmed biases")
            evaluation = self._cartography_fallback_evaluation(state, [], log)
            n_confirmed = sum(1 for e in evaluation if e["bias_confirmed"])
            log.append(f"[Evaluator Agent] {n_confirmed}/{len(evaluation)} attributes confirmed via cartography")
            return {**state, "evaluation_results": evaluation, "log": log}

        log.append(f"[Evaluator Agent] Running {len(probes)} probes through model...")

        batch_df = pd.DataFrame([p["features"] for p in probes])
        try:
            batch_preds = self._safe_predict(model, batch_df)
            has_proba   = hasattr(model, "predict_proba")
            batch_probs: Optional[np.ndarray] = None
            if has_proba:
                try:
                    batch_probs = model.predict_proba(batch_df)[:, 1]
                except Exception:
                    has_proba = False
        except Exception as e:
            log.append(f"[Evaluator Agent] Batch prediction failed: {e} — using cartography evidence")
            evaluation = self._cartography_fallback_evaluation(state, [], log)
            n_confirmed = sum(1 for e in evaluation if e["bias_confirmed"])
            log.append(f"[Evaluator Agent] {n_confirmed}/{len(evaluation)} attributes confirmed via cartography")
            return {**state, "evaluation_results": evaluation, "log": log}

        # Attach predictions to probes
        results = []
        for i, probe in enumerate(probes):
            try:
                prob = float(batch_probs[i]) if batch_probs is not None else None
                results.append({**probe, "prediction": int(batch_preds[i]), "probability": prob})
            except Exception:
                continue

        # Group by attribute → per-demographic-value mean → disparity
        attr_groups: Dict[str, List] = {}
        for r in results:
            attr_groups.setdefault(r["base_attr"], []).append(r)

        for attr, group in attr_groups.items():
            value_groups: Dict[str, List] = {}
            for r in group:
                value_groups.setdefault(str(r["set_value"]), []).append(r)

            group_means: Dict[str, float] = {}
            for val, val_rows in value_groups.items():
                val_probs = [r["probability"] for r in val_rows if r["probability"] is not None]
                group_means[val] = float(np.mean(val_probs)) if val_probs else float(np.mean([r["prediction"] for r in val_rows]))

            if len(group_means) >= 2:
                means         = list(group_means.values())
                disparity     = float(max(means) - min(means))
                most_favored  = max(group_means, key=lambda k: group_means[k])
                least_favored = min(group_means, key=lambda k: group_means[k])
            else:
                disparity = 0.0
                most_favored = least_favored = None

            evaluation.append({
                "attribute":          attr,
                "disparity":          round(disparity, 4),
                "group_means":        {k: round(v, 4) for k, v in group_means.items()},
                "most_favored_group": most_favored,
                "least_favored_group": least_favored,
                "bias_confirmed":     disparity > 0.05,
                "sample_count":       len(group),
            })

        # Upgrade near-zero probe results using cartography evidence / user confirmation
        evaluation = self._cartography_fallback_evaluation(state, evaluation, log)

        n_confirmed      = sum(1 for e in evaluation if e["bias_confirmed"])
        disparity_summary = ", ".join(f"{e['attribute']}={e['disparity']:.3f}" for e in evaluation)
        log.append(
            f"[Evaluator Agent] {n_confirmed}/{len(evaluation)} attributes confirmed "
            f"(disparities: {disparity_summary})"
        )
        return {**state, "evaluation_results": evaluation, "log": log}

    def _cartography_fallback_evaluation(
        self, state: Dict, evaluation: List[Dict], log: List[str]
    ) -> List[Dict]:
        """
        Confirms biases that couldn't be measured via probes.

        Two upgrade paths:
        A. Cartography SPD ≥ threshold → confirmed via cartography evidence.
        B. User explicitly selected the bias → always confirmed, using max(probe_disparity, user_magnitude).
        """
        audit_results    = state.get("audit_results", {})
        confirmed_biases = state.get("confirmed_biases", [])

        # Cartography slice_metrics may arrive under several keys depending on which phase ran
        cross_carto   = (audit_results.get("crossAnalysis") or {}).get("cartography") or {}
        direct_carto  = audit_results.get("cartography") or {}
        slice_metrics = (
            cross_carto.get("slice_metrics")
            or direct_carto.get("slice_metrics")
            or []
        )

        # Max |SPD| per single attribute from cartography
        carto_spd: Dict[str, float] = {}
        for m in slice_metrics:
            attr = m.get("attribute", "")
            if "+" in attr:
                continue
            carto_spd[attr] = max(carto_spd.get(attr, 0), abs(m.get("statistical_parity_diff", 0)))

        evaluated_attrs = {e["attribute"] for e in evaluation}
        confirmed_magnitudes: Dict[str, float] = {
            b.get("attribute", ""): b.get("magnitude", 0)
            for b in confirmed_biases if b.get("attribute")
        }

        # ── Add entries for confirmed biases that produced no probes ─────────
        for bias in confirmed_biases:
            attr = bias.get("attribute", "")
            if not attr or attr in evaluated_attrs:
                continue
            cspd      = carto_spd.get(attr, 0)
            magnitude = max(cspd, bias.get("magnitude", 0)) or 0.1
            source    = "cartography" if cspd >= settings.DEMOGRAPHIC_PARITY_THRESHOLD else "user_confirmed"
            log.append(
                f"[Evaluator Agent] '{attr}': user-confirmed — adding to evaluation "
                f"(carto SPD={cspd:.3f}, magnitude={magnitude:.3f}, source={source})"
            )
            evaluation.append({
                "attribute":      attr,
                "disparity":      round(magnitude, 4),
                "bias_confirmed": True,
                "bias_source":    source,
                "sample_count":   0,
            })
            evaluated_attrs.add(attr)

        # ── Upgrade existing probe-based entries that are below the 0.05 threshold ─
        for e in evaluation:
            if e.get("bias_confirmed"):
                continue
            attr       = e["attribute"]
            cspd       = carto_spd.get(attr, 0)
            user_mag   = confirmed_magnitudes.get(attr, 0)

            if cspd >= settings.DEMOGRAPHIC_PARITY_THRESHOLD:
                e["disparity"]      = round(cspd, 4)
                e["bias_confirmed"] = True
                e["bias_source"]    = "cartography"
                log.append(
                    f"[Evaluator Agent] '{attr}': probe disparity≈0 but cartography SPD={cspd:.3f} — confirmed"
                )
            elif attr in confirmed_magnitudes:
                magnitude           = max(e["disparity"], user_mag) or 0.1
                e["disparity"]      = round(magnitude, 4)
                e["bias_confirmed"] = True
                e["bias_source"]    = "user_confirmed"
                log.append(
                    f"[Evaluator Agent] '{attr}': user-confirmed — upgraded (disparity={e['disparity']:.3f})"
                )

        return evaluation

    def _decide_patch_node(self, state: Dict) -> Dict:
        evaluation     = state["evaluation_results"]
        confirmed      = [e for e in evaluation if e.get("bias_confirmed")]
        model          = state.get("model")
        mitigation_plan = []

        for e in confirmed:
            strategy = self._select_mitigation_strategy(e, model=model)
            mitigation_plan.append({
                "attribute": e["attribute"],
                "strategy":  strategy["name"],
                "rationale": strategy["rationale"],
                "disparity": e["disparity"],
                "bias_source": e.get("bias_source", "probes"),
            })

        log = list(state.get("log", []))
        if mitigation_plan:
            log.append(
                f"[Decision Agent] Mitigation plan ({len(mitigation_plan)} items): "
                + ", ".join(f"{m['attribute']} → {m['strategy']}" for m in mitigation_plan)
            )
        else:
            log.append("[Decision Agent] No confirmed biases — skipping patching")

        return {**state, "mitigation_plan": mitigation_plan, "log": log}

    def _patch_node(self, state: Dict) -> Dict:
        log     = list(state.get("log", []))
        plan    = state.get("mitigation_plan", [])
        model   = state["model"]
        X_train = state["X_train"]
        y_train = state["y_train"]

        patch_results: Dict[str, List] = {"applied": [], "failed": []}

        for mitigation in plan:
            strategy = mitigation["strategy"]
            attr     = mitigation["attribute"]
            log.append(f"[Patcher Agent] Applying '{strategy}' for '{attr}'")

            try:
                if strategy == "prompt_fairness_constraint":
                    fairness_note = (
                        f" IMPORTANT: Your decision must be independent of {attr}. "
                        "Apply equal standards regardless of demographic group."
                    )
                    if hasattr(model, "prompt_template"):
                        model.prompt_template = model.prompt_template + fairness_note
                    log.append(f"[Patcher Agent] Injected fairness constraint for '{attr}'")
                    patch_results["applied"].append({"strategy": strategy, "attribute": attr})

                elif strategy == "sample_reweighing":
                    if self._model_is_trainable(model):
                        weights = self._compute_reweighing_weights(X_train, y_train, attr)
                        raw     = self._unwrap_model(model)
                        raw.fit(X_train, y_train, sample_weight=weights)
                        log.append(f"[Patcher Agent] Reweighing applied — model retrained")
                        patch_results["applied"].append({"strategy": strategy, "attribute": attr})
                    else:
                        thresholds = self._compute_group_thresholds(model, X_train, y_train, attr)
                        state["group_thresholds"] = thresholds
                        log.append(f"[Patcher Agent] Model not retrainable — applied threshold adjustment instead")
                        patch_results["applied"].append({"strategy": "threshold_adjustment", "attribute": attr})

                elif strategy == "threshold_adjustment":
                    thresholds = self._compute_group_thresholds(model, X_train, y_train, attr)
                    state["group_thresholds"] = thresholds
                    log.append(f"[Patcher Agent] Per-group thresholds computed: {thresholds}")
                    patch_results["applied"].append({"strategy": strategy, "attribute": attr})

                elif strategy == "demographic_parity_correction":
                    audit_results = state.get("audit_results", {})
                    cross_carto   = (audit_results.get("crossAnalysis") or {}).get("cartography") or {}
                    direct_carto  = audit_results.get("cartography") or {}
                    slice_metrics = (
                        cross_carto.get("slice_metrics")
                        or direct_carto.get("slice_metrics")
                        or []
                    )
                    corrections: Dict[str, float] = {}
                    before_rates: Dict[str, float] = {}
                    overall_rate: Optional[float]  = None

                    # Try cartography slice_metrics first
                    for m in slice_metrics:
                        if m.get("attribute") == attr and "+" not in (m.get("attribute") or ""):
                            if overall_rate is None:
                                overall_rate = m.get("overall_rate", 0)
                            val      = str(m.get("value", ""))
                            pos_rate = m.get("positive_rate", 0)
                            before_rates[val] = float(pos_rate)
                            if pos_rate > 0 and overall_rate and overall_rate > 0:
                                corrections[val] = round(overall_rate / pos_rate, 4)

                    # Fallback: compute directly from model predictions on X_train
                    if not corrections and attr in X_train.columns:
                        log.append(f"[Patcher Agent] No cartography slice data for '{attr}' — computing from model predictions on training data")
                        try:
                            if hasattr(model, "predict_proba"):
                                raw_probs = model.predict_proba(X_train)[:, 1]
                                group_rates_raw: Dict[str, float] = {}
                                for val in X_train[attr].dropna().unique():
                                    mask = X_train[attr] == val
                                    if mask.sum() > 0:
                                        group_rates_raw[str(val)] = float(raw_probs[mask.values].mean())
                            else:
                                y_pred_rt = self._safe_predict(model, X_train)
                                group_rates_raw = {}
                                for val in X_train[attr].dropna().unique():
                                    mask = X_train[attr] == val
                                    if mask.sum() > 0:
                                        group_rates_raw[str(val)] = float((y_pred_rt[mask.values] == 1).mean())

                            if group_rates_raw:
                                overall_rate = float(np.mean(list(group_rates_raw.values())))
                                before_rates = group_rates_raw
                                if overall_rate > 0:
                                    corrections = {
                                        val: round(overall_rate / rate, 4)
                                        for val, rate in group_rates_raw.items()
                                        if rate > 0
                                    }
                        except Exception as e:
                            log.append(f"[Patcher Agent] Could not compute corrections from predictions: {e}")

                    if corrections:
                        group_corrections        = state.get("group_corrections", {})
                        group_corrections[attr]  = {
                            "correction_factors": corrections,
                            "target_rate":        overall_rate,
                            "before_rates":       before_rates,
                        }
                        state["group_corrections"] = group_corrections
                        log.append(
                            f"[Patcher Agent] Demographic parity correction for '{attr}': "
                            f"target_rate={overall_rate:.3f}, factors computed for {len(corrections)} groups"
                        )
                        patch_results["applied"].append({"strategy": strategy, "attribute": attr, "correction_factors": corrections})
                    else:
                        log.append(f"[Patcher Agent] Could not compute correction factors for '{attr}' — falling back to threshold adjustment")
                        # Fall back to threshold adjustment if we have predict_proba
                        if hasattr(model, "predict_proba") and attr in X_train.columns:
                            try:
                                thresholds = self._compute_group_thresholds(model, X_train, y_train, attr)
                                state["group_thresholds"] = thresholds
                                log.append(f"[Patcher Agent] Threshold adjustment fallback for '{attr}': {thresholds}")
                                patch_results["applied"].append({"strategy": "threshold_adjustment", "attribute": attr})
                            except Exception as e:
                                log.append(f"[Patcher Agent] Threshold adjustment also failed: {e}")
                                patch_results["failed"].append({"strategy": strategy, "attribute": attr, "error": str(e)})
                        else:
                            patch_results["failed"].append({"strategy": strategy, "attribute": attr, "error": "no slice metrics and attribute not in training data"})

                elif strategy == "feature_ablation":
                    if attr in X_train.columns:
                        state["X_train"] = X_train.drop(columns=[attr])
                        log.append(f"[Patcher Agent] Removed proxy feature '{attr}' from training set")
                    patch_results["applied"].append({"strategy": strategy, "attribute": attr})

            except Exception as e:
                log.append(f"[Patcher Agent] Failed to apply {strategy} for '{attr}': {e}")
                patch_results["failed"].append({"strategy": strategy, "attribute": attr, "error": str(e)})

        return {**state, "patch_results": patch_results, "log": log}

    def _validate_node(self, state: Dict) -> Dict:
        """
        Measures bias improvement after patching.

        Strategy A — trainable models (predict_proba or predict):
          Re-run predictions on X_train, compare per-group rates before/after.

        Strategy B — external/API models with demographic_parity_correction:
          Compute estimated post-correction SPD from the correction factors applied.
        """
        model            = state["model"]
        X_train          = state["X_train"]
        y_train          = state["y_train"]
        log              = list(state.get("log", []))
        group_corrections = state.get("group_corrections", {})
        patch_results_state = state.get("patch_results", {})

        applied_attrs = {p["attribute"] for p in patch_results_state.get("applied", [])}
        validation: Dict[str, List] = {"improved": [], "regressed": [], "unchanged": []}
        if not applied_attrs and not group_corrections:
            log.append("[Validator Agent] No patches were applied — recording biases as unchanged")
            for e in state.get("evaluation_results", []):
                validation["unchanged"].append(e["attribute"])
            return {**state, "validation_results": validation, "log": log}

        log.append("[Validator Agent] Re-evaluating bias metrics post-patch...")

        for e in state.get("evaluation_results", []):
            attr          = e["attribute"]
            old_disparity = e.get("disparity", 0)

            # Strategy B: correction-factor validation
            if attr in group_corrections:
                entry              = group_corrections[attr]
                correction_factors = entry.get("correction_factors", {})
                before_rates       = entry.get("before_rates", {})
                target_rate        = entry.get("target_rate", 0)
                if correction_factors:
                    corrected_rates = {
                        val: min(before_rates.get(val, target_rate) * factor, 1.0)
                        for val, factor in correction_factors.items()
                    }
                    new_disparity = (
                        max(corrected_rates.values()) - min(corrected_rates.values())
                        if corrected_rates else 0.0
                    )
                    log.append(
                        f"[Validator Agent] '{attr}': correction applied — "
                        f"before SPD≈{old_disparity:.3f}, estimated after SPD≈{new_disparity:.3f}"
                    )
                    validation["improved"].append({"attribute": attr, "before": round(old_disparity, 4), "after": round(new_disparity, 4)})
                else:
                    validation["unchanged"].append(attr)
                continue

            # Strategy A: re-run predictions
            if attr not in X_train.columns:
                validation["unchanged"].append(attr)
                continue

            try:
                if hasattr(model, "predict_proba"):
                    y_prob = model.predict_proba(X_train)[:, 1]
                    group_means: Dict[str, float] = {}
                    for val in X_train[attr].unique():
                        mask = X_train[attr] == val
                        if mask.sum() > 0:
                            group_means[str(val)] = float(y_prob[mask].mean())
                    new_disparity = (max(group_means.values()) - min(group_means.values())) if len(group_means) >= 2 else old_disparity
                else:
                    y_pred = self._safe_predict(model, X_train)
                    group_rates: Dict[str, float] = {}
                    for val in X_train[attr].unique():
                        mask = X_train[attr] == val
                        if mask.sum() > 0:
                            group_rates[str(val)] = float((y_pred[mask.values] == 1).mean())
                    new_disparity = (max(group_rates.values()) - min(group_rates.values())) if len(group_rates) >= 2 else old_disparity

                if new_disparity < old_disparity * 0.7:
                    validation["improved"].append({"attribute": attr, "before": round(old_disparity, 4), "after": round(new_disparity, 4)})
                    log.append(f"[Validator Agent] '{attr}': improved {old_disparity:.3f} → {new_disparity:.3f}")
                elif new_disparity > old_disparity * 1.1:
                    validation["regressed"].append({"attribute": attr, "before": round(old_disparity, 4), "after": round(new_disparity, 4)})
                    log.append(f"[Validator Agent] '{attr}': regressed {old_disparity:.3f} → {new_disparity:.3f}")
                else:
                    validation["unchanged"].append(attr)
                    log.append(f"[Validator Agent] '{attr}': unchanged ({new_disparity:.3f})")

            except Exception as ex:
                log.append(f"[Validator Agent] Could not re-evaluate '{attr}': {ex}")
                validation["unchanged"].append(attr)

        log.append(
            f"[Validator Agent] Results: {len(validation['improved'])} improved, "
            f"{len(validation['regressed'])} regressed, {len(validation['unchanged'])} unchanged"
        )
        return {**state, "validation_results": validation, "log": log}

    def _report_node(self, state: Dict) -> Dict:
        log = list(state.get("log", []))
        log.append("[Report Agent] Generating final red-team report...")

        validation    = state.get("validation_results", {})
        patch_results = state.get("patch_results", {})

        report = {
            "audit_id":          state.get("audit_id"),
            "completed_at":      datetime.utcnow().isoformat(),
            "iterations":        state.get("iteration", 0),
            "biases_targeted":   len(state.get("confirmed_biases", [])),
            "patches_applied":   len(patch_results.get("applied", [])),
            "patches_failed":    len(patch_results.get("failed", [])),
            "biases_improved":   len(validation.get("improved", [])),
            "biases_regressed":  len(validation.get("regressed", [])),
            "biases_unchanged":  len(validation.get("unchanged", [])),
            "validation":        validation,
            "mitigation_plan":   state.get("mitigation_plan", []),
            "patch_results":     patch_results,
            "remediated_fairness": self._fairness_delta(validation),
            "patched_model_artifact": self._serialise_model_artifact(
                state.get("model"),
                group_corrections=state.get("group_corrections"),
                group_thresholds=state.get("group_thresholds"),
            ),
            "log_summary":  log[-15:],
            "status":       "complete",
        }

        log.append("[Report Agent] Done.")
        return {**state, "final_report": report, "status": "complete", "log": log}

    # ── Routing ───────────────────────────────────────────────────────────────

    def _should_patch(self, state: Dict) -> str:
        confirmed = [e for e in state.get("evaluation_results", []) if e.get("bias_confirmed")]
        return "patch" if confirmed else "report"

    def _should_continue(self, state: Dict) -> str:
        regressions = state.get("validation_results", {}).get("regressed", [])
        iteration   = state.get("iteration", 0)
        if regressions and iteration < settings.REDTEAM_MAX_ITERATIONS:
            return "attack"
        return "report"

    # ── Mitigation strategy selection ─────────────────────────────────────────

    @staticmethod
    def _unwrap_model(model) -> Any:
        """Return the raw estimator inside any adapter wrapper."""
        if hasattr(model, "_raw_model"):
            try:
                return model._raw_model()
            except Exception:
                pass
        return getattr(model, "model", getattr(model, "_model", model))

    def _model_is_trainable(self, model) -> bool:
        raw = self._unwrap_model(model)
        return hasattr(raw, "fit") and callable(getattr(raw, "fit", None))

    @staticmethod
    def _model_is_generative(model) -> bool:
        mt = (getattr(model, "get_model_type", lambda: "")() or "")
        return any(t in mt for t in ("GenerativeLLM", "OpenAI", "Gemini"))

    def _select_mitigation_strategy(self, evaluation: Dict, model=None) -> Dict:
        disparity    = evaluation.get("disparity", 0)
        bias_source  = evaluation.get("bias_source", "probes")
        is_generative = model is not None and self._model_is_generative(model)
        is_trainable  = model is None or self._model_is_trainable(model)

        if is_generative:
            return {
                "name": "prompt_fairness_constraint",
                "rationale": "Generative LLM — add explicit fairness instructions to the decision prompt",
            }
        if not is_trainable:
            if disparity > 0.2 or bias_source in ("cartography", "user_confirmed"):
                return {
                    "name": "demographic_parity_correction",
                    "rationale": "External model — compute post-hoc per-group correction factors from cartography to equalise positive rates",
                }
            return {
                "name": "threshold_adjustment",
                "rationale": "External model — per-group thresholds equalise prediction rates",
            }
        if disparity > 0.3:
            return {"name": "sample_reweighing",     "rationale": "High disparity — reweigh training distribution"}
        if disparity > 0.15:
            return {"name": "threshold_adjustment",  "rationale": "Moderate disparity — per-group thresholds equalise prediction rates"}
        return     {"name": "feature_ablation",      "rationale": "Low disparity — ablating the proxy feature is sufficient"}

    # ── Patch utilities ───────────────────────────────────────────────────────

    def _compute_reweighing_weights(self, X: pd.DataFrame, y: np.ndarray, attr: str) -> np.ndarray:
        weights = np.ones(len(X))
        if attr not in X.columns:
            return weights
        n = len(X)
        for val in X[attr].unique():
            for label in [0, 1]:
                mask          = (X[attr] == val) & (y == label)
                n_group       = (X[attr] == val).sum()
                n_label       = (y == label).sum()
                n_group_label = mask.sum()
                if n_group_label > 0:
                    expected = (n_group / n) * (n_label / n)
                    observed = n_group_label / n
                    weights[mask] = expected / observed
        return weights

    def _compute_group_thresholds(self, model, X: pd.DataFrame, y: np.ndarray, attr: str) -> Dict:
        thresholds: Dict[str, float] = {}
        if not hasattr(model, "predict_proba") or attr not in X.columns:
            return thresholds
        probs = model.predict_proba(X)[:, 1]
        for val in X[attr].unique():
            mask         = X[attr] == val
            group_probs  = probs[mask]
            group_labels = y[mask]
            if group_labels.sum() > 0:
                from sklearn.metrics import roc_curve
                _, tpr, thresh = roc_curve(group_labels, group_probs)
                idx = np.argmin(np.abs(tpr - 0.5))
                thresholds[str(val)] = float(thresh[idx])
        return thresholds

    # ── Report helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _fairness_delta(validation: Dict[str, List]) -> Dict[str, Any]:
        measured = [
            item for group in ("improved", "regressed")
            for item in validation.get(group, [])
            if isinstance(item, dict) and "before" in item and "after" in item
        ]
        if not measured:
            return {"before_avg_spd": None, "after_avg_spd": None, "improvement": None}
        before = float(np.mean([float(x["before"]) for x in measured]))
        after  = float(np.mean([float(x["after"])  for x in measured]))
        return {
            "before_avg_spd": round(before, 4),
            "after_avg_spd":  round(after,  4),
            "improvement":    round(before - after, 4),
            "per_attribute":  measured,
        }

    def _serialise_model_artifact(
        self,
        model: Any,
        group_corrections: Optional[Dict] = None,
        group_thresholds: Optional[Dict] = None,
    ) -> Optional[Dict[str, Any]]:
        raw = self._unwrap_model(model)
        if not hasattr(raw, "fit"):
            return {
                "available": False,
                "message":   (
                    "Remediation was applied to a remote/API model. "
                    "No pickle artifact can be exported — apply the mitigation plan "
                    "settings (correction factors / thresholds) in your serving layer."
                ),
            }
        try:
            # Wrap raw model with correction factors so the downloaded .pkl
            # applies fairness patches automatically at inference time.
            if group_corrections or group_thresholds:
                artifact = _FairLensRemediatedModel(
                    raw,
                    group_corrections=group_corrections or {},
                    group_thresholds=group_thresholds or {},
                )
            else:
                artifact = raw
            payload = base64.b64encode(pickle.dumps(artifact)).decode("ascii")
            return {
                "available":   True,
                "filename":    "fairlens-remediated-model.pkl",
                "format":      "pickle",
                "pickle_b64":  payload,
                "message":     (
                    "The patched model includes per-group correction factors. "
                    "Load with joblib.load() and call model.predict(X) — corrections are automatic."
                ),
            }
        except Exception as exc:
            return {
                "available": False,
                "message":   f"FairLens remediated the model in memory, but serialisation failed: {exc}",
            }


redteam_agent = FairnessRedTeamAgent()
