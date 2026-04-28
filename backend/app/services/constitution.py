"""
Counterfactual Constitution Service
=====================================
Uses Gemini to generate a structured 'constitution' document that captures
what the model would have decided if demographic attributes were different.

Key innovation:
- Not just individual counterfactuals, but a VERSIONED DOCUMENT that captures
  the model's implicit decision policy across demographic axes
- Diffs the constitution across model versions to detect fairness drift
- Human-readable enough for legal/HR teams, precise enough for engineers
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
import json
from datetime import datetime

from app.services.gemini_client import ask_gemini
from app.core.config import settings

logger = logging.getLogger(__name__)


class CounterfactualConstitutionService:
    """
    Generates the Counterfactual Constitution via Gemini.
    The constitution answers: "What implicit rules is this model following,
    and how do those rules change when demographics change?"
    Uses the central gemini_client (Vertex AI via ADC).
    """

    async def generate_constitution(
        self,
        model: Optional[Any],
        X: pd.DataFrame,
        y_pred: np.ndarray,
        protected_cols: List[str],
        feature_names: List[str],
        cartography_results: Dict,
        audit_id: str,
    ) -> Dict[str, Any]:
        """
        Full pipeline:
        1. Generate counterfactual pairs for each protected attribute (skipped if no model)
        2. Aggregate patterns into policy statements
        3. Ask Gemini to synthesise into a structured Constitution document
        4. Return structured JSON + human-readable Markdown
        """
        logger.info(f"[{audit_id}] Generating Counterfactual Constitution (model={'provided' if model else 'none'})")

        # API-based models are slow — limit CF samples to avoid blocking the event loop.
        # HF classifiers: ~0.3s/call × many samples × attrs × flips adds up fast.
        # LLMs are even slower. Local sklearn models can handle 150 comfortably.
        model_type_str = (model.get_model_type() if hasattr(model, "get_model_type") else "") or ""
        is_generative = "GenerativeLLM" in model_type_str
        is_api_model = is_generative or any(t in model_type_str for t in ("HuggingFace", "REST:"))
        cf_n_samples = 10 if is_generative else 20 if is_api_model else 150

        # Step 1: Build counterfactual pairs in a thread pool so the async event loop
        # stays responsive (Cloud Run health checks keep succeeding while this runs).
        loop = asyncio.get_event_loop()
        try:
            cf_pairs = await asyncio.wait_for(
                loop.run_in_executor(
                    None,
                    lambda: self._generate_cf_pairs(model, X, y_pred, protected_cols, n_samples=cf_n_samples),
                ),
                timeout=75.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[{audit_id}] CF pair generation timed out — continuing with empty pairs")
            cf_pairs = []

        # Step 2: Extract decision patterns
        patterns = self._extract_patterns(cf_pairs, protected_cols)

        # Step 3: Gemini synthesis (capped so the whole request stays well under Cloud Run limits)
        try:
            constitution_text = await asyncio.wait_for(
                self._gemini_synthesise(
                    patterns, cf_pairs, cartography_results, protected_cols, feature_names, audit_id,
                    model_available=(model is not None),
                    model=model,
                ),
                timeout=60.0,
            )
        except asyncio.TimeoutError:
            logger.warning(f"[{audit_id}] Gemini constitution synthesis timed out — using fallback")
            constitution_text = "## Constitution\n\nConstitution synthesis timed out. Bias patterns are available in the cartography results."

        # Step 4: Parse into structured sections
        sections = self._parse_constitution(constitution_text)

        result = {
            "audit_id": audit_id,
            "generated_at": datetime.utcnow().isoformat(),
            "constitution_markdown": constitution_text,
            "sections": sections,
            "counterfactual_pairs": cf_pairs[:50],  # sample for frontend
            "patterns": patterns,
            "summary": {
                "total_cf_pairs": len(cf_pairs),
                "decision_flips": sum(1 for p in cf_pairs if p["decision_flipped"]),
                "flip_rate": round(sum(1 for p in cf_pairs if p["decision_flipped"]) / max(len(cf_pairs), 1), 3),
                "most_sensitive_attribute": max(
                    protected_cols,
                    key=lambda c: sum(1 for p in cf_pairs if p["changed_attr"] == c and p["decision_flipped"]),
                    default=None
                ) if protected_cols else None,
            }
        }

        logger.info(f"[{audit_id}] Constitution generated. Flip rate: {result['summary']['flip_rate']:.1%}")
        return result

    def _generate_cf_pairs(
        self, model: Optional[Any], X: pd.DataFrame, y_pred: np.ndarray,
        protected_cols: List[str], n_samples: int = 200
    ) -> List[Dict]:
        """
        For each sample, flip each protected attribute to all other observed values,
        re-predict, and record whether the decision changed.
        Returns empty list when no model is available.
        """
        if model is None:
            return []

        pairs = []
        present_cols = [c for c in protected_cols if c in X.columns]
        indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)

        for idx in indices:
            if idx >= len(y_pred):
                continue  # guard against predict returning fewer rows than X
            row = X.iloc[idx].copy()
            original_pred = y_pred[idx]

            for col in present_cols:
                original_val = row[col]
                other_vals = [v for v in X[col].unique() if v != original_val][:3]
                if not other_vals:
                    continue

                # Batch original + all counterfactuals so the encoder sees ALL values
                # for this column at once — prevents fresh single-row LabelEncoders
                # from mapping every value to 0 (making flips invisible to the model).
                row_dict = row.to_dict()
                cf_rows = []
                for v in other_vals:
                    cf = row_dict.copy()
                    cf[col] = v
                    cf_rows.append(cf)
                batch_df = pd.DataFrame([row_dict] + cf_rows)

                try:
                    batch_preds = model.predict(batch_df)
                    has_proba = hasattr(model, "predict_proba")
                    batch_probs = model.predict_proba(batch_df)[:, 1] if has_proba else None
                except Exception:
                    continue

                orig_prob = float(batch_probs[0]) if batch_probs is not None else None
                for i, alt_val in enumerate(other_vals):
                    cf_pred = batch_preds[i + 1]
                    cf_prob = float(batch_probs[i + 1]) if batch_probs is not None else None
                    pairs.append({
                        "sample_idx": int(idx),
                        "changed_attr": col,
                        "original_value": str(original_val),
                        "counterfactual_value": str(alt_val),
                        "original_prediction": int(original_pred),
                        "counterfactual_prediction": int(cf_pred),
                        "original_prob": orig_prob,
                        "counterfactual_prob": cf_prob,
                        "decision_flipped": int(original_pred) != int(cf_pred),
                        "prob_delta": float(cf_prob - orig_prob) if cf_prob is not None and orig_prob is not None else None,
                    })

        return pairs

    def _extract_patterns(self, cf_pairs: List[Dict], protected_cols: List[str]) -> List[Dict]:
        """Aggregate counterfactual pairs into pattern statements."""
        patterns = []
        for col in protected_cols:
            col_pairs = [p for p in cf_pairs if p["changed_attr"] == col]
            if not col_pairs:
                continue

            flip_rate = sum(1 for p in col_pairs if p["decision_flipped"]) / max(len(col_pairs), 1)
            avg_prob_delta = np.mean([abs(p["prob_delta"]) for p in col_pairs if p["prob_delta"] is not None])

            # Direction of bias: which value benefits?
            value_flip_rates = {}
            for val in set(p["original_value"] for p in col_pairs):
                val_pairs = [p for p in col_pairs if p["original_value"] == val]
                if val_pairs:
                    value_flip_rates[val] = sum(1 for p in val_pairs if p["decision_flipped"]) / len(val_pairs)

            patterns.append({
                "attribute": col,
                "flip_rate": round(flip_rate, 3),
                "avg_probability_shift": round(float(avg_prob_delta), 3) if not np.isnan(avg_prob_delta) else 0,
                "flip_rate_by_value": value_flip_rates,
                "bias_direction": max(value_flip_rates, key=value_flip_rates.get) if value_flip_rates else "unknown",
                "severity": "critical" if flip_rate > 0.3 else "high" if flip_rate > 0.15 else "medium" if flip_rate > 0.05 else "low",
            })

        return sorted(patterns, key=lambda p: p["flip_rate"], reverse=True)

    async def _gemini_synthesise(
        self,
        patterns: List[Dict],
        cf_pairs: List[Dict],
        cartography_results: Dict,
        protected_cols: List[str],
        feature_names: List[str],
        audit_id: str,
        model_available: bool = True,
        model: Optional[Any] = None,
    ) -> str:
        """Use Gemini to synthesise patterns + counterfactuals into a Constitution document."""
        hotspots_summary = json.dumps(
            cartography_results.get("hotspots", [])[:3], indent=2
        )
        patterns_summary = json.dumps(patterns, indent=2) if patterns else "No counterfactual flip patterns detected — the model may be insensitive to the protected attributes, or all samples had the same attribute value."
        flip_examples = json.dumps(
            [p for p in cf_pairs if p.get("decision_flipped")][:10], indent=2
        ) if cf_pairs else "No decision flips detected across counterfactual pairs."

        if not model_available:
            model_note = "\nNOTE: No ML model was provided. Analysis is based on dataset statistics and the bias topology map only. Counterfactual simulation is not available.\n"
        else:
            model_name = getattr(model, "_name", None) or getattr(model, "get_model_type", lambda: "")()
            if "AutoReference" in str(model_name):
                model_note = "\nNOTE: No user-provided model was uploaded. FairLens auto-trained a Logistic Regression reference model on this dataset to enable counterfactual simulation. The counterfactual results reveal data-level bias — the implicit patterns any model would learn from this training data.\n"
            else:
                model_note = ""

        prompt = f"""You are an AI fairness auditor generating a Counterfactual Constitution — a structured
document revealing the implicit rules an AI model follows with respect to demographic attributes.
Write so both a non-technical HR manager and a data scientist can read and act on it.

AUDIT ID: {audit_id}
PROTECTED ATTRIBUTES: {', '.join(protected_cols)}
MODEL FEATURES: {', '.join(feature_names[:20])}
{model_note}
═══ EVIDENCE ═══
COUNTERFACTUAL PATTERNS (flip rate = how often changing only this demographic changes the outcome):
{patterns_summary}

DECISION FLIP EXAMPLES (real cases where the same person, different demographic, got a different outcome):
{flip_examples}

BIAS HOTSPOTS (from statistical topology map):
{hotspots_summary}

═══ REQUIRED OUTPUT ═══
Write exactly these 7 sections in Markdown. Use **bold** for key numbers. Be specific — cite the numbers above.

## 1. Executive Summary
2-3 sentences. Name the most affected group and the flip rate. State the real-world consequence (e.g. "loan denied", "job application rejected"). Lead with impact, not methodology.

## 2. Implicit Decision Rules
List 3-6 rules the model appears to follow. Format each as a blockquote:

> **Rule N:** IF [demographic condition] THEN [outcome effect] — flip rate: X%, avg probability shift: ±Y%

## 3. Demographic Sensitivity Index

| Attribute | Flip Rate | Avg Prob Shift | Severity | Real-World Meaning |
|-----------|-----------|----------------|----------|--------------------|
(one row per protected attribute from the patterns data — fill in all columns)

Add 1-2 sentences per attribute explaining what the flip means in practice for a real person.

## 4. Most Affected Groups
The top 3 most disadvantaged groups, ranked by impact. For each: group name, how much worse their outcomes are, and one concrete example of what someone in this group experiences differently.

## 5. Structural vs. Proxy Bias
One paragraph: is the bias direct (model uses demographics explicitly) or through proxies (job title, zip code, biography length, etc.)? Name the likely proxy features from the model's feature list if evident.

## 6. Legal & Compliance Risk

| Framework | Threshold | Finding | Status |
|-----------|-----------|---------|--------|
| US EEOC 4/5ths Rule | DI ≥ 0.80 | [value from data] | ✅ Pass / ⚠️ Borderline / ❌ Fail |
| EU AI Act (High-Risk Systems) | SPD ≤ 0.10 | [value from data] | ... |
| UK Equality Act 2010 | No numeric threshold | [qualitative finding] | ... |

## 7. Remediation Priority
Numbered list of specific, actionable fixes in priority order (most urgent first). For each fix: what to change, why it matters most, and how to verify the bias is gone after the fix."""

        return await ask_gemini(prompt)

    def _parse_constitution(self, markdown_text: str) -> List[Dict]:
        """Parse markdown sections into structured JSON."""
        sections = []
        current_section = None

        for line in markdown_text.split("\n"):
            if line.startswith("## "):
                if current_section:
                    sections.append(current_section)
                current_section = {
                    "title": line.replace("## ", "").strip(),
                    "content": ""
                }
            elif current_section:
                current_section["content"] += line + "\n"

        if current_section:
            sections.append(current_section)

        return sections


constitution_service = CounterfactualConstitutionService()
