"""
Bias Cartography Service — Cloud-Native (Gemini-powered)
=========================================================
No SHAP. No UMAP.

1. Computes statistical bias metrics per demographic slice
2. Identifies intersectional bias patterns
3. Generates 2D topology coordinates via bias score + prediction scatter
4. Returns hotspot clusters with plain-English explanations via Gemini
"""

import pandas as pd
import numpy as np
import json
import logging
from typing import Dict, List, Any, Optional

from app.services.gemini_client import ask_gemini_json
from app.core.config import settings

logger = logging.getLogger(__name__)


class BiasCartographyService:

    async def run_cartography(
        self,
        dataset_csv: str,
        protected_cols: List[str],
        target_col: str,
        model_predictions: Optional[List] = None,
        audit_id: str = "",
    ) -> Dict[str, Any]:

        logger.info(f"[{audit_id}] Starting cloud Bias Cartography")

        import io
        df = pd.read_csv(io.StringIO(dataset_csv))
        sample_df = df.sample(min(300, len(df)), random_state=42)

        slice_metrics = self._compute_slice_metrics(df, protected_cols, target_col, model_predictions)
        gemini_analysis = await self._gemini_analyse(sample_df, protected_cols, target_col, slice_metrics, audit_id, model_predictions is not None)
        map_points = self._generate_map_points(df, protected_cols, target_col, slice_metrics, model_predictions)
        hotspots = self._identify_hotspots(slice_metrics)
        fair_score = self.compute_fair_score(slice_metrics)

        # Bootstrap CI for first protected column (representative, fast)
        ci_data: Dict[str, Any] = {}
        present_cols = [c for c in protected_cols if c in df.columns]
        if present_cols:
            ci_data = self.bootstrap_metric_ci(df, model_predictions, present_cols[0], target_col)

        return {
            "audit_id": audit_id,
            "map_points": map_points,
            "hotspots": hotspots,
            "slice_metrics": slice_metrics,
            "gemini_analysis": gemini_analysis,
            "fair_score": fair_score,
            "metric_confidence_intervals": {present_cols[0]: ci_data} if ci_data else {},
            "summary": {
                "total_samples": len(df),
                "hotspot_count": len(hotspots),
                "protected_cols_found": [c for c in protected_cols if c in df.columns],
                "overall_bias_score": round(
                    np.mean([abs(m["statistical_parity_diff"]) for m in slice_metrics]) if slice_metrics else 0, 3
                ),
                "most_biased_slice": slice_metrics[0]["label"] if slice_metrics else None,
            }
        }

    def _compute_slice_metrics(self, df, protected_cols, target_col, model_predictions=None):
        present = [c for c in protected_cols if c in df.columns]
        if not present or target_col not in df.columns:
            return []

        # y_pred: what we measure positive rates on (model output or ground truth)
        if model_predictions is not None and len(model_predictions) == len(df):
            target = pd.Series(model_predictions, index=df.index, dtype=float)
        else:
            target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)

        # y_true: ground truth labels — needed for EOD/EqODD (only when predictions differ)
        has_eod = model_predictions is not None and len(model_predictions) == len(df)
        if has_eod:
            y_true = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
            y_true_bin = (y_true > 0.5).astype(int)
            y_pred_bin = (target > 0.5).astype(int)

        overall_rate = float(target.mean())
        metrics = []

        for col in present:
            # Skip free-text columns — individual value slices are noise, not signal
            if df[col].dtype == object and df[col].nunique() > 20:
                continue
            for val in df[col].dropna().unique():
                mask = df[col] == val
                if mask.sum() < 5:
                    continue
                group_rate = float(target[mask].mean())
                spd = round(group_rate - overall_rate, 4)
                di = round(group_rate / overall_rate, 4) if overall_rate > 0 else None

                # Equal Opportunity Difference and Equalized Odds Difference
                eod = None
                eq_odds = None
                if has_eod:
                    eod, eq_odds = self._compute_eod(y_true_bin, y_pred_bin, mask)

                flagged = (
                    abs(spd) > settings.DEMOGRAPHIC_PARITY_THRESHOLD
                    or (di is not None and di < settings.DISPARATE_IMPACT_THRESHOLD)
                    or (eod is not None and abs(eod) > settings.EQUAL_OPPORTUNITY_THRESHOLD)
                    or (eq_odds is not None and eq_odds > settings.EQUALIZED_ODDS_THRESHOLD)
                )
                metrics.append({
                    "label": f"{col}={val}", "attribute": col, "value": str(val),
                    "size": int(mask.sum()), "positive_rate": round(group_rate, 4),
                    "overall_rate": round(overall_rate, 4),
                    "statistical_parity_diff": spd, "disparate_impact": di,
                    "equal_opportunity_diff": eod,
                    "equalized_odds_diff": eq_odds,
                    "bias_magnitude": round(abs(spd), 4),
                    "flagged": flagged,
                })

        if len(present) >= 2:
            for i, c1 in enumerate(present):
                for c2 in present[i+1:]:
                    # Skip intersectional slices if either column is high-cardinality text
                    if (df[c1].dtype == object and df[c1].nunique() > 20) or \
                       (df[c2].dtype == object and df[c2].nunique() > 20):
                        continue
                    for v1 in df[c1].dropna().unique()[:4]:
                        for v2 in df[c2].dropna().unique()[:4]:
                            mask = (df[c1] == v1) & (df[c2] == v2)
                            if mask.sum() < 5:
                                continue
                            group_rate = float(target[mask].mean())
                            spd = round(group_rate - overall_rate, 4)
                            di = round(group_rate / overall_rate, 4) if overall_rate > 0 else None
                            eod = None
                            eq_odds = None
                            if has_eod:
                                eod, eq_odds = self._compute_eod(y_true_bin, y_pred_bin, mask)
                            flagged = (
                                abs(spd) > settings.DEMOGRAPHIC_PARITY_THRESHOLD
                                or (di is not None and di < settings.DISPARATE_IMPACT_THRESHOLD)
                                or (eod is not None and abs(eod) > settings.EQUAL_OPPORTUNITY_THRESHOLD)
                                or (eq_odds is not None and eq_odds > settings.EQUALIZED_ODDS_THRESHOLD)
                            )
                            metrics.append({
                                "label": f"{c1}={v1} \u2229 {c2}={v2}", "attribute": f"{c1}+{c2}",
                                "value": f"{v1}+{v2}", "size": int(mask.sum()),
                                "positive_rate": round(group_rate, 4), "overall_rate": round(overall_rate, 4),
                                "statistical_parity_diff": spd, "disparate_impact": di,
                                "equal_opportunity_diff": eod,
                                "equalized_odds_diff": eq_odds,
                                "bias_magnitude": round(abs(spd), 4),
                                "flagged": flagged,
                            })
        return sorted(metrics, key=lambda m: m["bias_magnitude"], reverse=True)

    def _compute_eod(
        self,
        y_true: "pd.Series",
        y_pred: "pd.Series",
        mask: "pd.Series",
    ):
        """
        Returns (equal_opportunity_diff, equalized_odds_diff) for a demographic slice.

        Equal Opportunity Diff = TPR_group - TPR_overall
        Equalized Odds Diff    = max(|TPR diff|, |FPR diff|)

        Returns (None, None) when a denominator is zero (no positives or negatives overall).
        """
        # Overall rates
        pos_overall = y_true == 1
        neg_overall = y_true == 0

        tpr_overall = float(y_pred[pos_overall].mean()) if pos_overall.sum() > 0 else None
        fpr_overall = float(y_pred[neg_overall].mean()) if neg_overall.sum() > 0 else None

        if tpr_overall is None:
            return None, None

        # Group rates
        pos_group = pos_overall & mask
        neg_group = neg_overall & mask

        if pos_group.sum() < 2:
            return None, None

        tpr_group = float(y_pred[pos_group].mean())
        eod = round(tpr_group - tpr_overall, 4)

        eq_odds = None
        if fpr_overall is not None and neg_group.sum() >= 2:
            fpr_group = float(y_pred[neg_group].mean())
            eq_odds = round(max(abs(eod), abs(fpr_group - fpr_overall)), 4)

        return eod, eq_odds

    async def _gemini_analyse(self, df, protected_cols, target_col, slice_metrics, audit_id, using_model_predictions=False):
        top_slices = json.dumps(slice_metrics[:10], indent=2)
        # Convert numpy keys/values to Python-native types to avoid json.dumps TypeError
        col_summary = {
            col: {str(k): int(v) for k, v in df[col].value_counts().head(8).items()}
            for col in protected_cols if col in df.columns
        }
        analysis_source = "model prediction outputs (what the uploaded model actually decides)" if using_model_predictions else "dataset ground-truth labels"
        prompt = f"""You are an AI fairness auditor analysing a model for bias.
ANALYSIS SOURCE: {analysis_source}
DATASET: {len(df)} rows, target='{target_col}', protected={protected_cols}
DISTRIBUTIONS: {json.dumps(col_summary)}
TOP BIAS FINDINGS: {top_slices}
Return ONLY this JSON:
{{"severity":"critical|high|medium|low","headline":"one sentence","key_findings":["f1","f2","f3"],"most_affected_group":"group","bias_type":"direct|proxy|intersectional|systemic","real_world_impact":"impact","legal_risk":"risk","recommended_action":"action"}}"""
        try:
            return await ask_gemini_json(prompt)
        except Exception as e:
            logger.warning(f"Gemini analysis failed: {e}")
            return {"severity": "unknown", "headline": "Analysis unavailable", "key_findings": []}

    def _generate_map_points(self, df, protected_cols, target_col, slice_metrics, model_predictions=None):
        """
        Returns one point per demographic slice.
        x = statistical_parity_diff (signed: negative = disadvantaged group)
        y = positive_rate
        Single-attribute slices are full-size; intersectional slices are smaller.
        """
        if not slice_metrics:
            return []

        seen: dict = {}
        for m in slice_metrics:
            lbl = m["label"]
            if lbl not in seen or m["bias_magnitude"] > seen[lbl]["bias_magnitude"]:
                seen[lbl] = m

        points = []
        for m in seen.values():
            is_intersectional = "∩" in m["label"]
            points.append({
                "x":              round(float(m["statistical_parity_diff"]), 4),
                "y":              round(float(m["positive_rate"]), 4),
                "bias_score":     round(float(m["bias_magnitude"]), 4),
                "slice_label":    m["label"],
                "attribute":      m.get("attribute", ""),
                "size":           m["size"],
                "flagged":        bool(m.get("flagged", False)),
                "overall_rate":   round(float(m.get("overall_rate", 0)), 4),
                "intersectional": is_intersectional,
                "disparate_impact": m.get("disparate_impact"),
                "eod":            m.get("equal_opportunity_diff"),
                "eq_odds":        m.get("equalized_odds_diff"),
            })
        return points

    def _identify_hotspots(self, slice_metrics):
        """
        Returns flagged slices as hotspot records.
        centroid_x = signed SPD (matches the map x-axis).
        centroid_y = positive_rate (matches the map y-axis).
        """
        flagged = [m for m in slice_metrics if m.get("flagged")]
        hotspots = []
        for i, m in enumerate(flagged[:12]):
            severity = (
                "critical" if m["bias_magnitude"] > 0.3
                else "high"  if m["bias_magnitude"] > 0.15
                else "medium"
            )
            hotspots.append({
                "cluster_id":          i,
                # signed SPD so the ring sits on the correct x position in the scatter
                "centroid_x":          m["statistical_parity_diff"],
                "centroid_y":          m["positive_rate"],
                "size":                m["size"],
                "mean_bias_magnitude": m["bias_magnitude"],
                "dominant_slice":      m["label"],
                "attribute":           m.get("attribute", ""),
                "severity":            severity,
                "statistical_parity_diff": m["statistical_parity_diff"],
                "disparate_impact":    m.get("disparate_impact"),
                "equal_opportunity_diff": m.get("equal_opportunity_diff"),
                "equalized_odds_diff": m.get("equalized_odds_diff"),
                "overall_rate":        m.get("overall_rate", 0),
                "intersectional":      "∩" in m["label"],
            })
        return hotspots

    def compute_fair_score(self, slice_metrics: list) -> dict:
        """Composite fairness score 0–100. Higher = fairer."""
        if not slice_metrics:
            return {"score": 100, "label": "Fair", "color": "green"}

        single = [m for m in slice_metrics if "\u2229" not in m["label"]]
        avg_spd = float(np.mean([abs(m["statistical_parity_diff"]) for m in single])) if single else 0.0
        di_penalties = [
            max(0, settings.DISPARATE_IMPACT_THRESHOLD - m["disparate_impact"])
            for m in single if m.get("disparate_impact") is not None
        ]
        avg_di_penalty = float(np.mean(di_penalties)) if di_penalties else 0.0
        flagged_ratio = sum(1 for m in single if m.get("flagged")) / max(len(single), 1)

        # Penalise for Equal Opportunity and Equalized Odds violations
        eod_values = [abs(m["equal_opportunity_diff"]) for m in single if m.get("equal_opportunity_diff") is not None]
        avg_eod_penalty = float(np.mean(eod_values)) if eod_values else 0.0
        eq_odds_values = [m["equalized_odds_diff"] for m in single if m.get("equalized_odds_diff") is not None]
        avg_eq_odds_penalty = float(np.mean(eq_odds_values)) if eq_odds_values else 0.0

        raw = (
            100
            - (avg_spd * 200)
            - (avg_di_penalty * 100)
            - (flagged_ratio * 20)
            - (avg_eod_penalty * 100)
            - (avg_eq_odds_penalty * 80)
        )
        score = int(max(0, min(100, round(raw))))

        if score >= 80:
            label, color = "Fair", "green"
        elif score >= 60:
            label, color = "Caution", "yellow"
        else:
            label, color = "Biased", "red"

        return {"score": score, "label": label, "color": color}

    def bootstrap_metric_ci(
        self,
        df: "pd.DataFrame",
        predictions,
        protected_col: str,
        target_col: str,
        n_bootstrap: int = 200,
        ci: float = 0.95,
    ) -> dict:
        """Bootstrap confidence intervals for SPD per protected attribute value."""
        if protected_col not in df.columns:
            return {}

        if predictions is not None and len(predictions) == len(df):
            target = pd.Series(predictions, index=df.index, dtype=float)
        elif target_col in df.columns:
            target = pd.to_numeric(df[target_col], errors="coerce").fillna(0)
        else:
            return {}

        overall_rate = float(target.mean())
        if overall_rate == 0:
            return {}

        alpha = (1 - ci) / 2
        results = {}
        rng = np.random.default_rng(42)

        for val in df[protected_col].dropna().unique():
            mask = (df[protected_col] == val).values
            if mask.sum() < 10:
                continue
            group_target = target.values[mask]
            boots = []
            for _ in range(n_bootstrap):
                idx = rng.choice(len(group_target), size=len(group_target), replace=True)
                sample_rate = group_target[idx].mean()
                boots.append(float(sample_rate - overall_rate))
            boots_arr = np.array(boots)
            results[str(val)] = {
                "spd_mean": round(float(boots_arr.mean()), 4),
                "spd_lower": round(float(np.quantile(boots_arr, alpha)), 4),
                "spd_upper": round(float(np.quantile(boots_arr, 1 - alpha)), 4),
            }
        return results


cartography_service = BiasCartographyService()
