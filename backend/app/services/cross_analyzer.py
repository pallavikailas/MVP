"""
Cross-Analysis Service
=======================
Compares model biases (Phase 1) against dataset biases (Phase 2) to expose
hidden compounded risks that only become visible when both are seen together.

Three categories of cross-bias
-------------------------------
  Aligned bias         — same protected attribute is biased in BOTH model AND dataset
                         → highest legal risk; bias is structural and self-reinforcing
  Proxy amplification  — a proxy chain in the dataset routes into an attribute
                         the model is already biased against → proxy makes bias invisible
  Blind spot           — bias detected in one source but not the other
                         → model may be learning from a biased feature it "doesn't know" exists

Phase 3 — model + user dataset grouped and run through all three analysis stages
(cartography + constitution + proxy hunter) — surfaces interaction biases that
neither Phase 1 nor Phase 2 can see alone.
"""

import json
import logging
from typing import Any, Dict, List

from app.services.gemini_client import ask_gemini_json

logger = logging.getLogger(__name__)


class CrossAnalyzer:

    async def analyze(
        self,
        model_probe_results: Dict,
        dataset_probe_results: Dict,
        audit_id: str,
    ) -> Dict[str, Any]:
        """
        Cross-analyze *model_probe_results* vs *dataset_probe_results*.
        """
        logger.info(f"[{audit_id}] Starting cross-analysis (model × dataset)")

        model_biases   = model_probe_results.get("model_biases", [])
        dataset_biases = dataset_probe_results.get("dataset_biases", [])
        proxy_results  = dataset_probe_results.get("proxy", {})

        aligned              = self._find_aligned(model_biases, dataset_biases)
        proxy_amplifications = self._find_proxy_amplifications(model_biases, proxy_results)
        blind_spots          = self._find_blind_spots(model_biases, dataset_biases)

        risk_matrix      = self._build_risk_matrix(aligned, proxy_amplifications, blind_spots)
        combined_biases  = self._build_combined_biases(aligned, proxy_amplifications, blind_spots)
        gemini_analysis  = await self._gemini_synthesise(
            model_biases, dataset_biases, aligned, proxy_amplifications, audit_id
        )

        return {
            "audit_id":               audit_id,
            "analysis_type":          "cross_analysis",
            "aligned_biases":         aligned,
            "proxy_amplifications":   proxy_amplifications,
            "blind_spots":            blind_spots,
            "risk_matrix":            risk_matrix,
            "combined_biases":        combined_biases,
            "gemini_analysis":        gemini_analysis,
            "summary": {
                "aligned_count":           len(aligned),
                "proxy_amplification_count": len(proxy_amplifications),
                "blind_spot_count":        len(blind_spots),
                "total_compounded_risks":  len(combined_biases),
                "highest_risk_attribute":  combined_biases[0]["attribute"] if combined_biases else None,
            },
        }

    # ── Category finders ──────────────────────────────────────────────────────

    @staticmethod
    def _find_aligned(model_biases: List[Dict], dataset_biases: List[Dict]) -> List[Dict]:
        model_map   = {b["attribute"]: b for b in model_biases}
        dataset_map = {b["attribute"]: b for b in dataset_biases}

        aligned = []
        for attr in set(model_map) & set(dataset_map):
            mb = model_map[attr]
            db = dataset_map[attr]
            compounded = round((mb["magnitude"] + db["magnitude"]) / 2, 4)
            aligned.append({
                "attribute":         attr,
                "model_magnitude":   mb["magnitude"],
                "dataset_magnitude": db["magnitude"],
                "compounded_risk":   compounded,
                "model_severity":    mb.get("severity", "unknown"),
                "dataset_severity":  db.get("severity", "unknown"),
                "risk_type":         "compounded",
                "description":       (
                    f"'{attr}' bias appears in both model decisions (magnitude={mb['magnitude']:.3f}) "
                    f"and dataset distribution (magnitude={db['magnitude']:.3f}). "
                    "Self-reinforcing: the model likely learned the bias from the data."
                ),
            })

        return sorted(aligned, key=lambda x: x["compounded_risk"], reverse=True)

    @staticmethod
    def _find_proxy_amplifications(
        model_biases: List[Dict], proxy_results: Dict
    ) -> List[Dict]:
        model_attrs = {b["attribute"] for b in model_biases}
        amplifications = []

        for chain in proxy_results.get("proxy_chains", []):
            protected_attr = chain.get("protected_attribute", "")
            if protected_attr in model_attrs:
                amplifications.append({
                    "proxy_feature":      chain.get("start_feature", ""),
                    "reaches_attribute":  protected_attr,
                    "chain_path":         chain.get("path", []),
                    "proxy_risk_score":   float(chain.get("risk_score", 0)),
                    "risk_type":          "proxy_amplification",
                    "description": (
                        f"Feature '{chain.get('start_feature', '')}' is a proxy that routes "
                        f"to '{protected_attr}', which the model is already biased against. "
                        "The proxy provides a legally 'neutral' pathway to the same discriminatory outcome."
                    ),
                })

        return sorted(amplifications, key=lambda x: x["proxy_risk_score"], reverse=True)

    @staticmethod
    def _find_blind_spots(
        model_biases: List[Dict], dataset_biases: List[Dict]
    ) -> List[Dict]:
        model_attrs   = {b["attribute"] for b in model_biases}
        dataset_attrs = {b["attribute"] for b in dataset_biases}
        model_map     = {b["attribute"]: b for b in model_biases}
        dataset_map   = {b["attribute"]: b for b in dataset_biases}

        spots = []

        for attr in model_attrs - dataset_attrs:
            mb = model_map[attr]
            spots.append({
                "attribute":  attr,
                "found_in":   "model_only",
                "magnitude":  mb["magnitude"],
                "description": (
                    f"Model is biased against '{attr}' (magnitude={mb['magnitude']:.3f}) "
                    "but the dataset statistics don't surface this — the bias is internal to the model."
                ),
            })

        for attr in dataset_attrs - model_attrs:
            db = dataset_map[attr]
            spots.append({
                "attribute":  attr,
                "found_in":   "dataset_only",
                "magnitude":  db["magnitude"],
                "description": (
                    f"Dataset shows '{attr}' disparity (magnitude={db['magnitude']:.3f}) "
                    "that did not surface in model-probe analysis — the model may be masking it."
                ),
            })

        return sorted(spots, key=lambda x: x["magnitude"], reverse=True)

    # ── Risk matrix & combined bias list ──────────────────────────────────────

    @staticmethod
    def _build_risk_matrix(
        aligned: List[Dict],
        proxy_amplifications: List[Dict],
        blind_spots: List[Dict],
    ) -> List[Dict]:
        matrix = []

        for a in aligned:
            matrix.append({
                "risk_type": "Compounded Bias",
                "attribute": a["attribute"],
                "severity":  "critical" if a["compounded_risk"] > 0.3 else "high" if a["compounded_risk"] > 0.15 else "medium",
                "risk_score": a["compounded_risk"],
                "description": a["description"],
            })

        for p in proxy_amplifications:
            matrix.append({
                "risk_type":  "Proxy Amplification",
                "attribute":  p["proxy_feature"],
                "severity":   "high",
                "risk_score": p["proxy_risk_score"],
                "description": p["description"],
            })

        for b in blind_spots:
            matrix.append({
                "risk_type":  "Blind Spot",
                "attribute":  b["attribute"],
                "severity":   "medium",
                "risk_score": b["magnitude"],
                "description": b["description"],
            })

        return sorted(matrix, key=lambda x: x["risk_score"], reverse=True)

    @staticmethod
    def _build_combined_biases(
        aligned: List[Dict],
        proxy_amplifications: List[Dict],
        blind_spots: List[Dict],
    ) -> List[Dict]:
        """Build unified bias list for red-team targeting."""
        combined = []

        for a in aligned:
            combined.append({
                "attribute": a["attribute"],
                "type":      "compounded",
                "severity":  "critical" if a["compounded_risk"] > 0.3 else "high",
                "magnitude": a["compounded_risk"],
                "source":    "cross_analysis_aligned",
            })

        for p in proxy_amplifications:
            combined.append({
                "attribute": p["proxy_feature"],
                "type":      "proxy_amplification",
                "severity":  "high",
                "magnitude": p["proxy_risk_score"],
                "source":    "cross_analysis_proxy",
            })

        for b in blind_spots:
            if b["magnitude"] > 0.1:
                combined.append({
                    "attribute": b["attribute"],
                    "type":      f"blind_spot_{b['found_in']}",
                    "severity":  "medium",
                    "magnitude": b["magnitude"],
                    "source":    "cross_analysis_blind_spot",
                })

        return sorted(combined, key=lambda x: x["magnitude"], reverse=True)

    # ── Gemini synthesis ──────────────────────────────────────────────────────

    async def _gemini_synthesise(
        self,
        model_biases: List[Dict],
        dataset_biases: List[Dict],
        aligned: List[Dict],
        proxy_amplifications: List[Dict],
        audit_id: str,
    ) -> Dict:
        prompt = f"""You are an AI fairness auditor conducting cross-analysis between model-intrinsic
biases and dataset-structural biases.

MODEL BIASES (found by probing model on embedded reference dataset):
{json.dumps(model_biases[:5], indent=2)}

DATASET BIASES (found in user-uploaded dataset, no model):
{json.dumps(dataset_biases[:5], indent=2)}

ALIGNED BIASES (present in BOTH model and dataset — highest compounded risk):
{json.dumps(aligned[:3], indent=2)}

PROXY AMPLIFICATIONS (proxy in dataset routes into attribute model is already biased against):
{json.dumps(proxy_amplifications[:3], indent=2)}

Return ONLY this JSON (no markdown, no extra text):
{{
  "severity": "critical|high|medium|low",
  "headline": "one sentence naming the highest-risk cross-bias",
  "key_findings": ["finding1", "finding2", "finding3"],
  "compounded_risk_groups": ["most affected group 1", "most affected group 2"],
  "interaction_mechanism": "explain how model bias and dataset bias amplify each other",
  "legal_risk": "specific legal framework and threshold at risk",
  "recommended_action": "highest-priority remediation step"
}}"""

        try:
            return await ask_gemini_json(prompt)
        except Exception as e:
            logger.warning(f"[{audit_id}] Cross-analysis Gemini synthesis failed: {e}")
            return {
                "severity":    "unknown",
                "headline":    "Cross-analysis synthesis unavailable",
                "key_findings": [],
            }


cross_analyzer_service = CrossAnalyzer()
