"""
Dataset Bias Probe Service
===========================
Analyzes the user's uploaded dataset for structural biases WITHOUT any model.
Uses ground-truth labels only, so findings represent biases baked into the
data distribution itself — independent of any model's decisions.

Pipeline
--------
1. Bias Cartography on dataset labels (no model predictions)  → demographic disparity map
2. Proxy Variable Hunter → proxy chains linking features to protected attributes
3. Extract structured dataset bias list for cross-analysis + red-team targeting
"""

import io
import logging
import pandas as pd
from typing import Any, Dict, List

from app.services.cartography import cartography_service
from app.services.proxy_hunter import ProxyVariableHunter

logger = logging.getLogger(__name__)


class DatasetBiasProbe:

    async def probe(
        self,
        dataset_csv: str,
        protected_cols: List[str],
        target_col: str,
        audit_id: str,
    ) -> Dict[str, Any]:
        """
        Analyze *dataset_csv* for structural biases.
        No model involved — ground-truth labels drive all metrics.
        """
        logger.info(f"[{audit_id}] Starting dataset bias probe (no-model mode)")

        df = pd.read_csv(io.StringIO(dataset_csv))

        # ── 1. Cartography on dataset labels ─────────────────────────────────
        carto_results = await cartography_service.run_cartography(
            dataset_csv=dataset_csv,
            protected_cols=protected_cols,
            target_col=target_col,
            model_predictions=None,   # ground-truth labels only
            audit_id=audit_id,
        )

        # ── 2. Proxy Variable Hunter ──────────────────────────────────────────
        hunter = ProxyVariableHunter()
        feature_cols = [c for c in df.columns if c != target_col]
        X = df[feature_cols]
        y = (
            pd.to_numeric(df[target_col], errors="coerce").fillna(0)
            if target_col in df.columns
            else None
        )

        proxy_results = await hunter.run_hunt(
            X=X,
            y=y,
            protected_cols=protected_cols,
            audit_id=audit_id,
        )

        # ── 3. Extract structured bias list ──────────────────────────────────
        dataset_biases = self._extract_dataset_biases(carto_results, proxy_results)

        return {
            "audit_id":       audit_id,
            "analysis_type":  "dataset_probe",
            "dataset_size":   len(df),
            "protected_cols": protected_cols,
            "target_col":     target_col,
            "cartography":    carto_results,
            "proxy":          proxy_results,
            "dataset_biases": dataset_biases,
            "summary": {
                "fair_score":          carto_results.get("fair_score", {}),
                "bias_count":          len(dataset_biases),
                "proxy_count":         len(proxy_results.get("proxy_chains", [])),
                "critical_proxies":    proxy_results.get("summary", {}).get("critical_proxies", 0),
                "analysis_source":     "user_dataset_ground_truth_labels",
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_dataset_biases(carto: Dict, proxy: Dict) -> List[Dict]:
        """Collect structured bias objects from cartography + proxy hunter results."""
        biases: Dict[str, Dict] = {}

        # From cartography hotspots (statistical disparity in ground-truth labels)
        for hotspot in carto.get("hotspots", []):
            attr = str(hotspot.get("dominant_slice", "")).split("=")[0].strip()
            if not attr:
                continue
            entry = {
                "attribute": attr,
                "value":     hotspot.get("dominant_slice", ""),
                "type":      "dataset_statistical_disparity",
                "severity":  hotspot.get("severity", "medium"),
                "magnitude": float(hotspot.get("mean_bias_magnitude", 0)),
                "source":    "dataset_probe_cartography",
                "spd":       float(hotspot.get("statistical_parity_diff", 0)),
            }
            if attr not in biases or entry["magnitude"] > biases[attr]["magnitude"]:
                biases[attr] = entry

        # From proxy chains (critical + high risk only)
        for chain in proxy.get("proxy_chains", []):
            if chain.get("risk_level") not in ("critical", "high"):
                continue
            attr = chain.get("start_feature", "")
            if not attr:
                continue
            entry = {
                "attribute":        attr,
                "value":            chain.get("protected_attribute", ""),
                "type":             "proxy_chain",
                "severity":         chain.get("risk_level", "medium"),
                "magnitude":        float(chain.get("risk_score", 0)),
                "source":           "dataset_probe_proxy",
                "chain_path":       chain.get("path", []),
                "protected_target": chain.get("protected_attribute", ""),
            }
            if attr not in biases or entry["magnitude"] > biases[attr]["magnitude"]:
                biases[attr] = entry

        return sorted(biases.values(), key=lambda x: x["magnitude"], reverse=True)


dataset_probe_service = DatasetBiasProbe()
