"""
Proxy Variable Hunter
======================
Detects features that act as indirect proxies for protected attributes AND
correlate with the target outcome — the dual-correlation risk that makes
proxy discrimination legally and ethically significant.

Graph traversal finds chains like:
  zip_code → neighbourhood_income → race  (3-hop proxy)
  job_title → years_experience → gender   (2-hop proxy)

Risk score = corr_with_protected × corr_with_target × path_decay
"""

import pandas as pd
import numpy as np
import networkx as nx
import json
import logging
from scipy.stats import chi2_contingency, pointbiserialr
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from typing import Dict, List, Any, Optional

from app.services.gemini_client import ask_gemini_json

logger = logging.getLogger(__name__)


class ProxyVariableHunter:

    STRONG_CORRELATION = 0.3
    MODERATE_CORRELATION = 0.15

    def __init__(self):
        self.graph = nx.DiGraph()

    async def run_hunt(
        self,
        X: "pd.DataFrame",
        y: "Optional[pd.Series]",
        protected_cols: List[str],
        audit_id: str,
    ) -> Dict[str, Any]:
        logger.info(f"[{audit_id}] Starting proxy variable hunt on {len(X.columns)} features")

        self._build_correlation_graph(X, protected_cols)

        # Per-feature: correlation with protected attributes and with target outcome
        feature_protected_corr = self._feature_protected_correlations(X, protected_cols)
        feature_target_corr = self._feature_target_correlations(X, y) if y is not None else {}
        mi_scores = self._mutual_information(X, y) if y is not None else {}

        proxy_chains = self._find_proxy_chains(protected_cols, X.columns.tolist())
        risk_scored = self._score_proxy_risk(
            proxy_chains, feature_protected_corr, feature_target_corr, mi_scores, protected_cols
        )
        recommendations = self._generate_recommendations(risk_scored)
        gemini_narrative = await self._gemini_analyse(
            risk_scored, protected_cols, audit_id, len(X), y is not None
        )

        result = {
            "audit_id": audit_id,
            "graph": self._graph_to_json(),
            "proxy_chains": risk_scored,
            "recommendations": recommendations,
            "gemini_analysis": gemini_narrative,
            "summary": {
                "total_features_analyzed": len(X.columns),
                "proxy_features_found": len(set(c["start_feature"] for c in risk_scored)),
                "critical_proxies": sum(1 for c in risk_scored if c["risk_level"] == "critical"),
                "high_proxies": sum(1 for c in risk_scored if c["risk_level"] == "high"),
                "most_dangerous_proxy": risk_scored[0]["start_feature"] if risk_scored else None,
            },
        }
        logger.info(f"[{audit_id}] Proxy hunt complete. {len(risk_scored)} chains found.")
        return result

    # ── Correlation helpers ────────────────────────────────────────────────

    def _encode_df(self, X: pd.DataFrame) -> pd.DataFrame:
        enc = X.copy()
        le = LabelEncoder()
        for col in enc.select_dtypes(include=["object", "category"]).columns:
            try:
                enc[col] = le.fit_transform(enc[col].astype(str))
            except Exception:
                enc[col] = 0
        return enc.fillna(0)

    def _pair_corr(self, X: pd.DataFrame, col1: str, col2: str) -> float:
        """Best-fit correlation for any pair of column dtypes."""
        s1, s2 = X[col1], X[col2]
        cat1 = s1.dtype == object or str(s1.dtype) == "category"
        cat2 = s2.dtype == object or str(s2.dtype) == "category"

        if cat1 and cat2:
            ct = pd.crosstab(s1, s2)
            if ct.shape[0] < 2 or ct.shape[1] < 2:
                return 0.0
            chi2, _, _, _ = chi2_contingency(ct)
            n = ct.sum().sum()
            k = min(ct.shape) - 1
            return float(np.sqrt(chi2 / (n * k))) if k > 0 and n > 0 else 0.0

        if cat1 != cat2:
            # Point-biserial: numeric × binary; fall back to Pearson on label-encoded
            num_col = col2 if cat1 else col1
            bin_col = col1 if cat1 else col2
            le = LabelEncoder()
            bin_enc = le.fit_transform(X[bin_col].astype(str))
            if len(np.unique(bin_enc)) == 2:
                try:
                    r, _ = pointbiserialr(bin_enc, X[num_col].fillna(0))
                    return abs(float(r))
                except Exception:
                    pass
            # Fallback — Pearson on encoded
            enc = self._encode_df(X[[col1, col2]])
            return abs(float(enc[col1].corr(enc[col2])))

        # Both numeric — Pearson
        return abs(float(s1.fillna(0).corr(s2.fillna(0))))

    def _feature_protected_correlations(
        self, X: pd.DataFrame, protected_cols: List[str]
    ) -> Dict[str, float]:
        """Max correlation of each feature with any protected attribute."""
        result = {}
        for feat in X.columns:
            if feat in protected_cols:
                continue
            max_corr = 0.0
            for prot in protected_cols:
                if prot not in X.columns:
                    continue
                try:
                    max_corr = max(max_corr, self._pair_corr(X, feat, prot))
                except Exception:
                    pass
            result[feat] = max_corr
        return result

    def _feature_target_correlations(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Correlation of each feature with the target outcome."""
        result = {}
        y_clean = pd.to_numeric(y, errors="coerce").fillna(0)
        for feat in X.columns:
            try:
                result[feat] = self._pair_corr(
                    X.assign(_y=y_clean.values), feat, "_y"
                )
            except Exception:
                result[feat] = 0.0
        return result

    def _mutual_information(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Dict[str, float]:
        """Normalised mutual information of each feature with the target."""
        try:
            enc = self._encode_df(X)
            y_clean = pd.to_numeric(y, errors="coerce").fillna(0).astype(int)
            mi = mutual_info_classif(enc, y_clean, random_state=42)
            max_mi = max(mi.max(), 1e-9)
            return {col: float(mi[i] / max_mi) for i, col in enumerate(X.columns)}
        except Exception:
            return {}

    # ── Graph construction ─────────────────────────────────────────────────

    def _build_correlation_graph(self, X: pd.DataFrame, protected_cols: List[str]):
        self.graph.clear()
        all_cols = X.columns.tolist()
        for col in all_cols:
            self.graph.add_node(
                col,
                is_protected=col in protected_cols,
                dtype=str(X[col].dtype),
            )

        for i, col1 in enumerate(all_cols):
            for col2 in all_cols[i + 1:]:
                try:
                    corr = self._pair_corr(X, col1, col2)
                    if corr <= self.MODERATE_CORRELATION:
                        continue
                    if col2 in protected_cols:
                        self.graph.add_edge(col1, col2, weight=corr, correlation=corr)
                    elif col1 in protected_cols:
                        self.graph.add_edge(col2, col1, weight=corr, correlation=corr)
                    else:
                        self.graph.add_edge(col1, col2, weight=corr, correlation=corr)
                        self.graph.add_edge(col2, col1, weight=corr, correlation=corr)
                except Exception:
                    continue

    # ── Path finding ───────────────────────────────────────────────────────

    def _find_proxy_chains(
        self, protected_cols: List[str], all_cols: List[str]
    ) -> List[Dict]:
        chains = []
        non_protected = [c for c in all_cols if c not in protected_cols]

        for protected in protected_cols:
            if protected not in self.graph:
                continue
            for source in non_protected:
                if source not in self.graph:
                    continue
                try:
                    for path in nx.all_simple_paths(
                        self.graph, source=source, target=protected, cutoff=4
                    ):
                        edges = []
                        # Use min edge weight along the path (weakest link)
                        min_weight = 1.0
                        for j in range(len(path) - 1):
                            w = self.graph.get_edge_data(path[j], path[j + 1], {}).get("weight", 0.1)
                            min_weight = min(min_weight, w)
                            edges.append({"from": path[j], "to": path[j + 1], "correlation": round(w, 3)})
                        if min_weight > 0.01:
                            chains.append({
                                "start_feature": source,
                                "target_protected": protected,
                                "path": path,
                                "path_length": len(path) - 1,
                                "chain_strength": round(min_weight, 4),
                                "edges": edges,
                            })
                except nx.NetworkXError:
                    continue

        return sorted(chains, key=lambda c: c["chain_strength"], reverse=True)[:50]

    # ── Risk scoring ────────────────────────────────────────────────────────

    def _name_similarity(self, feat: str, protected_cols: List[str]) -> float:
        """Simple token-overlap score between feature name and protected attribute names."""
        feat_tokens = set(feat.lower().replace("_", " ").split())
        best = 0.0
        for prot in protected_cols:
            prot_tokens = set(prot.lower().replace("_", " ").split())
            if not feat_tokens or not prot_tokens:
                continue
            overlap = len(feat_tokens & prot_tokens) / len(feat_tokens | prot_tokens)
            # Also flag if prot token is a substring of feat name
            if any(pt in feat.lower() for pt in prot_tokens):
                overlap = max(overlap, 0.5)
            best = max(best, overlap)
        return best

    def _score_proxy_risk(
        self,
        chains: List[Dict],
        feature_protected_corr: Dict[str, float],
        feature_target_corr: Dict[str, float],
        mi_scores: Dict[str, float],
        protected_cols: List[str],
    ) -> List[Dict]:
        scored = []
        for chain in chains:
            feat = chain["start_feature"]
            prot_corr = feature_protected_corr.get(feat, chain["chain_strength"])
            target_corr = feature_target_corr.get(feat, 0.0)
            mi = mi_scores.get(feat, 0.0)
            name_sim = self._name_similarity(feat, protected_cols)
            path_len = chain["path_length"]

            # Dual-correlation risk: dangerous only when correlated with BOTH protected AND target
            dual_risk = prot_corr * max(target_corr, mi)
            # Path decay: shorter paths are more direct and riskier
            path_factor = 1.0 / path_len
            risk_score = (dual_risk * 0.65) + (name_sim * 0.15) + (path_factor * 0.2)

            risk_level = (
                "critical" if risk_score > 0.25 else
                "high" if risk_score > 0.12 else
                "medium" if risk_score > 0.05 else
                "low"
            )

            path_str = " → ".join(chain["path"])
            explanation = (
                f"'{feat}' is a {risk_level}-risk proxy for '{chain['target_protected']}' "
                f"via: {path_str}. "
                f"Correlation with protected attribute: {prot_corr:.1%}; "
                f"with target outcome: {max(target_corr, mi):.1%}."
            )
            if name_sim > 0.3:
                explanation += f" Feature name semantically overlaps with protected attribute ({name_sim:.0%} token similarity)."

            scored.append({
                **chain,
                "corr_with_protected": round(prot_corr, 4),
                "corr_with_target": round(target_corr, 4),
                "mutual_information": round(mi, 4),
                "name_similarity": round(name_sim, 4),
                "risk_score": round(risk_score, 4),
                "risk_level": risk_level,
                "explanation": explanation,
            })

        return sorted(scored, key=lambda c: c["risk_score"], reverse=True)

    # ── Recommendations ─────────────────────────────────────────────────────

    def _generate_recommendations(self, risk_scored: List[Dict]) -> List[Dict]:
        recommendations = []
        seen = set()
        for chain in risk_scored:
            if chain["risk_level"] not in ("critical", "high"):
                continue
            feat = chain["start_feature"]
            if feat in seen:
                continue
            seen.add(feat)
            recommendations.append({
                "feature": feat,
                "risk_level": chain["risk_level"],
                "action": self._recommend_action(chain),
                "chain": " → ".join(chain["path"]),
                "corr_with_protected": chain["corr_with_protected"],
                "corr_with_target": chain["corr_with_target"],
            })
        return recommendations

    def _recommend_action(self, chain: Dict) -> str:
        feat = chain["start_feature"]
        prot = chain["target_protected"]
        target_corr = chain.get("corr_with_target", 0)
        if chain["path_length"] == 1:
            if target_corr > 0.3:
                return (
                    f"REMOVE '{feat}' from model inputs — it directly encodes '{prot}' "
                    f"and is strongly predictive of the outcome ({target_corr:.0%} target correlation). "
                    f"This is a textbook proxy discrimination feature."
                )
            return (
                f"AUDIT '{feat}' — it directly encodes '{prot}'. "
                f"Apply orthogonal projection or adversarial debiasing before retraining."
            )
        elif chain["path_length"] == 2:
            return (
                f"AUDIT '{feat}' — 2-hop proxy via {chain['path'][1]} → '{prot}'. "
                f"Measure conditional independence P({feat} | {prot}) and consider reweighting "
                f"or using a fairness constraint (equalised odds) during training."
            )
        return (
            f"MONITOR '{feat}' — {chain['path_length']}-hop indirect proxy for '{prot}'. "
            f"Add to fairness-sensitive feature list. Consider a fairness regulariser at next retrain."
        )

    # ── Gemini narrative ────────────────────────────────────────────────────

    async def _gemini_analyse(
        self,
        risk_scored: List[Dict],
        protected_cols: List[str],
        audit_id: str,
        n_rows: int,
        has_target: bool,
    ) -> Dict:
        top = risk_scored[:8]
        summary_json = json.dumps([{
            "feature": c["start_feature"],
            "protected": c["target_protected"],
            "risk_level": c["risk_level"],
            "path": " → ".join(c["path"]),
            "corr_with_protected": c["corr_with_protected"],
            "corr_with_target": c.get("corr_with_target", "n/a"),
        } for c in top], indent=2)

        prompt = f"""You are an AI fairness auditor reviewing proxy variable analysis.
DATASET: {n_rows} rows, protected attributes: {protected_cols}
TARGET CORRELATION AVAILABLE: {has_target}
TOP PROXY CHAINS:
{summary_json}

Return ONLY this JSON:
{{"severity":"critical|high|medium|low","headline":"one sentence summary","key_findings":["f1","f2","f3"],"most_dangerous_proxy":"feature name","legal_risk":"legal risk explanation","recommended_action":"highest-priority action","debiasing_strategy":"specific technical approach (e.g. orthogonal projection, reweighting, fairness constraint)"}}"""
        try:
            return await ask_gemini_json(prompt)
        except Exception as e:
            logger.warning(f"[{audit_id}] Gemini proxy analysis failed: {e}")
            return {"severity": "unknown", "headline": "Analysis unavailable", "key_findings": []}

    # ── Graph serialisation ─────────────────────────────────────────────────

    def _graph_to_json(self) -> Dict:
        nodes = [
            {"id": n, "is_protected": d.get("is_protected", False), "dtype": d.get("dtype", "unknown")}
            for n, d in self.graph.nodes(data=True)
        ]
        edges = [
            {"source": u, "target": v, "weight": round(d.get("weight", 0), 3)}
            for u, v, d in self.graph.edges(data=True)
        ]
        return {"nodes": nodes, "edges": edges}


proxy_hunter_service = ProxyVariableHunter()
