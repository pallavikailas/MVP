"""
Regulatory Compliance Mapper
Maps computed fairness metrics to real-world legal thresholds.
Returns PASS / FAIL / CAUTION verdicts per regulation.
"""

from typing import Any

REGULATIONS = [
    {
        "id": "EEOC_4FIFTHS",
        "label": "EEOC 4/5ths Rule",
        "description": "Adverse impact ratio must be ≥ 0.80 (80% rule) for employment decisions.",
        "domain": "Employment",
        "di_threshold": 0.80,
        "spd_threshold": None,
        "eod_threshold": None,
    },
    {
        "id": "EU_AI_ACT",
        "label": "EU AI Act Art. 10",
        "description": "High-risk AI systems must ensure SPD < 0.05 and DI ≥ 0.90 for protected groups.",
        "domain": "General / High-Risk AI",
        "di_threshold": 0.90,
        "spd_threshold": 0.05,
        "eod_threshold": None,
    },
    {
        "id": "ECOA_FCRA",
        "label": "ECOA / FCRA",
        "description": "Equal Credit Opportunity Act requires SPD < 0.10 across race, sex, national origin.",
        "domain": "Credit / Lending",
        "di_threshold": None,
        "spd_threshold": 0.10,
        "eod_threshold": None,
    },
    {
        "id": "FHA",
        "label": "Fair Housing Act",
        "description": "Fair Housing Act prohibits disparate impact in housing with SPD < 0.10.",
        "domain": "Housing",
        "di_threshold": None,
        "spd_threshold": 0.10,
        "eod_threshold": None,
    },
    {
        "id": "TITLE_VII",
        "label": "Title VII (Civil Rights Act)",
        "description": (
            "Prohibits employment discrimination based on race, color, religion, sex, national origin, "
            "language/accent, and political beliefs in some jurisdictions. SPD < 0.10."
        ),
        "domain": "Employment",
        "di_threshold": 0.80,
        "spd_threshold": 0.10,
        "eod_threshold": None,
    },
    {
        "id": "ADEA",
        "label": "ADEA (Age Discrimination in Employment Act)",
        "description": "Prohibits employment discrimination against people aged 40+. SPD < 0.10.",
        "domain": "Employment / Age",
        "di_threshold": None,
        "spd_threshold": 0.10,
        "eod_threshold": None,
    },
    {
        "id": "ADA",
        "label": "ADA (Americans with Disabilities Act)",
        "description": (
            "Prohibits discrimination based on disability or physical appearance covered under disability law. "
            "Includes BMI/obesity in some jurisdictions. SPD < 0.10."
        ),
        "domain": "Employment / Disability / Physical Appearance",
        "di_threshold": None,
        "spd_threshold": 0.10,
        "eod_threshold": None,
    },
    {
        "id": "BAN_THE_BOX",
        "label": "Ban-the-Box / Fair Chance Laws",
        "description": (
            "Restricts use of criminal history in hiring and lending decisions in many US states. "
            "DI ≥ 0.80 and SPD < 0.10 across criminal-history groups."
        ),
        "domain": "Employment / Criminal History",
        "di_threshold": 0.80,
        "spd_threshold": 0.10,
        "eod_threshold": None,
    },
    {
        "id": "EQUAL_OPPORTUNITY",
        "label": "Equal Opportunity (TPR Parity)",
        "description": (
            "Best-practice fairness standard requiring equal True Positive Rates across groups. "
            "Equal Opportunity Difference (EOD) < 0.10."
        ),
        "domain": "General / Best Practice",
        "di_threshold": None,
        "spd_threshold": None,
        "eod_threshold": 0.10,
    },
]


def _verdict(violations: list[str]) -> str:
    if not violations:
        return "PASS"
    # Caution if only barely over one threshold, FAIL if clearly over
    return "FAIL"


def check_compliance(slice_metrics: list[dict]) -> list[dict[str, Any]]:
    """
    Given slice_metrics from cartography, compute compliance verdicts.
    Uses worst-case SPD, minimum DI, and worst-case EOD across all single-attribute slices.
    """
    if not slice_metrics:
        return [
            {**r, "status": "PASS", "violations": [], "worst_spd": 0.0, "worst_di": 1.0, "worst_eod": 0.0}
            for r in REGULATIONS
        ]

    single = [m for m in slice_metrics if "∩" not in m["label"]]
    if not single:
        single = slice_metrics

    worst_spd = max(abs(m["statistical_parity_diff"]) for m in single)
    worst_di = min(m["disparate_impact"] for m in single if m.get("disparate_impact") is not None) if any(
        m.get("disparate_impact") is not None for m in single
    ) else 1.0
    worst_slice = max(single, key=lambda m: abs(m["statistical_parity_diff"]))["label"]

    eod_values = [abs(m["equal_opportunity_diff"]) for m in single if m.get("equal_opportunity_diff") is not None]
    worst_eod = max(eod_values) if eod_values else None
    worst_eod_slice = (
        max((m for m in single if m.get("equal_opportunity_diff") is not None),
            key=lambda m: abs(m["equal_opportunity_diff"]))["label"]
        if eod_values else None
    )

    results = []
    for reg in REGULATIONS:
        violations = []

        if reg["spd_threshold"] is not None and worst_spd > reg["spd_threshold"]:
            violations.append(
                f"SPD {worst_spd:.3f} exceeds limit {reg['spd_threshold']} "
                f"(worst: {worst_slice})"
            )

        if reg["di_threshold"] is not None and worst_di < reg["di_threshold"]:
            violations.append(
                f"DI {worst_di:.3f} below threshold {reg['di_threshold']} "
                f"(worst: {worst_slice})"
            )

        if reg.get("eod_threshold") is not None and worst_eod is not None and worst_eod > reg["eod_threshold"]:
            violations.append(
                f"EOD {worst_eod:.3f} exceeds limit {reg['eod_threshold']} "
                f"(worst: {worst_eod_slice})"
            )

        results.append({
            "id": reg["id"],
            "label": reg["label"],
            "description": reg["description"],
            "domain": reg["domain"],
            "status": _verdict(violations),
            "violations": violations,
            "worst_spd": round(worst_spd, 4),
            "worst_di": round(worst_di, 4),
            "worst_eod": round(worst_eod, 4) if worst_eod is not None else None,
        })

    return results
