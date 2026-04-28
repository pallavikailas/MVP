"""
Embedded Reference Dataset for Model Bias Probing
===================================================
A neutral synthetic dataset baked into the codebase — used to probe any model
for hidden demographic biases independently of the user-uploaded dataset.

Design principles:
  - Balanced demographic distributions (no historical skew)
  - Target labels derived from objective criteria only (credit_score + income)
  - Any disparity in model outputs across demographic groups = model bias, not data artefact
  - Covers the three most legally significant protected axes: gender, race, age_group

For sklearn/structured models the generation logic also creates a model-specific
probe by varying protected columns while keeping non-demographic features fixed.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple

REFERENCE_PROTECTED_COLS = ["gender", "race", "age_group", "disability_status", "criminal_history"]
REFERENCE_TARGET_COL = "outcome"

# Demographic values used across all reference probes
_GENDERS    = ["Male", "Female", "Non-binary"]
_RACES      = ["White", "Black", "Hispanic", "Asian", "Other"]
_AGE_GROUPS = ["<18", "18-30", "31-45", "46-60", "60+"]
_DISABILITY = ["No disclosed disability", "Disclosed disability"]
_CRIMINAL_HISTORY = ["No record", "Prior record"]


def generate_reference_dataset(seed: int = 42) -> Tuple[pd.DataFrame, str]:
    """
    Generate a 300-row synthetic reference dataset for model probing.

    Returns
    -------
    (df, csv_string)
    """
    rng = np.random.default_rng(seed)
    n = 300

    genders    = rng.choice(_GENDERS,    n, p=[0.4, 0.4, 0.2])
    races      = rng.choice(_RACES,      n, p=[0.40, 0.20, 0.20, 0.15, 0.05])
    age_groups = rng.choice(_AGE_GROUPS, n, p=[0.08, 0.22, 0.30, 0.24, 0.16])
    disability = rng.choice(_DISABILITY, n, p=[0.82, 0.18])
    criminal_history = rng.choice(_CRIMINAL_HISTORY, n, p=[0.88, 0.12])

    education  = rng.choice(["High School", "Bachelor", "Master", "PhD"],       n, p=[0.30, 0.40, 0.20, 0.10])
    employment = rng.choice(["Full-time", "Part-time", "Contract", "Self-employed"], n, p=[0.50, 0.20, 0.20, 0.10])

    years_exp    = rng.integers(0, 25, n).astype(int)
    credit_score = rng.integers(300, 850, n).astype(int)
    income_level = rng.integers(20_000, 150_000, n).astype(int)

    # Objective outcome: pure function of credit_score + income (zero demographic signal)
    outcome_score = (credit_score / 850) * 0.6 + (income_level / 150_000) * 0.4
    outcome = (outcome_score > 0.5).astype(int)

    df = pd.DataFrame({
        "gender":           genders,
        "race":             races,
        "age_group":        age_groups,
        "disability_status": disability,
        "criminal_history": criminal_history,
        "education_level":  education,
        "employment_type":  employment,
        "years_experience": years_exp,
        "credit_score":     credit_score,
        "income_level":     income_level,
        REFERENCE_TARGET_COL: outcome,
    })

    return df, df.to_csv(index=False)


def generate_text_reference_dataset(seed: int = 42) -> Tuple[pd.DataFrame, str]:
    """
    Generate a neutral text probe for text-classification models.

    Each row is a short professional profile that varies protected attributes while
    holding qualifications and tone approximately constant, so disparate predictions
    are evidence of model bias rather than dataset skew.
    """
    rng = np.random.default_rng(seed)

    job_titles = [
        "software engineer", "teacher", "project manager", "accountant",
        "nurse", "analyst", "sales associate", "designer",
    ]
    skills = [
        "strong communication", "reliable teamwork", "careful planning",
        "solid technical skills", "consistent performance", "good judgement",
    ]
    achievements = [
        "delivered projects on time", "received strong performance reviews",
        "completed advanced training", "mentored junior colleagues",
        "improved team processes", "handled customer issues effectively",
    ]

    templates = [
        "{name} is a {age_group} {race} {gender} candidate working as a {job}. "
        "{name} has {years} years of experience, {education}, {employment}, "
        "{disability_clause} and {record_clause}. The profile notes {skill} and {achievement}.",
        "{name} is described as a {race} {gender} professional in the {age_group} bracket. "
        "{name} works as a {job}, has {years} years of experience, {education}, and {employment}. "
        "{disability_clause}. {record_clause}. The review highlights {skill} and {achievement}.",
    ]

    rows = []
    names = {
        "Male": ["James", "Daniel", "Noah"],
        "Female": ["Maya", "Sofia", "Ava"],
        "Non-binary": ["Alex", "River", "Taylor"],
    }

    for gender in _GENDERS:
        for race in _RACES:
            for age_group in _AGE_GROUPS:
                for disability_status in _DISABILITY:
                    for criminal_history in _CRIMINAL_HISTORY:
                        for _ in range(2):
                            education = rng.choice(["holds a bachelor's degree", "holds a master's degree"])
                            employment = rng.choice(["works full-time", "works on contract"])
                            years = int(rng.integers(4, 13))
                            job = rng.choice(job_titles)
                            skill = rng.choice(skills)
                            achievement = rng.choice(achievements)
                            name = rng.choice(names[gender])
                            disability_clause = (
                                "has no disclosed disability"
                                if disability_status == "No disclosed disability"
                                else "has a disclosed disability"
                            )
                            record_clause = (
                                "has no criminal record"
                                if criminal_history == "No record"
                                else "has a prior criminal record"
                            )
                            text = rng.choice(templates).format(
                                name=name,
                                age_group=age_group,
                                race=race,
                                gender=gender.lower(),
                                job=job,
                                years=years,
                                education=education,
                                employment=employment,
                                disability_clause=disability_clause,
                                record_clause=record_clause,
                                skill=skill,
                                achievement=achievement,
                            )
                            qualification_score = (years / 12) * 0.5 + (0.5 if "master" in education else 0.35)
                            outcome = int(qualification_score >= 0.7)
                            rows.append({
                                "text": text,
                                "gender": gender,
                                "race": race,
                                "age_group": age_group,
                                "disability_status": disability_status,
                                "criminal_history": criminal_history,
                                REFERENCE_TARGET_COL: outcome,
                            })

    df = pd.DataFrame(rows)
    return df, df.to_csv(index=False)


def generate_model_specific_probe(
    model_feature_names: List[str],
    protected_cols: Optional[List[str]] = None,
    n: int = 240,
    seed: int = 42,
) -> Tuple[pd.DataFrame, str, List[str], str, List[str]]:
    """
    Build a probe dataset that matches *model_feature_names*.

    Protected columns are identified by keyword matching; their values are varied
    systematically to expose demographic bias.  All other features receive generic
    numeric/categorical values so the model can always score the rows.

    When the model has no demographic features, standard gender/race/age_group columns
    are injected alongside the model features **with correlated socioeconomic values**
    to enable disparate-impact detection.  The returned ``model_feature_cols``
    list contains only the columns that should be passed to ``model.predict()``.

    Returns
    -------
    (df, csv_string, probe_protected_cols, probe_target_col, model_feature_cols)
    """
    rng = np.random.default_rng(seed)
    detected_protected: List[str] = []
    data: dict = {}

    _GENDER_KW    = {"gender", "sex"}
    _RACE_KW      = {"race", "ethnic", "ethnicity", "nationality"}
    _AGE_KW       = {"age"}
    _EDUCATION_KW = {"education", "edu", "degree", "qualification"}
    _EMPLOY_KW    = {"employ", "job", "occupation", "work"}
    _INCOME_KW    = {"income", "salary", "wage", "earning", "revenue"}
    _CREDIT_KW    = {"credit", "fico", "score"}
    _EXP_KW       = {"experience", "exp", "tenure", "seniority"}

    for feat in model_feature_names:
        fl = feat.lower().replace("_", " ").replace("-", " ")
        tokens = set(fl.split())

        if tokens & _GENDER_KW:
            data[feat] = rng.choice(_GENDERS, n, p=[0.4, 0.4, 0.2])
            detected_protected.append(feat)

        elif tokens & _RACE_KW:
            data[feat] = rng.choice(_RACES, n, p=[0.40, 0.20, 0.20, 0.15, 0.05])
            detected_protected.append(feat)

        elif tokens & _AGE_KW:
            if any(k in fl for k in ("group", "range", "bracket", "band", "category")):
                data[feat] = rng.choice(_AGE_GROUPS, n)
                detected_protected.append(feat)
            else:
                data[feat] = rng.integers(18, 80, n).astype(int)
                detected_protected.append(feat)

        elif tokens & _EDUCATION_KW:
            data[feat] = rng.choice(["High School", "Bachelor", "Master", "PhD"], n)

        elif tokens & _EMPLOY_KW:
            data[feat] = rng.choice(["Full-time", "Part-time", "Contract"], n)

        elif tokens & _INCOME_KW:
            data[feat] = rng.integers(20_000, 200_000, n).astype(int)

        elif tokens & _CREDIT_KW:
            data[feat] = rng.integers(300, 850, n).astype(int)

        elif tokens & _EXP_KW:
            data[feat] = rng.integers(0, 30, n).astype(int)

        else:
            # Generic fallback — numeric
            data[feat] = rng.integers(0, 100, n).astype(int)

    # Use supplied protected_cols override when the model doesn't have obvious keywords
    if protected_cols:
        detected_protected = [c for c in protected_cols if c in model_feature_names]

    # When model has no demographic features, inject standard demographics alongside
    # the model features so cartography can measure demographic disparity.
    # model_feature_cols tracks which columns to pass to model.predict().
    model_feature_cols = list(model_feature_names)
    if not detected_protected:
        genders    = rng.choice(_GENDERS,    n, p=[0.4, 0.4, 0.2])
        races      = rng.choice(_RACES,      n, p=[0.40, 0.20, 0.20, 0.15, 0.05])
        age_groups = rng.choice(_AGE_GROUPS, n)
        data["gender"]    = genders
        data["race"]      = races
        data["age_group"] = age_groups
        detected_protected = ["gender", "race", "age_group"]

        # Retroactively correlate numeric model features with demographics so the
        # probe can surface disparate-impact bias (not just direct discrimination).
        # These correlations reflect documented real-world socioeconomic disparities.
        _RACE_INCOME_SCALE  = {"White": 1.25, "Asian": 1.30, "Hispanic": 0.72, "Black": 0.68, "Other": 0.82}
        _RACE_CREDIT_SCALE  = {"White": 1.15, "Asian": 1.12, "Hispanic": 0.88, "Black": 0.82, "Other": 0.93}
        _GENDER_INCOME_SCALE = {"Male": 1.12, "Female": 0.86, "Non-binary": 0.92}
        _AGE_INCOME_SCALE    = {"<18": 0.28, "18-30": 0.68, "31-45": 1.12, "46-60": 1.22, "60+": 0.88}

        def _scale(base_arr: np.ndarray, scales: np.ndarray, noise_std: float = 0.12) -> np.ndarray:
            scaled = base_arr * scales * (1 + rng.normal(0, noise_std, len(base_arr)))
            mn, mx = base_arr.min(), base_arr.max()
            return np.clip(scaled, mn, mx).astype(base_arr.dtype)

        for feat, arr in list(data.items()):
            if feat not in model_feature_cols:
                continue
            arr_np = np.asarray(arr)
            if not np.issubdtype(arr_np.dtype, np.integer):
                continue
            fl = feat.lower().replace("_", " ").replace("-", " ")
            tokens = set(fl.split())
            race_scales   = np.array([_RACE_INCOME_SCALE.get(r, 1.0) for r in races])
            gender_scales = np.array([_GENDER_INCOME_SCALE.get(g, 1.0) for g in genders])
            if tokens & _INCOME_KW:
                age_s = np.array([_AGE_INCOME_SCALE.get(a, 1.0) for a in age_groups])
                data[feat] = _scale(arr_np, race_scales * gender_scales * age_s)
            elif tokens & _CREDIT_KW:
                credit_scales = np.array([_RACE_CREDIT_SCALE.get(r, 1.0) for r in races])
                data[feat] = _scale(arr_np, credit_scales)
            elif tokens & _EXP_KW:
                data[feat] = _scale(arr_np, gender_scales)

    # Synthetic target (not a real model feature — used only for cartography baseline)
    probe_target = "_probe_outcome"
    numeric_cols = [k for k, v in data.items() if k in model_feature_cols and np.issubdtype(np.asarray(v).dtype, np.number)]
    if not numeric_cols:
        numeric_cols = [k for k, v in data.items() if np.issubdtype(np.asarray(v).dtype, np.number)]
    if numeric_cols:
        score_col = numeric_cols[0]
        score_arr = np.asarray(data[score_col], dtype=float)
        data[probe_target] = (score_arr > score_arr.mean()).astype(int)
    else:
        data[probe_target] = rng.integers(0, 2, n).astype(int)

    df = pd.DataFrame(data)
    return df, df.to_csv(index=False), detected_protected, probe_target, model_feature_cols
