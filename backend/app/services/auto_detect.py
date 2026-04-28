"""
Auto-Detection Service
======================
Uses Gemini to automatically detect:
1. Which columns are protected attributes (gender, race, age, etc.)
2. Which column is the target/label being predicted
3. What type of dataset this is

This means users never need to manually specify columns.
"""
import pandas as pd
import io
import logging
from typing import Dict, List
from app.services.gemini_client import ask_gemini_json

logger = logging.getLogger(__name__)

# Common protected attribute keywords for fast detection
PROTECTED_KEYWORDS = [
    # Original protected attributes
    'gender', 'sex', 'race', 'ethnicity', 'ethnic', 'nationality', 'nation',
    'religion', 'age', 'disability', 'marital', 'pregnant', 'orientation',
    'colour', 'color', 'caste', 'tribe', 'indigenous', 'veteran',
    # Socioeconomic status
    'income', 'wealth', 'poverty', 'socioeconomic', 'ses', 'economic_class',
    'financial_status', 'low_income', 'welfare', 'benefit',
    # Educational background
    'education', 'degree', 'credential', 'diploma', 'college', 'school',
    'qualification', 'literacy', 'educated',
    # Geographic / zip code (redlining-adjacent)
    'zip', 'zipcode', 'zip_code', 'postal', 'postcode', 'neighbourhood',
    'neighborhood', 'district', 'borough', 'precinct',
    # Language / accent
    'language', 'accent', 'linguistic', 'dialect', 'native_language',
    'first_language', 'english_proficiency',
    # Criminal history
    'criminal', 'conviction', 'convicted', 'arrest', 'arrested', 'felony',
    'misdemeanor', 'offense', 'incarcerated', 'prior_record', 'background_check',
    # Political affiliation
    'political', 'party', 'affiliation', 'ideology', 'partisan',
    # Physical appearance
    'appearance', 'height', 'weight', 'bmi', 'obesity', 'obese',
    'physical_appearance', 'attractiveness',
]

# Common target column keywords
TARGET_KEYWORDS = [
    'hired', 'approved', 'accepted', 'granted', 'label', 'target', 'outcome',
    'decision', 'result', 'status', 'prediction', 'class', 'y', 'output',
    'loan', 'credit', 'risk', 'score', 'admit', 'diagnos', 'fraud',
]


async def auto_detect_columns(dataset_csv: str, audit_id: str = "") -> Dict:
    """
    Auto-detect protected attributes and target column from dataset.
    Uses fast keyword matching first, then Gemini for uncertain cases.
    """
    try:
        df = pd.read_csv(io.StringIO(dataset_csv))
    except Exception as e:
        logger.error(f"Could not parse dataset CSV: {e}")
        return {"protected_cols": [], "target_col": "", "confidence": "low"}

    columns = df.columns.tolist()
    logger.info(f"[{audit_id}] Auto-detecting from {len(columns)} columns: {columns}")

    # Fast keyword detection
    protected_fast = []
    target_fast = ""

    for col in columns:
        col_lower = col.lower().replace('_', ' ').replace('-', ' ')
        if any(kw in col_lower for kw in PROTECTED_KEYWORDS):
            protected_fast.append(col)
        if not target_fast and any(kw in col_lower for kw in TARGET_KEYWORDS):
            target_fast = col

    # If we found enough columns confidently, skip Gemini
    if len(protected_fast) >= 1 and target_fast:
        logger.info(f"[{audit_id}] Fast detection: protected={protected_fast}, target={target_fast}")
        return {
            "protected_cols": protected_fast,
            "target_col": target_fast,
            "confidence": "high",
            "method": "keyword",
        }

    # Use Gemini for uncertain cases
    sample = df.head(5).to_csv(index=False)
    value_counts = {}
    for col in columns[:20]:
        unique = df[col].nunique()
        sample_vals = df[col].dropna().unique()[:5].tolist()
        value_counts[col] = {"unique_count": unique, "sample_values": [str(v) for v in sample_vals]}

    prompt = f"""You are analysing a dataset to find bias. Identify:
1. Which columns contain PROTECTED ATTRIBUTES (demographic info that should not influence decisions):
   - Gender, sex, race, ethnicity, age, nationality, religion, disability, marital status
   - Socioeconomic status: income level, wealth, poverty, financial class
   - Educational background: degree, credential, diploma, qualification
   - Geographic: zip code, postal code, neighbourhood, district (redlining-adjacent proxies)
   - Language / accent: native language, English proficiency, linguistic background
   - Criminal history: prior convictions, arrests, felonies, background check results
   - Political affiliation: party membership, ideology, partisan leanings
   - Physical appearance: height, weight, BMI, obesity
2. Which column is the TARGET variable (what the model predicts / the decision being made):
   - hired, approved, loan_granted, admitted, diagnosed, fraud, etc.

DATASET COLUMNS AND SAMPLE VALUES:
{str(value_counts)}

FIRST 5 ROWS:
{sample}

Return JSON only:
{{
  "protected_cols": ["col1", "col2"],
  "target_col": "col_name",
  "reasoning": "brief explanation",
  "dataset_type": "hiring | lending | healthcare | criminal_justice | other"
}}

If uncertain about a column, include it in protected_cols (better to over-include than miss bias).
If no clear target column, pick the binary/categorical column most likely to be a decision outcome."""

    try:
        result = await ask_gemini_json(prompt)
        logger.info(f"[{audit_id}] Gemini detection: {result}")

        # Merge with fast detection results
        protected = list(set(protected_fast + result.get("protected_cols", [])))
        target = result.get("target_col", target_fast)

        # Validate columns exist in dataset
        protected = [c for c in protected if c in columns]
        if target not in columns:
            target = target_fast or ""

        return {
            "protected_cols": protected,
            "target_col": target,
            "confidence": "high" if protected and target else "medium",
            "method": "gemini",
            "dataset_type": result.get("dataset_type", "other"),
            "reasoning": result.get("reasoning", ""),
        }

    except Exception as e:
        logger.warning(f"[{audit_id}] Gemini auto-detect failed, using keyword results: {e}")
        # Fallback: use any binary column as target if nothing found
        if not target_fast:
            for col in columns:
                if df[col].nunique() == 2:
                    target_fast = col
                    break
        return {
            "protected_cols": protected_fast,
            "target_col": target_fast,
            "confidence": "low",
            "method": "fallback",
        }
