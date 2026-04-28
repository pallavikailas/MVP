"""Tests for Bias Cartography Service."""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from unittest.mock import patch, AsyncMock

from app.services.cartography import BiasCartographyService


@pytest.fixture
def sample_data():
    """Synthetic hiring dataset with known gender bias."""
    np.random.seed(42)
    n = 500
    gender = np.random.choice(["male", "female"], n)
    experience = np.random.randint(1, 15, n)
    education = np.random.choice(["bachelors", "masters", "phd"], n)

    # Inject bias: males 30% more likely to be hired all else equal
    bias_factor = np.where(gender == "male", 0.3, 0.0)
    prob = np.clip(0.3 + 0.05 * experience + bias_factor, 0, 1)
    hired = (np.random.rand(n) < prob).astype(int)

    df = pd.DataFrame({"gender": gender, "experience": experience, "education": education, "hired": hired})
    return df


@pytest.fixture
def trained_model(sample_data):
    from sklearn.preprocessing import LabelEncoder
    X = sample_data.drop(columns=["hired"])
    y = sample_data["hired"]
    le = LabelEncoder()
    X_enc = X.copy()
    for col in X_enc.select_dtypes(include="object").columns:
        X_enc[col] = le.fit_transform(X_enc[col])
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_enc, y)
    return clf, X_enc, y.values


@pytest.mark.asyncio
async def test_cartography_detects_gender_bias(trained_model, sample_data):
    model, X_enc, y = trained_model
    service = BiasCartographyService()

    with patch.object(service, "_log_to_bigquery", new_callable=AsyncMock):
        result = await service.run_cartography(
            model=model,
            X=X_enc,
            y_pred=model.predict(X_enc),
            y_true=y,
            protected_cols=["gender"],
            audit_id="test-001",
        )

    assert "map_points" in result
    assert "hotspots" in result
    assert "slice_metrics" in result
    assert result["summary"]["total_samples"] == len(X_enc)
    # Expect at least one flagged slice given injected bias
    flagged = [m for m in result["slice_metrics"] if m.get("flagged")]
    assert len(flagged) > 0, "Expected at least one flagged slice given injected gender bias"


def test_intersectional_slices_created(sample_data):
    service = BiasCartographyService()
    from sklearn.preprocessing import LabelEncoder
    X = sample_data.drop(columns=["hired"])
    y = sample_data["hired"].values
    X_enc = X.copy()
    le = LabelEncoder()
    for col in X_enc.select_dtypes(include="object").columns:
        X_enc[col] = le.fit_transform(X_enc[col])
    y_pred = np.random.randint(0, 2, len(X))

    slices = service._build_intersectional_slices(X, y_pred, y, ["gender", "education"])
    intersectional = [s for s in slices if s["type"] == "intersectional"]
    assert len(intersectional) > 0, "Expected intersectional slices for gender × education"


def test_bias_residuals_nonzero(sample_data):
    service = BiasCartographyService()
    X = sample_data.drop(columns=["hired"])
    y_pred = np.random.randint(0, 2, len(X))
    residuals = service._compute_bias_residuals(X, y_pred, None, ["gender"])
    assert residuals.sum() > 0
