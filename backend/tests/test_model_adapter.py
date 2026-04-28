"""Tests for FairLens Universal Model Adapter."""
import pytest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.services.model_adapter import (
    FairLensAdapter,
    SklearnAdapter,
    CallableAdapter,
    HuggingFaceAdapter,
    normalize_hf_token,
)


@pytest.fixture
def binary_data():
    np.random.seed(0)
    X = pd.DataFrame({"age": np.random.randint(20, 60, 200), "score": np.random.rand(200), "gender": np.random.choice(["M", "F"], 200)})
    y = (X["score"] + np.where(X["gender"] == "M", 0.2, 0)).gt(0.5).astype(int)
    return X, y.values


@pytest.fixture
def trained_rf(binary_data):
    from sklearn.preprocessing import LabelEncoder
    X, y = binary_data
    Xe = X.copy()
    Xe["gender"] = LabelEncoder().fit_transform(Xe["gender"])
    clf = RandomForestClassifier(n_estimators=20, random_state=0)
    clf.fit(Xe, y)
    return clf, Xe


def test_sklearn_adapter_predict(trained_rf):
    model, X = trained_rf
    adapter = FairLensAdapter.from_sklearn(model)
    preds = adapter.predict(X)
    assert preds.shape == (len(X),)
    assert set(preds).issubset({0, 1})


def test_sklearn_adapter_predict_proba(trained_rf):
    model, X = trained_rf
    adapter = FairLensAdapter.from_sklearn(model)
    proba = adapter.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)


def test_sklearn_adapter_shap(trained_rf):
    model, X = trained_rf
    adapter = FairLensAdapter.from_sklearn(model)
    explainer = adapter.get_shap_explainer(X)
    assert explainer is not None


def test_callable_adapter():
    X = pd.DataFrame({"a": [1, 2, 3], "b": [0.1, 0.8, 0.4]})

    def my_predict(df):
        return (df["b"] > 0.5).astype(int).values

    def my_proba(df):
        p = df["b"].values
        return np.column_stack([1 - p, p])

    adapter = FairLensAdapter.from_callable(my_predict, my_proba, "MyModel")
    assert adapter.predict(X).tolist() == [0, 1, 0]
    proba = adapter.predict_proba(X)
    assert proba.shape == (3, 2)
    assert adapter.get_model_type() == "MyModel"


def test_auto_detect_sklearn(trained_rf):
    model, X = trained_rf
    adapter = FairLensAdapter.auto_detect(model)
    assert isinstance(adapter, SklearnAdapter)
    preds = adapter.predict(X)
    assert len(preds) == len(X)


def test_pickle_roundtrip(trained_rf, tmp_path):
    import pickle
    model, X = trained_rf
    pkl_path = tmp_path / "model.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(model, f)
    adapter = FairLensAdapter.from_pickle(str(pkl_path))
    preds = adapter.predict(X)
    assert len(preds) == len(X)


def test_normalize_hf_token_strips_bearer_and_whitespace(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    assert normalize_hf_token("  Bearer hf_test_token  \n") == "hf_test_token"


def test_normalize_hf_token_uses_env_fallback(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", " Bearer hf_env_token ")
    assert normalize_hf_token("") == "hf_env_token"


def test_huggingface_adapter_stores_normalized_token():
    adapter = HuggingFaceAdapter("cardiffnlp/twitter-roberta-base-sentiment-latest", hf_token=" Bearer hf_inline ")
    assert adapter.hf_token == "hf_inline"


def test_from_huggingface_auto_uses_normalized_auth_header(monkeypatch):
    captured = {}

    class DummyResponse:
        status_code = 200

        @staticmethod
        def json():
            return {"pipeline_tag": "text-classification"}

    def fake_get(url, headers, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["timeout"] = timeout
        return DummyResponse()

    import requests
    monkeypatch.setattr(requests, "get", fake_get)

    adapter = FairLensAdapter.from_huggingface_auto(
        "cardiffnlp/twitter-roberta-base-sentiment-latest",
        hf_token=" Bearer hf_header_test ",
    )

    assert isinstance(adapter, HuggingFaceAdapter)
    assert adapter.hf_token == "hf_header_test"
    assert captured["headers"]["Authorization"] == "Bearer hf_header_test"


def test_generative_hf_429_is_raised_as_actionable_value_error(monkeypatch):
    adapter = FairLensAdapter.from_generative_huggingface(
        "EleutherAI/gpt-neo-1.3B",
        hf_token="hf_test",
    )

    class FakeInferenceClient:
        def __init__(self, *args, **kwargs):
            pass

        def text_generation(self, *args, **kwargs):
            raise Exception("429 Too Many Requests: you have reached your api rate limit.")

    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "InferenceClient", FakeInferenceClient)

    with pytest.raises(ValueError, match="rate limit reached"):
        adapter.predict_proba(pd.DataFrame({"text": ["hello world"]}))


def test_classifier_hf_auth_error_falls_back_to_raw_http(monkeypatch):
    adapter = HuggingFaceAdapter(
        "Hate-speech-CNERG/dehatebert-mono-english",
        hf_token="hf_test",
    )

    class FakeInferenceClient:
        def __init__(self, *args, **kwargs):
            pass

        def text_classification(self, *args, **kwargs):
            raise Exception("401 Unauthorized")

    class FakeResponse:
        status_code = 200

        @staticmethod
        def json():
            return [[{"label": "LABEL_1", "score": 0.91}]]

        @staticmethod
        def raise_for_status():
            return None

    import huggingface_hub
    import requests

    captured = {}

    def fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(huggingface_hub, "InferenceClient", FakeInferenceClient)
    monkeypatch.setattr(requests, "post", fake_post)

    probs = adapter.predict_proba(pd.DataFrame({"text": ["hello world"]}))

    assert probs.shape == (1, 2)
    assert probs[0, 1] == pytest.approx(0.91)
    assert captured["headers"]["Authorization"] == "Bearer hf_test"
