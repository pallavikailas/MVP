"""
FairLens Universal Model Adapter
=================================
The plugin layer that makes FairLens model-agnostic.

Any model — sklearn, PyTorch, TensorFlow, HuggingFace, XGBoost, LightGBM,
a REST API, a LLM, or your own custom class — can be wrapped in a FairLensAdapter
so FairLens can audit it.

Usage:
    from app.services.model_adapter import FairLensAdapter

    # scikit-learn
    adapter = FairLensAdapter.from_sklearn(my_rf_model)

    # PyTorch
    adapter = FairLensAdapter.from_pytorch(my_net, input_size=10)

    # HuggingFace
    adapter = FairLensAdapter.from_huggingface("bert-base-uncased", task="text-classification")

    # REST API (any model behind an endpoint)
    adapter = FairLensAdapter.from_api("https://my-model-api.com/predict")

    # Custom callable
    adapter = FairLensAdapter.from_callable(my_predict_fn, my_proba_fn)

All adapters expose the same interface:
    adapter.predict(X: pd.DataFrame) -> np.ndarray
    adapter.predict_proba(X: pd.DataFrame) -> np.ndarray   # shape (n, 2)
    adapter.get_model_type() -> str
    adapter.supports_shap() -> bool
    adapter.get_shap_explainer(X_background) -> shap.Explainer
"""

from __future__ import annotations

import abc
import logging
import os
from typing import Any, Callable, Optional

import numpy as np
import pandas as pd
import shap

logger = logging.getLogger(__name__)


def normalize_hf_token(token: str = "") -> str:
    """
    Normalise Hugging Face tokens coming from UI forms or env vars.

    Users often paste tokens with a leading "Bearer " prefix or trailing
    newlines/spaces; Hugging Face clients expect just the raw `hf_...` token.
    """
    raw = (token or "").strip()
    if not raw:
        raw = (
            os.getenv("HF_TOKEN", "").strip()
            or os.getenv("HUGGINGFACEHUB_API_TOKEN", "").strip()
            or os.getenv("HUGGINGFACE_API_TOKEN", "").strip()
        )
    if raw.lower().startswith("bearer "):
        raw = raw.split(None, 1)[1].strip() if len(raw.split(None, 1)) > 1 else ""
    return raw


# ── Abstract base ─────────────────────────────────────────────────────────────

class BaseModelAdapter(abc.ABC):
    """
    Abstract interface every FairLens adapter must implement.
    Implement this class to plug ANY model into FairLens.
    """

    @abc.abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions, shape (n,)."""

    @abc.abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability estimates, shape (n, 2) for binary classification."""

    def get_model_type(self) -> str:
        return self.__class__.__name__

    def supports_shap(self) -> bool:
        """Override to False for models that can't use TreeExplainer."""
        return True

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        """
        Return the best SHAP explainer for this model type.
        Override for custom behaviour.
        """
        try:
            return shap.TreeExplainer(self._raw_model())
        except Exception:
            bg = shap.sample(X_background, min(100, len(X_background)))
            return shap.KernelExplainer(self.predict_proba, bg)

    def _raw_model(self) -> Any:
        """Return the underlying model object, if available."""
        raise NotImplementedError


# ── sklearn adapter ───────────────────────────────────────────────────────────

class SklearnAdapter(BaseModelAdapter):
    """
    Wraps any scikit-learn compatible estimator.
    Supports: RandomForest, XGBoost, LightGBM, LogisticRegression,
              SVM, GradientBoosting, CatBoost, and any Pipeline.
    """

    TREE_MODELS = (
        "RandomForest", "GradientBoosting", "XGB", "LGBM",
        "CatBoost", "DecisionTree", "ExtraTrees", "HistGradientBoosting"
    )

    def __init__(self, model: Any):
        self.model = model

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(self._prepare(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_prep = self._prepare(X)
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X_prep)
            if proba.ndim == 1:
                return np.column_stack([1 - proba, proba])
            return proba
        # Decision function fallback (SVM etc.)
        scores = self.model.decision_function(X_prep)
        from scipy.special import expit
        pos = expit(scores)
        return np.column_stack([1 - pos, pos])

    def get_model_type(self) -> str:
        return type(self.model).__name__

    def supports_shap(self) -> bool:
        return True

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        model_name = type(self.model).__name__
        if any(t in model_name for t in self.TREE_MODELS):
            return shap.TreeExplainer(self.model)
        bg = shap.sample(self._prepare(X_background), min(100, len(X_background)))
        return shap.KernelExplainer(self.predict_proba, bg)

    def _raw_model(self) -> Any:
        return self.model

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        """Auto-encode categoricals for models that need numeric input."""
        from sklearn.preprocessing import LabelEncoder
        X_enc = X.copy()
        for col in X_enc.select_dtypes(include=["object", "category"]).columns:
            try:
                X_enc[col] = LabelEncoder().fit_transform(X_enc[col].astype(str))
            except Exception:
                X_enc[col] = 0
        return X_enc.fillna(0)


# ── PyTorch adapter ───────────────────────────────────────────────────────────

class PyTorchAdapter(BaseModelAdapter):
    """
    Wraps a PyTorch nn.Module for binary or multi-class classification.
    The model must accept a float tensor of shape (n, input_size).
    """

    def __init__(self, model: Any, input_size: int, device: str = "cpu", threshold: float = 0.5):
        self.model = model
        self.input_size = input_size
        self.device = device
        self.threshold = threshold
        self.model.eval()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import torch
        import torch.nn.functional as F

        tensor = torch.tensor(
            X.select_dtypes(include=[np.number]).fillna(0).values,
            dtype=torch.float32
        ).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            if out.shape[-1] == 1:
                pos = torch.sigmoid(out).squeeze(-1).cpu().numpy()
                return np.column_stack([1 - pos, pos])
            proba = F.softmax(out, dim=-1).cpu().numpy()
            if proba.shape[1] == 2:
                return proba
            # Multi-class: return [1-max_prob, max_prob] as binary proxy
            max_p = proba.max(axis=1)
            return np.column_stack([1 - max_p, max_p])

    def supports_shap(self) -> bool:
        return True

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        bg = X_background.select_dtypes(include=[np.number]).fillna(0).values[:100]
        return shap.GradientExplainer(self.model, bg)

    def get_model_type(self) -> str:
        return f"PyTorch:{type(self.model).__name__}"


# ── TensorFlow / Keras adapter ────────────────────────────────────────────────

class TensorFlowAdapter(BaseModelAdapter):
    """
    Wraps a tf.keras.Model or any TensorFlow SavedModel.
    """

    def __init__(self, model: Any, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= self.threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import tensorflow as tf
        arr = X.select_dtypes(include=[np.number]).fillna(0).values.astype(np.float32)
        out = self.model.predict(arr, verbose=0)
        if out.ndim == 1 or out.shape[-1] == 1:
            pos = out.flatten()
            return np.column_stack([1 - pos, pos])
        return out[:, :2]

    def supports_shap(self) -> bool:
        return True

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        bg = X_background.select_dtypes(include=[np.number]).fillna(0).values[:100]
        return shap.GradientExplainer(self.model, bg)

    def get_model_type(self) -> str:
        return f"TensorFlow:{type(self.model).__name__}"


# ── HuggingFace adapter ───────────────────────────────────────────────────────

class HuggingFaceAdapter(BaseModelAdapter):
    """
    Wraps a HuggingFace text-classification model via the Inference API (HTTP).
    No local model download — safe for Cloud Run.
    For tabular DataFrames without a 'text' column, rows are serialised to
    "key: value" strings so any text classifier can score them.
    """

    # New HF Inference Providers URL (requires token); legacy URL as fallback
    _HF_INFERENCE_URL = "https://router.huggingface.co/hf-inference/models/{model}"
    _HF_LEGACY_URL = "https://api-inference.huggingface.co/models/{model}"

    def __init__(self, model_name: Any, task: str = "text-classification", hf_token: str = ""):
        # Accept a string model name or a pre-built pipeline object (legacy path)
        if isinstance(model_name, str):
            self.model_name = model_name
            self._pipeline = None
        else:
            # Pre-built pipeline passed directly — use locally (backward-compat)
            self.model_name = getattr(model_name, "model", "unknown")
            self._pipeline = model_name
        self.task = task
        self.hf_token = normalize_hf_token(hf_token)

    @staticmethod
    def _to_text(X: pd.DataFrame) -> list[str]:
        """Convert a DataFrame row to a text string the classifier can score."""
        if "text" in X.columns:
            return X["text"].fillna("").tolist()
        # Serialise tabular row as "col: value, col: value, ..."
        return [
            ", ".join(f"{col}: {val}" for col, val in row.items() if pd.notna(val))
            for _, row in X.iterrows()
        ]

    def _query_api(self, texts: list[str]) -> list[dict]:
        """
        Call HF Inference API via huggingface_hub.InferenceClient (primary path).
        Falls back to raw HTTP if the library is unavailable.
        InferenceClient handles URL routing, auth, and model-type detection correctly,
        avoiding the 404s caused by the new Inference Providers API path changes.
        """
        fallback_to_raw_http = False
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(
                model=self.model_name,
                token=self.hf_token if self.hf_token else None,
                timeout=90,
            )
            results = []
            for text in texts:
                try:
                    raw = client.text_classification(text)
                    # raw is a list of ClassificationOutput(label, score)
                    results.append([{"label": r.label, "score": r.score} for r in raw])
                except Exception as item_err:
                    err_str = str(item_err).lower()
                    if "401" in err_str or "403" in err_str or "unauthorized" in err_str or "forbidden" in err_str:
                        # InferenceClient may fail auth/provider lookup for some gated
                        # or provider-routed models even when direct bearer-auth HTTP works.
                        fallback_to_raw_http = True
                        break
                    if "404" in err_str or "not found" in err_str:
                        raise ValueError(
                            f"HuggingFace model '{self.model_name}' not found or not available "
                            "on the Inference API. Check the model ID at huggingface.co."
                        )
                    if "503" in err_str or "loading" in err_str:
                        raise RuntimeError(
                            f"HuggingFace model '{self.model_name}' is loading — wait ~20s and retry."
                        )
                    raise RuntimeError(
                        f"HuggingFace classification failed for '{self.model_name}': {item_err}"
                    )
            if not fallback_to_raw_http:
                return results

        except (ImportError, ModuleNotFoundError):
            pass  # huggingface_hub not installed — fall through to raw HTTP

        # ── Raw HTTP fallback (legacy path) ──────────────────────────────────
        import requests
        headers = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"

        legacy_url = self._HF_LEGACY_URL.format(model=self.model_name)
        results = []
        batch_size = 8
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            resp = requests.post(legacy_url, json={"inputs": batch}, headers=headers, timeout=90)
            if resp.status_code == 503:
                raise RuntimeError(f"HuggingFace model '{self.model_name}' is loading — wait ~20s and retry.")
            if resp.status_code in (401, 403):
                raise ValueError(
                    f"HuggingFace model '{self.model_name}' requires authentication. "
                    "Enter your HF token (hf_...) in the token field."
                )
            if resp.status_code == 404:
                raise ValueError(
                    f"HuggingFace model '{self.model_name}' not found on the Inference API. "
                    "Check the model ID at huggingface.co."
                )
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise ValueError(
                    f"Unexpected HuggingFace response format for '{self.model_name}': {type(data).__name__}"
                )
            if len(data) == 1 and isinstance(data[0], list) and len(data[0]) == len(batch):
                items = data[0]
            elif len(data) == len(batch):
                items = data
            else:
                items = data
            results.extend([item if isinstance(item, list) else [item] for item in items])
        return results

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        texts = self._to_text(X)

        if self._pipeline is not None:
            # Legacy local pipeline path
            raw = self._pipeline(texts, truncation=True, max_length=512)
        else:
            raw = self._query_api(texts)

        # Exact-match sets avoid "NON_TOXIC" matching the "TOXIC" substring check
        _NEG_LABELS = {
            "NEG", "NEGATIVE", "LABEL_0", "NON_TOXIC", "NONTOXIC", "NOT_TOXIC",
            "SAFE", "NOT_HATE", "NOTHATE", "CLEAN", "HAM", "0", "FALSE", "NORMAL",
            "NON-TOXIC", "NOT TOXIC",
        }
        _POS_LABELS = {
            "POS", "POSITIVE", "LABEL_1", "TOXIC", "HATE", "HARMFUL",
            "OFFENSIVE", "SPAM", "JUNK", "1", "TRUE", "ABNORMAL",
        }

        probas = []
        for item in raw:
            candidates = item if isinstance(item, list) else [item]
            pos_score = None
            for c in candidates:
                label_norm = c.get("label", "").upper().replace("-", "_").replace(" ", "_")
                score = float(c.get("score", 0.5))
                if label_norm in _NEG_LABELS:
                    pos_score = 1.0 - score  # negative class → positive prob is complement
                    break
                if label_norm in _POS_LABELS:
                    pos_score = score
                    break
            if pos_score is None:
                raise ValueError(
                    f"Could not map HuggingFace labels to a binary decision for '{self.model_name}': "
                    f"{[c.get('label', '') for c in candidates]}"
                )
            probas.append([1 - pos_score, pos_score])
        return np.array(probas)

    def supports_shap(self) -> bool:
        return False

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        return shap.Explainer(
            lambda texts: self.predict_proba(pd.DataFrame({"text": texts})),
            self._to_text(X_background)[:10],
            output_names=["negative", "positive"],
        )

    def get_model_type(self) -> str:
        return "HuggingFace"


# ── Generative LLM adapter ────────────────────────────────────────────────────

class GenerativeLLMAdapter(BaseModelAdapter):
    """
    Wraps any generative LLM for bias auditing via decision prompts.

    Converts each tabular row into a decision scenario, asks the model to
    respond YES/NO, and returns predict_proba from that answer.
    Works with:
      - HuggingFace text-generation (Gemma, Llama, Mistral, etc.)
      - OpenAI  (gpt-4o, gpt-4, gpt-3.5-turbo, etc.)
      - Gemini  (gemini-1.5-flash, gemini-2.0-flash, etc.)

    Usage:
        adapter = GenerativeLLMAdapter(backend="openai",   model_name="gpt-4o",              api_key="sk-...")
        adapter = GenerativeLLMAdapter(backend="huggingface", model_name="google/gemma-3-1b-it", hf_token="hf_...")
        adapter = GenerativeLLMAdapter(backend="gemini",   model_name="gemini-2.0-flash",    api_key="AIza...")
    """

    _DEFAULT_PROMPT = (
        "You are an impartial decision-maker. Based only on the profile below, "
        "give a single-word decision: YES or NO.\n\n"
        "Profile:\n{profile}\n\n"
        "Decision (YES or NO):"
    )

    _HF_INFERENCE_URL = "https://router.huggingface.co/hf-inference/models/{model}"
    _HF_LEGACY_URL = "https://api-inference.huggingface.co/models/{model}"

    def __init__(
        self,
        backend: str,                        # "openai" | "huggingface" | "gemini"
        model_name: str,
        api_key: str = "",
        hf_token: str = "",
        prompt_template: str = "",
        max_new_tokens: int = 20,
        positive_threshold: float = 0.5,
    ):
        self.backend = backend
        self.model_name = model_name
        self.api_key = api_key
        self.hf_token = normalize_hf_token(hf_token)
        self.prompt_template = prompt_template or self._DEFAULT_PROMPT
        self.max_new_tokens = max_new_tokens
        self.positive_threshold = positive_threshold
        # HuggingFace uses the Inference API — no local pipeline/weights download

    # ── Public interface ──────────────────────────────────────────────────────

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self.positive_threshold).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        probas = []
        for _, row in X.iterrows():
            p = self._query(row)  # raises ValueError on permanent failures (404 etc.)
            probas.append([1.0 - p, p])
        return np.array(probas)

    def supports_shap(self) -> bool:
        return False

    def get_model_type(self) -> str:
        return f"GenerativeLLM:{self.backend}:{self.model_name}"

    # ── Prompt helpers ────────────────────────────────────────────────────────

    def _build_prompt(self, row: pd.Series) -> str:
        profile = "\n".join(f"  {k}: {v}" for k, v in row.items())
        return self.prompt_template.format(profile=profile)

    def _parse_response(self, text: str) -> float:
        """Return 0.9 for positive-class signals, 0.1 for negative, 0.5 for ambiguous."""
        t = text.lower().strip().split()[0] if text.strip() else ""
        if any(w in t for w in ("yes", "approv", "accept", "hire", "grant", "admit", "positive", "true", "1")):
            return 0.9
        if any(w in t for w in ("no", "reject", "deny", "declin", "negative", "false", "0")):
            return 0.1
        # Scan full text as fallback
        full = text.lower()
        pos = sum(full.count(w) for w in ("yes", "approv", "accept", "hire", "grant"))
        neg = sum(full.count(w) for w in ("no", "reject", "deny", "declin", "refused"))
        if pos > neg:
            return 0.75
        if neg > pos:
            return 0.25
        return 0.5

    # ── Backend query methods ─────────────────────────────────────────────────

    def _query(self, row: pd.Series) -> float:
        prompt = self._build_prompt(row)
        try:
            if self.backend == "openai":
                return self._query_openai(prompt)
            if self.backend == "huggingface":
                return self._query_huggingface(prompt)
            if self.backend == "gemini":
                return self._query_gemini(prompt)
        except (ValueError, RuntimeError):
            # Hard/permanent or service-level errors should abort the run so callers
            # can surface a single actionable message instead of silently returning 0.5
            raise
        except Exception as exc:
            logger.warning(f"[GenerativeLLMAdapter] query failed: {exc}")
            raise RuntimeError(f"Model query failed for '{self.model_name}': {exc}") from exc

    def _query_openai(self, prompt: str) -> float:
        import openai
        client = openai.OpenAI(api_key=self.api_key)
        resp = client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_new_tokens,
            temperature=0,
        )
        return self._parse_response(resp.choices[0].message.content or "")

    def _query_huggingface(self, prompt: str) -> float:
        """Call HuggingFace Inference API via InferenceClient (handles routing automatically)."""
        try:
            from huggingface_hub import InferenceClient
            client = InferenceClient(
                model=self.model_name,
                token=self.hf_token if self.hf_token else None,
                timeout=90,
            )
            try:
                result = client.text_generation(
                    prompt,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                )
                return self._parse_response(result if isinstance(result, str) else str(result))
            except Exception as exc:
                err_str = str(exc).lower()
                if "429" in err_str or "too many requests" in err_str or "rate limit" in err_str:
                    raise ValueError(
                        f"HuggingFace rate limit reached for '{self.model_name}'. "
                        "Cross-analysis on generative HF models makes many inference calls; "
                        "retry later, use a valid HF token with higher limits, or switch to a text-classification model."
                    )
                if "401" in err_str or "403" in err_str or "unauthorized" in err_str:
                    raise ValueError(
                        f"HuggingFace model '{self.model_name}' requires authentication. "
                        "Enter your HF token in the 'HuggingFace Token' field."
                    )
                if "404" in err_str or "not found" in err_str:
                    raise ValueError(
                        f"Model '{self.model_name}' not found on HuggingFace Inference API. "
                        "Check the model ID at huggingface.co."
                    )
                if "503" in err_str or "loading" in err_str:
                    raise RuntimeError(
                        f"HuggingFace model '{self.model_name}' is loading — wait ~20s and retry."
                    )
                if "not supported" in err_str or "task" in err_str:
                    raise ValueError(
                        f"HuggingFace model '{self.model_name}' does not support text-generation. "
                        "Try a text-generation model like google/flan-t5-base or tiiuae/falcon-7b-instruct."
                    )
                raise
        except (ImportError, ModuleNotFoundError):
            pass

        # Legacy raw HTTP fallback (only if huggingface_hub unavailable)
        import requests
        headers = {"Content-Type": "application/json"}
        if self.hf_token:
            headers["Authorization"] = f"Bearer {self.hf_token}"
        url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        resp = requests.post(
            url,
            json={"inputs": prompt, "parameters": {"max_new_tokens": self.max_new_tokens, "return_full_text": False}},
            headers=headers,
            timeout=90,
        )
        if resp.status_code == 429:
            raise ValueError(
                f"HuggingFace rate limit reached for '{self.model_name}'. "
                "Cross-analysis on generative HF models makes many inference calls; "
                "retry later, use a valid HF token with higher limits, or switch to a text-classification model."
            )
        if resp.status_code == 503:
            raise RuntimeError(f"HuggingFace model '{self.model_name}' is loading — wait ~20s and retry.")
        if resp.status_code in (401, 403):
            raise ValueError(f"HuggingFace model '{self.model_name}' requires authentication.")
        if resp.status_code == 404:
            raise ValueError(f"Model '{self.model_name}' not found on HuggingFace Inference API.")
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            generated = data[0].get("generated_text", "")
        elif isinstance(data, dict):
            generated = data.get("generated_text", "")
        else:
            generated = str(data)
        return self._parse_response(generated)

    def _query_gemini(self, prompt: str) -> float:
        from google import genai
        client = genai.Client(api_key=self.api_key)
        resp = client.models.generate_content(model=self.model_name, contents=prompt)
        return self._parse_response(resp.text or "")


# ── REST API adapter ──────────────────────────────────────────────────────────

class RESTAPIAdapter(BaseModelAdapter):
    """
    Wraps any model served behind a REST endpoint.
    Sends rows as JSON and parses the response.

    Expected API contract:
        POST /predict
        Body: {"instances": [[f1, f2, ...], ...]}
        Response: {"predictions": [0, 1, ...], "probabilities": [[0.3,0.7], ...]}

    Override _format_request / _parse_response to match your API's schema.
    """

    def __init__(
        self,
        endpoint: str,
        headers: Optional[dict] = None,
        timeout: int = 30,
        request_format: str = "instances",   # or "data", "inputs" etc.
        auth_token: Optional[str] = None,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.headers = headers or {"Content-Type": "application/json"}
        if auth_token:
            self.headers["Authorization"] = f"Bearer {auth_token}"
        self.timeout = timeout
        self.request_format = request_format

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        import requests
        payload = self._format_request(X)
        resp = requests.post(
            f"{self.endpoint}/predict",
            json=payload,
            headers=self.headers,
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return self._parse_response(resp.json(), len(X))

    def supports_shap(self) -> bool:
        return False  # Use KernelExplainer

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        bg = shap.sample(X_background, min(50, len(X_background)))
        return shap.KernelExplainer(self.predict_proba, bg)

    def _format_request(self, X: pd.DataFrame) -> dict:
        return {self.request_format: X.fillna(0).values.tolist()}

    def _parse_response(self, response: dict, n: int) -> np.ndarray:
        if "probabilities" in response:
            proba = np.array(response["probabilities"])
            if proba.ndim == 1:
                return np.column_stack([1 - proba, proba])
            return proba
        if "predictions" in response:
            preds = np.array(response["predictions"])
            return np.column_stack([1 - preds, preds.astype(float)])
        if "scores" in response:
            from scipy.special import expit
            scores = expit(np.array(response["scores"]))
            return np.column_stack([1 - scores, scores])
        raise ValueError(f"Unrecognised API response format: {list(response.keys())}")

    def get_model_type(self) -> str:
        return f"REST:{self.endpoint}"


# ── Callable adapter ──────────────────────────────────────────────────────────

class CallableAdapter(BaseModelAdapter):
    """
    Wraps any Python callable as a FairLens model.
    Useful for custom ensembles, business-rule systems, or any predict function.

    adapter = FairLensAdapter.from_callable(
        predict_fn=lambda X: my_model.predict(X),
        predict_proba_fn=lambda X: my_model.predict_proba(X),
    )
    """

    def __init__(
        self,
        predict_fn: Callable[[pd.DataFrame], np.ndarray],
        predict_proba_fn: Optional[Callable[[pd.DataFrame], np.ndarray]] = None,
        model_name: str = "CustomCallable",
    ):
        self._predict_fn = predict_fn
        self._predict_proba_fn = predict_proba_fn
        self._name = model_name

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.array(self._predict_fn(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self._predict_proba_fn:
            proba = np.array(self._predict_proba_fn(X))
            if proba.ndim == 1:
                return np.column_stack([1 - proba, proba])
            return proba
        preds = self.predict(X).astype(float)
        return np.column_stack([1 - preds, preds])

    def supports_shap(self) -> bool:
        return True

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        bg = shap.sample(X_background, min(100, len(X_background)))
        return shap.KernelExplainer(self.predict_proba, bg)

    def get_model_type(self) -> str:
        return self._name


# ── Vertex AI / GCP Model adapter ─────────────────────────────────────────────

class VertexAIAdapter(BaseModelAdapter):
    """
    Wraps a deployed Vertex AI Endpoint.
    Calls the endpoint via google-cloud-aiplatform SDK.
    """

    def __init__(self, endpoint_id: str, project: str, location: str = "us-central1"):
        from google.cloud import aiplatform
        aiplatform.init(project=project, location=location)
        self.endpoint = aiplatform.Endpoint(endpoint_id)
        self.project = project
        self.location = location

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        instances = X.fillna(0).to_dict(orient="records")
        response = self.endpoint.predict(instances=instances)
        preds = np.array(response.predictions)
        if preds.ndim == 1:
            return np.column_stack([1 - preds, preds])
        return preds

    def supports_shap(self) -> bool:
        return False

    def get_shap_explainer(self, X_background: pd.DataFrame) -> shap.Explainer:
        bg = shap.sample(X_background, min(50, len(X_background)))
        return shap.KernelExplainer(self.predict_proba, bg)

    def get_model_type(self) -> str:
        return f"VertexAI:{self.endpoint.resource_name}"


# ── Main FairLensAdapter factory ──────────────────────────────────────────────

class FairLensAdapter:
    """
    Factory class and public API for the FairLens plugin system.

    Usage examples:

        # scikit-learn / XGBoost / LightGBM / CatBoost
        adapter = FairLensAdapter.from_sklearn(model)

        # PyTorch
        adapter = FairLensAdapter.from_pytorch(net, input_size=20)

        # TensorFlow/Keras
        adapter = FairLensAdapter.from_tensorflow(keras_model)

        # HuggingFace
        adapter = FairLensAdapter.from_huggingface("distilbert-base-uncased-finetuned-sst-2-english")

        # Any REST API
        adapter = FairLensAdapter.from_api("https://my-model.run.app", auth_token="abc123")

        # Any Python callable
        adapter = FairLensAdapter.from_callable(predict_fn, predict_proba_fn)

        # Vertex AI deployed endpoint
        adapter = FairLensAdapter.from_vertex_ai("1234567890", project="my-gcp-project")

        # Auto-detect from a .pkl file
        adapter = FairLensAdapter.from_pickle("model.pkl")
    """

    @staticmethod
    def from_sklearn(model: Any) -> SklearnAdapter:
        """Wrap any scikit-learn compatible model."""
        logger.info(f"Creating SklearnAdapter for {type(model).__name__}")
        return SklearnAdapter(model)

    @staticmethod
    def from_pytorch(model: Any, input_size: int, device: str = "cpu") -> PyTorchAdapter:
        """Wrap a PyTorch nn.Module."""
        logger.info(f"Creating PyTorchAdapter for {type(model).__name__}")
        return PyTorchAdapter(model, input_size=input_size, device=device)

    @staticmethod
    def from_tensorflow(model: Any) -> TensorFlowAdapter:
        """Wrap a Keras/TensorFlow model."""
        logger.info(f"Creating TensorFlowAdapter for {type(model).__name__}")
        return TensorFlowAdapter(model)

    @staticmethod
    def from_huggingface(
        model_name_or_pipeline: Any,
        task: str = "text-classification",
        hf_token: str = "",
    ) -> HuggingFaceAdapter:
        """Wrap a HuggingFace text-classification model via the Inference API."""
        logger.info(f"Creating HuggingFaceAdapter (Inference API)")
        return HuggingFaceAdapter(model_name_or_pipeline, task=task, hf_token=hf_token)

    @staticmethod
    def from_huggingface_auto(
        model_name: str,
        hf_token: str = "",
    ) -> "BaseModelAdapter":
        """
        Auto-detect whether model_name is a text-classification or generative model
        by querying the HuggingFace Hub API, then return the right adapter.
        """
        import requests
        hf_token = normalize_hf_token(hf_token)
        headers = {"Content-Type": "application/json"}
        if hf_token:
            headers["Authorization"] = f"Bearer {hf_token}"
        try:
            resp = requests.get(
                f"https://huggingface.co/api/models/{model_name}",
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                pipeline_tag = resp.json().get("pipeline_tag", "")
            else:
                pipeline_tag = ""
        except Exception:
            pipeline_tag = ""

        GENERATIVE_TASKS = {"text-generation", "text2text-generation", "conversational", "summarization"}
        if pipeline_tag in GENERATIVE_TASKS:
            logger.info(f"Auto-detected '{model_name}' as generative ({pipeline_tag}) → GenerativeLLMAdapter")
            return GenerativeLLMAdapter(backend="huggingface", model_name=model_name, hf_token=hf_token)
        else:
            logger.info(f"Auto-detected '{model_name}' as classifier ({pipeline_tag or 'unknown'}) → HuggingFaceAdapter")
            return HuggingFaceAdapter(model_name, task="text-classification", hf_token=hf_token)

    @staticmethod
    def from_api(
        endpoint: str,
        headers: Optional[dict] = None,
        auth_token: Optional[str] = None,
        request_format: str = "instances",
    ) -> RESTAPIAdapter:
        """Wrap any REST API endpoint."""
        logger.info(f"Creating RESTAPIAdapter for {endpoint}")
        return RESTAPIAdapter(endpoint, headers=headers, auth_token=auth_token, request_format=request_format)

    @staticmethod
    def from_callable(
        predict_fn: Callable,
        predict_proba_fn: Optional[Callable] = None,
        model_name: str = "CustomCallable",
    ) -> CallableAdapter:
        """Wrap any Python predict function."""
        return CallableAdapter(predict_fn, predict_proba_fn, model_name)

    @staticmethod
    def from_openai(
        model_name: str = "gpt-4o",
        api_key: str = "",
        prompt_template: str = "",
    ) -> "GenerativeLLMAdapter":
        """Wrap OpenAI ChatGPT / GPT-4 for decision-prompt bias auditing."""
        logger.info(f"Creating GenerativeLLMAdapter (OpenAI:{model_name})")
        return GenerativeLLMAdapter(backend="openai", model_name=model_name, api_key=api_key, prompt_template=prompt_template)

    @staticmethod
    def from_generative_huggingface(
        model_name: str,
        hf_token: str = "",
        prompt_template: str = "",
    ) -> "GenerativeLLMAdapter":
        """Wrap a HuggingFace generative model (Gemma, Llama, Mistral, etc.)."""
        logger.info(f"Creating GenerativeLLMAdapter (HuggingFace:{model_name})")
        return GenerativeLLMAdapter(backend="huggingface", model_name=model_name, hf_token=hf_token, prompt_template=prompt_template)

    @staticmethod
    def from_gemini(
        model_name: str = "gemini-2.0-flash",
        api_key: str = "",
        prompt_template: str = "",
    ) -> "GenerativeLLMAdapter":
        """Wrap a Gemini model for decision-prompt bias auditing."""
        logger.info(f"Creating GenerativeLLMAdapter (Gemini:{model_name})")
        return GenerativeLLMAdapter(backend="gemini", model_name=model_name, api_key=api_key, prompt_template=prompt_template)

    @staticmethod
    def from_vertex_ai(endpoint_id: str, project: str, location: str = "us-central1") -> VertexAIAdapter:
        """Wrap a Vertex AI deployed endpoint."""
        return VertexAIAdapter(endpoint_id, project=project, location=location)

    @staticmethod
    def from_pickle(path: str) -> SklearnAdapter:
        """Load a .pkl file and auto-wrap it."""
        import pickle
        with open(path, "rb") as f:
            model = pickle.load(f)
        logger.info(f"Loaded model from {path}: {type(model).__name__}")
        return SklearnAdapter(model)

    @staticmethod
    def auto_detect(model: Any) -> BaseModelAdapter:
        """
        Auto-detect model type and return the appropriate adapter.
        Falls back to SklearnAdapter (which handles most cases).
        """
        class_name = type(model).__name__
        module = type(model).__module__ or ""

        if "torch" in module or "pytorch" in module.lower():
            raise ValueError("PyTorch models need input_size — use FairLensAdapter.from_pytorch(model, input_size=N)")

        if "keras" in module or "tensorflow" in module:
            return TensorFlowAdapter(model)

        if "transformers" in module:
            return HuggingFaceAdapter(model)

        # Default: treat as sklearn-compatible
        return SklearnAdapter(model)
