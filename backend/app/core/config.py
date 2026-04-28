"""FairLens config — reads from environment variables."""
from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    APP_NAME: str = "FairLens"
    DEBUG: bool = False
    ENVIRONMENT: str = "production"

    # GCP
    GOOGLE_CLOUD_PROJECT: str = "fairlens-493318"

    # Vertex AI — used for Gemini 2.5 Flash and text embeddings
    VERTEX_AI_LOCATION: str = "us-central1"
    GEMINI_MODEL: str = "gemini-2.5-flash"

    # Direct Gemini API key (for local dev — no GCP needed)
    # Get a free key at https://aistudio.google.com/apikey
    GEMINI_API_KEY: str = ""  # optional — Vertex AI ADC is used by default

    # Red-team agent
    REDTEAM_MAX_ITERATIONS: int = 3
    REDTEAM_BATCH_SIZE: int = 100

    # Fairness thresholds (industry standard)
    DEMOGRAPHIC_PARITY_THRESHOLD: float = 0.1
    DISPARATE_IMPACT_THRESHOLD: float = 0.8
    EQUAL_OPPORTUNITY_THRESHOLD: float = 0.1   # max allowed TPR difference across groups
    EQUALIZED_ODDS_THRESHOLD: float = 0.1      # max allowed max(|TPR diff|, |FPR diff|)

    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "https://fairlens.web.app",
        "https://fairlens-frontend-nrk2z2yadq-uc.a.run.app",
        "https://pallavikailas.github.io",
    ]

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"


settings = Settings()
