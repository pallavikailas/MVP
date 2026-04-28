"""Health and readiness endpoints for Cloud Run."""
from fastapi import APIRouter
from datetime import datetime
from app.core.config import settings

router = APIRouter()


@router.get("/")
async def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@router.get("/ready")
async def ready():
    return {"status": "ready"}


@router.get("/gemini")
async def gemini_diagnostic():
    """Test Vertex AI / Gemini connectivity. Visit /health/gemini to debug."""
    try:
        import vertexai
        from vertexai.generative_models import GenerativeModel
        vertexai.init(project=settings.GOOGLE_CLOUD_PROJECT, location=settings.VERTEX_AI_LOCATION)
        model = GenerativeModel(settings.GEMINI_MODEL)
        response = await model.generate_content_async("Say OK")
        return {
            "project": settings.GOOGLE_CLOUD_PROJECT,
            "location": settings.VERTEX_AI_LOCATION,
            "model": settings.GEMINI_MODEL,
            "status": "ok",
            "response": response.text.strip(),
        }
    except Exception as e:
        return {
            "project": settings.GOOGLE_CLOUD_PROJECT,
            "model": settings.GEMINI_MODEL,
            "status": "error",
            "error": str(e),
            "fix": "Run `gcloud auth application-default login` locally, or ensure the Cloud Run service account has roles/aiplatform.user",
        }
