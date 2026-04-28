"""
PDF Report endpoint — receives full audit result JSON, returns a PDF blob.
"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from typing import Any

from app.services.report_generator import generate_pdf_report

router = APIRouter()


@router.post("/pdf")
async def export_pdf_report(result: dict[str, Any]) -> Response:
    try:
        pdf_bytes = generate_pdf_report(result)
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={"Content-Disposition": 'attachment; filename="fairlens-report.pdf"'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF generation failed: {str(e)}")
