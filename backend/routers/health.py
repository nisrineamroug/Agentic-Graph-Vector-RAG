# -*- coding: utf-8 -*-
"""
Health-check endpoint.
"""

from fastapi import APIRouter
from backend.config import get_settings
from backend.models.schemas import HealthResponse

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    settings = get_settings()
    return HealthResponse(status="ok", version=settings.APP_VERSION)
