# -*- coding: utf-8 -*-
"""TalentGraph API v1 Package"""

from fastapi import APIRouter
from src.api.v1.assessment import router as assessment_router

api_v1_router = APIRouter(prefix="/v1")
api_v1_router.include_router(assessment_router)

__all__ = ["api_v1_router"]
