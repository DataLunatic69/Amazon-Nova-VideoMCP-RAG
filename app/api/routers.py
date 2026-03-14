"""
API router aggregation.

All v1 endpoint routers are collected here and mounted under the /api/v1
prefix so the FastAPI app factory (api/app.py) has a single include call.
"""

from fastapi import APIRouter

from api.endpoints.v1 import chat, video

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(chat.router)
api_router.include_router(video.router)

__all__ = ["api_router"]
