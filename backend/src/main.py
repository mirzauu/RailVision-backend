from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.config.logging import setup_logging
from src.config.settings import settings
from src.api.v1.router import api_router

setup_logging()

app = FastAPI(
    title="AI C-Suite Agent SaaS",
    version="1.0.0",
    docs_url="/docs" if settings.is_development else None,  # Disable docs in production
    redoc_url="/redoc" if settings.is_development else None,
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "environment": settings.app_env
    }
