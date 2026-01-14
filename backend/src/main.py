from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from src.config.logging import setup_logging
from src.config.settings import settings
from src.api.v1.router import api_router
from src.infrastructure.graph.indexes import create_indexes
from fastapi.staticfiles import StaticFiles
import os

setup_logging()

app = FastAPI(
    title="AI C-Suite Agent SaaS",
    version="1.0.0",
    docs_url="/docs" if settings.is_development else None,  # Disable docs in production
    redoc_url="/redoc" if settings.is_development else None,
)

# Mount storage for static file access
os.makedirs("storage", exist_ok=True)
app.mount("/storage", StaticFiles(directory="storage"), name="storage")

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

@app.on_event("startup")
def _startup_graph_indexes() -> None:
    if settings.neo4j_uri and settings.neo4j_username and settings.neo4j_password:
        try:
            create_indexes()
        except Exception:
            pass

# Serve React Frontend
@app.get("/")
async def serve_root():
    """Serve the index.html for the root path."""
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "message": "Backend running. Frontend not found in /static."}

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str):
    """Serve static files or fallback to index.html for SPA routing."""
    # Skip if path starts with api/ to avoid masking 404s for API
    if full_path.startswith("api/"):
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Not Found")

    # Check for file in static directory
    file_path = os.path.join("static", full_path)
    if os.path.isfile(file_path):
        return FileResponse(file_path)
    
    # If it looks like a file (has extension) but doesn't exist, return 404
    # This prevents serving index.html for missing images/js
    _, ext = os.path.splitext(full_path)
    if ext:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="File not found")

    # Fallback to index.html for SPA routing
    index_path = os.path.join("static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
        
    from fastapi import HTTPException
    raise HTTPException(status_code=404, detail="Frontend not found")
