"""
QuOptuna Next - FastAPI Backend

Modern, high-performance backend for quantum machine learning optimization.
"""

from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.types import Scope

from quoptuna.server.api.v1 import analysis, data, optimize, system
from quoptuna.server.core.config import settings

app = FastAPI(
    title="QuOptuna Next API",
    description="Backend API for quantum machine learning optimization with Optuna",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(data.router, prefix="/api/v1/data", tags=["data"])
app.include_router(optimize.router, prefix="/api/v1/optimize", tags=["optimization"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(system.router, prefix="/api/v1", tags=["system"])


@app.get("/api")
async def api_root():
    """API metadata endpoint."""
    return {
        "message": "QuOptuna Next API",
        "version": "2.0.0",
        "docs": "/api/docs",
    }


@app.get("/api/v1/health")
async def health_check():
    """Health check endpoint"""
    return JSONResponse(
        status_code=200,
        content={
            "status": "healthy",
            "message": "QuOptuna Next API is running",
        },
    )


class SPAStaticFiles(StaticFiles):
    """Serve the bundled single-page app, falling back to index.html on 404."""

    async def get_response(self, path: str, scope: Scope):
        response = await super().get_response(path, scope)
        if response.status_code == 404:
            response = await super().get_response("index.html", scope)
        return response


# The statically-exported frontend is bundled into the wheel at quoptuna/web by
# the build pipeline. When present, serve it at the root so the API and UI share
# a single origin/port. API routes above take precedence (registered first).
_WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if _WEB_DIR.is_dir():
    app.mount("/", SPAStaticFiles(directory=_WEB_DIR, html=True), name="ui")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "quoptuna.server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
