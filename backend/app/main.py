"""
QuOptuna Next - FastAPI Backend

Modern, high-performance backend for quantum machine learning optimization.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1 import analysis, data, optimize, system, workflows
from app.core.config import settings

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
app.include_router(workflows.router, prefix="/api/v1/workflows", tags=["workflows"])
app.include_router(optimize.router, prefix="/api/v1/optimize", tags=["optimization"])
app.include_router(analysis.router, prefix="/api/v1/analysis", tags=["analysis"])
app.include_router(system.router, prefix="/api/v1", tags=["system"])


@app.get("/")
async def root():
    """Root endpoint"""
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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
