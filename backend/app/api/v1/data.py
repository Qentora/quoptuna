"""
Data management endpoints
"""

import os
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings

router = APIRouter()

# Ensure upload directory exists
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset"""
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    # Generate unique file ID
    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(file.filename)[1]
    saved_filename = f"{file_id}{file_extension}"
    file_path = UPLOAD_DIR / saved_filename

    # Save file
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        # Validate CSV and get basic info
        df = pd.read_csv(file_path)

        return JSONResponse(
            status_code=201,
            content={
                "message": "Dataset uploaded successfully",
                "filename": file.filename,
                "id": file_id,
                "file_path": str(file_path),
                "rows": len(df),
                "columns": list(df.columns),
            },
        )
    except Exception as e:
        # Clean up file if it was created
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")


@router.get("/uci")
async def list_uci_datasets():
    """List available UCI datasets"""
    # TODO: Implement UCI dataset listing
    return {
        "datasets": [
            {
                "id": "iris",
                "name": "Iris Dataset",
                "description": "Classic iris flower dataset",
                "rows": 150,
                "features": 4,
            },
            {
                "id": "wine",
                "name": "Wine Quality",
                "description": "Wine quality dataset",
                "rows": 1599,
                "features": 11,
            },
        ]
    }


@router.get("/uci/{dataset_id}")
async def fetch_uci_dataset(dataset_id: int):
    """Fetch a specific UCI dataset"""
    try:
        from ucimlrepo import fetch_ucirepo

        # Fetch dataset from UCI repository
        dataset = fetch_ucirepo(id=dataset_id)

        # Combine features and targets into single dataframe
        if dataset.data.targets is not None:
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        else:
            df = dataset.data.features

        # Get dataset name from metadata
        dataset_name = dataset.metadata.get("name", f"UCI Dataset {dataset_id}")

        return {
            "message": "Dataset fetched successfully",
            "dataset_id": str(dataset_id),
            "name": dataset_name,
            "rows": len(df),
            "columns": list(df.columns),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to fetch UCI dataset {dataset_id}: {str(e)}"
        )


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset information"""
    # TODO: Implement dataset retrieval
    return {
        "id": dataset_id,
        "name": "Example Dataset",
        "rows": 100,
        "columns": 5,
    }


@router.delete("/{dataset_id}")
async def delete_dataset(dataset_id: str):
    """Delete a dataset"""
    # TODO: Implement dataset deletion
    return {"message": "Dataset deleted successfully"}
