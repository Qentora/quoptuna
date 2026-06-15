"""
Data management endpoints
"""

import json
import os
import uuid
from pathlib import Path

import pandas as pd
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.services import dataset_registry

router = APIRouter()

# Ensure upload directory exists
UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Curated list of popular UCI datasets (numeric ids) suitable for binary
# classification. These ids are passed straight to ``ucimlrepo.fetch_ucirepo``.
POPULAR_UCI_DATASETS = [
    {
        "id": 17,
        "name": "Breast Cancer Wisconsin (Diagnostic)",
        "description": "569 samples, 30 features - diagnostic classification",
        "num_instances": 569,
        "num_features": 30,
    },
    {
        "id": 176,
        "name": "Blood Transfusion Service Center",
        "description": "748 samples, 4 features - donation prediction",
        "num_instances": 748,
        "num_features": 4,
    },
    {
        "id": 267,
        "name": "Banknote Authentication",
        "description": "1372 samples, 4 features - authentic vs forged",
        "num_instances": 1372,
        "num_features": 4,
    },
    {
        "id": 45,
        "name": "Heart Disease",
        "description": "303 samples, 13 features - presence of heart disease",
        "num_instances": 303,
        "num_features": 13,
    },
    {
        "id": 225,
        "name": "ILPD (Indian Liver Patient Dataset)",
        "description": "583 samples, 10 features - liver patient classification",
        "num_instances": 583,
        "num_features": 10,
    },
    {
        "id": 143,
        "name": "Statlog (Australian Credit Approval)",
        "description": "690 samples, 14 features - credit approval",
        "num_instances": 690,
        "num_features": 14,
    },
]

MAX_UNIQUE_FOR_TARGET = 20


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a CSV dataset and register it."""
    filename = file.filename
    if not filename or not filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed")

    file_id = str(uuid.uuid4())
    file_extension = os.path.splitext(filename)[1]
    saved_filename = f"{file_id}{file_extension}"
    file_path = UPLOAD_DIR / saved_filename

    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)

        df = pd.read_csv(file_path)

        dataset_registry.register(
            {
                "id": file_id,
                "name": filename,
                "source": "upload",
                "file_path": str(file_path),
                "rows": len(df),
                "columns": list(df.columns),
            }
        )

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
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=400, detail=f"Failed to process CSV file: {str(e)}")


@router.get("/uci")
async def list_uci_datasets():
    """List curated, popular UCI datasets with numeric ids."""
    return {"datasets": POPULAR_UCI_DATASETS}


@router.post("/uci/{dataset_id}/load")
async def load_uci_dataset(dataset_id: int):
    """Fetch a UCI dataset, persist it as CSV and register it for reuse."""
    try:
        from ucimlrepo import fetch_ucirepo

        dataset = fetch_ucirepo(id=dataset_id)

        if dataset.data.targets is not None:
            df = pd.concat([dataset.data.features, dataset.data.targets], axis=1)
        else:
            df = dataset.data.features

        dataset_name = dataset.metadata.get("name", f"UCI Dataset {dataset_id}")

        registry_id = str(dataset_id)
        file_path = UPLOAD_DIR / f"uci_{dataset_id}.csv"
        df.to_csv(file_path, index=False)

        dataset_registry.register(
            {
                "id": registry_id,
                "name": dataset_name,
                "source": "uci",
                "file_path": str(file_path),
                "rows": len(df),
                "columns": list(df.columns),
            }
        )

        return {
            "message": "Dataset loaded successfully",
            "id": registry_id,
            "dataset_id": dataset_id,
            "name": dataset_name,
            "file_path": str(file_path),
            "rows": len(df),
            "columns": list(df.columns),
        }
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to fetch UCI dataset {dataset_id}: {str(e)}"
        )


@router.get("/uci/{dataset_id}")
async def fetch_uci_dataset(dataset_id: int):
    """Backwards-compatible fetch endpoint (delegates to /load)."""
    return await load_uci_dataset(dataset_id)


@router.get("/{dataset_id}/preview")
async def preview_dataset(dataset_id: str):
    """Return a preview (head, dtypes, missing counts, candidate target values)."""
    record = dataset_registry.get(dataset_id)
    if record is None or not record.get("file_path"):
        raise HTTPException(status_code=404, detail="Dataset not found. Load it first.")

    file_path = record["file_path"]
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Dataset file is missing on the server.")

    df = pd.read_csv(file_path)

    dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
    missing = {col: int(df[col].isna().sum()) for col in df.columns}

    # JSON-safe head rows (NaN -> null).
    head = json.loads(df.head(10).to_json(orient="records"))

    target_values_by_column: dict[str, list] = {}
    for col in df.columns:
        unique = df[col].dropna().unique()
        if len(unique) <= MAX_UNIQUE_FOR_TARGET:
            target_values_by_column[col] = json.loads(pd.Series(unique).to_json(orient="values"))

    return {
        "id": dataset_id,
        "columns": list(df.columns),
        "dtypes": dtypes,
        "head": head,
        "num_rows": len(df),
        "missing": missing,
        "target_values_by_column": target_values_by_column,
    }


@router.get("/{dataset_id}")
async def get_dataset(dataset_id: str):
    """Get dataset information from the registry."""
    record = dataset_registry.get(dataset_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Dataset not found")
    return record
