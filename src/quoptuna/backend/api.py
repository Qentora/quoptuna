import asyncio
import uuid
from typing import Dict

import pandas as pd
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlmodel import Session, SQLModel, create_engine

from quoptuna.backend.models import DataReference, Task
from quoptuna.backend.optimizer import Optimizer

app = FastAPI()
tasks: Dict[str, dict] = {}
data_storage: Dict[str, pd.DataFrame] = {}

DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(DATABASE_URL)
SQLModel.metadata.create_all(engine)


class OptimizationTask(BaseModel):
    task_id: str
    status: str
    progress: float
    result: dict = None


def objective(trial):
    x = trial.suggest_float("x", -10, 10)
    return (x - 2) ** 2


async def run_optimization(task_id: str, data_id: str):
    optimizer = Optimizer(db_name="test", data_id=data_id)
    tasks[task_id].status = "Running"

    for i in range(100):
        optimizer.optimize(n_trials=1)
        tasks[task_id].progress = (i + 1) / 100
        await asyncio.sleep(0.1)

    tasks[task_id].status = "Completed"
    tasks[task_id].result = {
        "best_params": optimizer.study.best_params,
        "best_value": optimizer.study.best_value,
    }

    with Session(engine) as session:
        task = session.query(Task).filter(Task.task_id == task_id).first()
        task.status = "Completed"
        task.result = str(tasks[task_id].result)
        session.add(task)
        session.commit()


@app.post("/upload")
async def upload_data(file: UploadFile = File(...)):
    data_id = str(uuid.uuid4())
    data = pd.read_csv(file.file)
    data_storage[data_id] = data
    data.to_pickle(f"./data/{data_id}.pkl")

    with Session(engine) as session:
        data_ref = DataReference(data_id=data_id, file_path=f"./data/{data_id}.pkl")
        session.add(data_ref)
        session.commit()

    return {"data_id": data_id, "message": "Data uploaded successfully"}


@app.post("/preprocess/{data_id}")
async def preprocess_data(data_id: str):
    if data_id not in data_storage:
        raise HTTPException(status_code=404, detail="Data not found")
    data = data_storage[data_id]
    # Add your preprocessing logic here
    data_storage[data_id] = data  # Update the preprocessed data
    data.to_pickle(f"./data/{data_id}.pkl")

    with Session(engine) as session:
        data_ref = session.query(DataReference).filter(DataReference.data_id == data_id).first()
        data_ref.file_path = f"./data/{data_id}.pkl"
        session.add(data_ref)
        session.commit()

    return {"data_id": data_id, "message": "Data preprocessed successfully"}


@app.post("/optimize/{data_id}")
async def start_optimization(data_id: str, background_tasks: BackgroundTasks):
    if data_id not in data_storage:
        raise HTTPException(status_code=404, detail="Data not found")
    task_id = str(uuid.uuid4())
    tasks[task_id] = OptimizationTask(task_id=task_id, status="Pending", progress=0.0)
    background_tasks.add_task(run_optimization, task_id, data_id)

    with Session(engine) as session:
        task = Task(task_id=task_id, status="Pending", progress=0.0)
        session.add(task)
        session.commit()

    return {"task_id": task_id, "message": "Optimization task started"}


@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    if task_id not in tasks:
        return {"error": "Task not found"}
    return tasks[task_id]
