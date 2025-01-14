from fastapi.testclient import TestClient

from quoptuna.backend.api import app

client = TestClient(app)


def test_upload_data():
    response = client.post(
        "/upload", files={"file": ("test.csv", "feature1,feature2,target\n1,2,0\n3,4,1")}
    )
    assert response.status_code == 200
    assert "data_id" in response.json()
    assert response.json()["message"] == "Data uploaded successfully"


def test_preprocess_data():
    # First, upload data to get a data_id
    upload_response = client.post(
        "/upload", files={"file": ("test.csv", "feature1,feature2,target\n1,2,0\n3,4,1")}
    )
    data_id = upload_response.json()["data_id"]

    # Now, preprocess the data using the data_id
    preprocess_response = client.post(f"/preprocess/{data_id}")
    assert preprocess_response.status_code == 200
    assert preprocess_response.json()["data_id"] == data_id
    assert preprocess_response.json()["message"] == "Data preprocessed successfully"


def test_start_optimization():
    # First, upload data to get a data_id
    upload_response = client.post(
        "/upload", files={"file": ("test.csv", "feature1,feature2,target\n1,2,0\n3,4,1")}
    )
    data_id = upload_response.json()["data_id"]

    # Now, start optimization using the data_id
    optimization_response = client.post(f"/optimize/{data_id}")
    assert optimization_response.status_code == 200
    assert "task_id" in optimization_response.json()
    assert optimization_response.json()["message"] == "Optimization task started"


def test_get_task_status():
    # First, upload data to get a data_id
    upload_response = client.post(
        "/upload", files={"file": ("test.csv", "feature1,feature2,target\n1,2,0\n3,4,1")}
    )
    data_id = upload_response.json()["data_id"]

    # Now, start optimization using the data_id
    optimization_response = client.post(f"/optimize/{data_id}")
    task_id = optimization_response.json()["task_id"]

    # Get the task status
    task_status_response = client.get(f"/task/{task_id}")
    assert task_status_response.status_code == 200
    assert task_status_response.json()["task_id"] == task_id
