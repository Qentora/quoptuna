import socket
import threading
from pathlib import Path
from wsgiref.simple_server import make_server

import numpy as np
import pandas as pd
from optuna_dashboard import wsgi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

"""
Data utils for loading and preprocessing the data.
"""


def load_data(file_path):
    """
    Load the data from the file path.
    """
    data_frame = pd.read_csv(file_path)
    y = data_frame["Crystal"]
    x = data_frame.drop(columns=["Crystal"])
    return x, y


def preprocess_data(x, y):
    """
    Preprocess the data.
    """
    from quoptuna.backend.task_type import TaskSpec

    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    # Encode via TaskSpec so the label convention matches the rest of the
    # system: binary class_labels[0] -> -1 / class_labels[1] -> +1, multiclass
    # -> integer codes 0..K-1 (raw string labels included). Already-encoded
    # targets ({-1,+1} binary or 0..K-1 codes) pass through unchanged.
    classes = np.unique(y)
    already_pm = set(np.asarray(classes).tolist()) <= {-1, 1}
    already_codes = len(classes) > 2 and np.array_equal(  # noqa: PLR2004
        np.sort(np.asarray(classes)), np.arange(len(classes))
    )
    if not (already_pm or already_codes):
        y = TaskSpec.from_target(y).encode(y)
    return train_test_split(x, y, random_state=42)


def find_free_port():
    """
    Find a port number that is not in use and returns the port number.
    """
    for port in range(6000, 7000):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", port))
        if result == 0:
            sock.close()
            continue
        return port
    return None


def start_optuna_dashboard(storage: str, port: int):
    app = wsgi(storage)
    httpd = make_server("localhost", port, app)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    return f"http://localhost:{port}"


def mock_csv_data(data, tmp_path, file_name=None):
    dataframe = pd.DataFrame(data)
    file_name = file_name or "mock_csv"
    file_path = Path(tmp_path) / f"{file_name}.csv"
    dataframe.to_csv(file_path, index=False)
    return file_path
