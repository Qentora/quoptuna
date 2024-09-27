import threading
from wsgiref.simple_server import make_server

import numpy as np
import pandas as pd
from optuna_dashboard import wsgi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    df = pd.read_csv(file_path)
    y = df["Crystal"]
    X = df.drop(columns=["Crystal"])
    return X, y


def preprocess_data(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    classes = np.unique(y)
    y = np.where(y == classes[0], 1, -1)
    return train_test_split(X, y, random_state=42)


# find a port number that is not in use and returns the port number
def find_free_port():
    import socket

    for port in range(6000, 7000):
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(("localhost", port))
        if result == 0:
            sock.close()
            continue
        else:
            return port
    return None


def start_optuna_dashboard(storage: str, port: int):
    app = wsgi(storage)
    httpd = make_server("localhost", port, app)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    return f"http://localhost:{port}"
