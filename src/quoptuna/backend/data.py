import socket
import threading
from wsgiref.simple_server import make_server

import numpy as np
import pandas as pd
from optuna_dashboard import wsgi
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_data(file_path):
    data_frame = pd.read_csv(file_path)
    y = data_frame["Crystal"]
    x = data_frame.drop(columns=["Crystal"])
    return x, y


def preprocess_data(x, y):
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    classes = np.unique(y)
    y = np.where(y == classes[0], 1, -1)
    return train_test_split(x, y, random_state=42)


# find a port number that is not in use and returns the port number
def find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    for port in range(6000, 7000):
        result = sock.connect_ex(("localhost", port))
        if result != 0:
            sock.close()
            return port

    sock.close()
    return None


def start_optuna_dashboard(storage: str, port: int):
    app = wsgi(storage)
    httpd = make_server("localhost", port, app)
    thread = threading.Thread(target=httpd.serve_forever)
    thread.start()
    return f"http://localhost:{port}"
