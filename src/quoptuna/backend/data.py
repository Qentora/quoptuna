import numpy as np
import pandas as pd
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
