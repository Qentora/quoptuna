from typing import TypedDict

import pandas as pd


class DataVariable(TypedDict):
    x_train: pd.DataFrame
    y_train: pd.Series


DataSet = dict[str, pd.DataFrame | pd.Series]
