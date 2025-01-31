from typing import TypedDict

import pandas as pd


class DataVariable(TypedDict):
    """
    DataVariable class for storing the data variables.
    """

    x_train: pd.DataFrame
    y_train: pd.Series


DataSet = dict[str, pd.DataFrame | pd.Series]
