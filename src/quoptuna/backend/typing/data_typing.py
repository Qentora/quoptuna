from typing import TypedDict

import pandas as pd


class DataVariable(TypedDict):
    """
    DataVariable class for storing the data variables.
    """

    x: pd.DataFrame
    y: pd.Series


DataSet = dict[str, pd.DataFrame | pd.Series]
