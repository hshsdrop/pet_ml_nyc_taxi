import pandas as pd
import numpy as np
from catboost import CatBoostRegressor


def predict(df: pd.DataFrame) -> int:
    """Function makes taxi trip predictions in minutes  on the input data.
    :param df: input data
    :return: taxi trip duration
    """
    model = CatBoostRegressor().load_model('backend/models/catboost.bin')
    prediction = (np.exp(model.predict(df)[0]) - 1)//60
    return prediction
