import pandas as pd
from catboost import CatBoostRegressor, Pool
from config.import_config import import_config


def train_catboost(df: pd.DataFrame):
    """Function trains the Catboost model on the input data and returns it.
    :param df: Train dataset
    :return: Catboost trained model
    """
    # Config
    cfg = import_config('config/params.yaml')
    catboost_params = cfg['catboost']
    catboost_params['verbose'] = 400
    catboost_params['iterations'] = catboost_params['iterations']
    catboost_params['learning_rate'] = catboost_params['learning_rate']
    # Data preprocessing
    cat_feature_indices = ['passenger_count', 'pickup_month', 'pickup_weekday',
                           'pickup_hour', 'high_traffic', 'anomaly', 'route']
    X = df.drop('trip_duration', axis=1)
    y = df.trip_duration
    pool_train = Pool(X, y,cat_features=cat_feature_indices)
    # Model training
    model = CatBoostRegressor(**catboost_params)
    model.fit(pool_train)
    model.save_model('catboost.bin', format='cbm')
    print('Model saved at models/catboost.bin')
    return model