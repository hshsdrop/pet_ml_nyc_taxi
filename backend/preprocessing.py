import pickle
import pandas as pd
import numpy as np


def add_features(data: pd.DataFrame, path_kmeans: str, purpose: str = 'predict') -> pd.DataFrame:
    """Data proccessing for tuning model or prediction.
    :param data: df 
    :param purpose: df purpose, train model or predict
    :return df: df with generated features
    """
    df = data.copy()
    # time col to datetime
    df['pickup_datetime'] = pd.to_datetime(df.pickup_datetime)
    # month, weekday, hour of pickup date
    df['pickup_month'] = df['pickup_datetime'].dt.month
    df['pickup_weekday'] = df['pickup_datetime'].dt.weekday
    df['pickup_hour'] = df['pickup_datetime'].dt.hour
    df['high_traffic'] = df['pickup_hour'].apply(lambda x: 1 if  8 <= x <= 20 else 0)
    # anomaly day
    anomaly_days = [pd.to_datetime('2016-01-23'), pd.to_datetime('2016-01-24')]
    df['anomaly'] = df.pickup_datetime.between(anomaly_days[0], anomaly_days[1], inclusive='both').astype(int)
    df.drop('pickup_datetime', axis=1, inplace=True)
    # estimation of the trip distance
    meas_ang = 0.506 # 29 angle degree = 0.506 radian
    df['diff_latitude'] = (df['dropoff_latitude'] - df['pickup_latitude']).abs()*111
    df['diff_longitude'] = (df['dropoff_longitude'] - df['pickup_longitude']).abs()*80
    df['Euclidean'] = (df.diff_latitude**2 + df.diff_longitude**2)**0.5 
    df['delta_manh_long'] = (df.Euclidean*np.sin(np.arctan(df.diff_longitude / df.diff_latitude)-meas_ang)).abs()
    df['delta_manh_lat'] = (df.Euclidean*np.cos(np.arctan(df.diff_longitude / df.diff_latitude)-meas_ang)).abs()
    df['manh_length'] = df.delta_manh_long + df.delta_manh_lat
    df.drop(['diff_latitude', 'diff_longitude', 'Euclidean', 'delta_manh_long', 'delta_manh_lat'], axis=1, inplace=True)
    # n_passengers
    df['passenger_count'] = df['passenger_count'].apply(lambda x: 1 if (x < 5) else 2)
    # route with kmeans
    kmeans = pickle.load(open(path_kmeans, 'rb'))
    df['dropoff_cluster'] = kmeans.predict(df[['dropoff_longitude', 'dropoff_latitude']].rename(columns={'dropoff_longitude': 'longitude', 'dropoff_latitude': 'latitude'}))
    df['pickup_cluster'] = kmeans.predict(df[['pickup_longitude', 'pickup_latitude']].rename(columns={'pickup_longitude': 'longitude', 'pickup_latitude': 'latitude'}))
    df['route'] = (df.dropoff_cluster.astype(str) + ' ' + df.pickup_cluster.astype(str)).str.split().apply(sorted).apply(lambda x: x[0] + '_' + x[1])


    # list of needed cols
    columns = ['passenger_count', 'pickup_longitude', 'pickup_latitude',
               'dropoff_longitude', 'dropoff_latitude', 'pickup_month',
               'pickup_weekday', 'pickup_hour', 'high_traffic', 'anomaly', 'manh_length', 'route']

    if purpose == 'tuning':
        df.trip_duration = np.log1p(df.trip_duration)
        columns.append('trip_duration')

    df = df[columns]
    return df



def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """Function cleans data from outliers and irrelevant trips.
    :param df: raw dataframe.
    :return df: cleared df.
    """
    def find_line(A, B):
        y_1, x_1 = A
        y_2, x_2 = B
        k = (y_2 - y_1)/(x_2 - x_1)
        b = y_1 - k*x_1
        return k, b

    df = data.copy()
    mask_1 = df['vendor_id'] == 1
    mask_2 = df['store_and_fwd_flag'] == 'N'
    mask_3 = df['passenger_count'] != 0
    mask = mask_1 & mask_2 & mask_3
    df = df[mask].reset_index(drop=True)
    df = df.drop(['vendor_id', 'store_and_fwd_flag'], axis=1)

    # 13 Bridges in Manhattan
    coords = {}
    coords['middle_henry_bridge'] = (40.877758, -73.922214)
    coords['middle_broadway_bridge'] = (40.873689, -73.910933)
    coords['middle_washington_bridge'] = (40.851470, -73.952140)
    coords['middle_linkoln_tunnel'] = (40.762886, -74.009588)
    coords['middle_holland_tunnel'] = (40.727318, -74.021006)
    coords['middle_brooklyn_bridge'] = (40.704914, -73.995546)
    coords['middle_williamsburg_bridge'] = (40.713333, -73.971641)
    coords['middle_midtown_tunnel'] = (40.744429, -73.963135)
    coords['middle_quins_bridge'] = (40.755947, -73.952374)
    coords['middle_kennedy_bridge'] = (40.798973, -73.919060)
    coords['middle_avenue_bridge'] = (40.807628, -73.932374)
    coords['middle_river_bridge'] = (40.834455, -73.934541)
    coords['middle_heights_bridge'] = (40.862674, -73.914647)

    # Building Borders (Manhattan)
    borders_coefs = {}
    borders_list = [('middle_henry_bridge', 'middle_broadway_bridge'), ('middle_washington_bridge', 'middle_linkoln_tunnel'),
                    ('middle_linkoln_tunnel', 'middle_holland_tunnel'), ('middle_brooklyn_bridge', 'middle_williamsburg_bridge'),
                    ('middle_williamsburg_bridge', 'middle_midtown_tunnel'), ('middle_midtown_tunnel', 'middle_quins_bridge'),
                    ('middle_kennedy_bridge', 'middle_avenue_bridge'), ('middle_avenue_bridge', 'middle_river_bridge'),
                    ('middle_river_bridge', 'middle_heights_bridge')]
    for border in borders_list:
        a, b = border
        borders_coefs[f'{a.split("_")[1]}_{b.split("_")[1]}_path'] = find_line(coords[a], coords[b])

    for border_path, border_coef in borders_coefs.items():
        k, b = border_coef
        df[f'{border_path}_drop'] = k*df.dropoff_longitude + b
        df[f'{border_path}_pick'] = k*df.pickup_longitude + b

    # Filtering Irrelevant Trips
    mask_0_1 = df.henry_broadway_path_pick >= df.pickup_latitude
    mask_0_2 = df.henry_broadway_path_drop >= df.dropoff_latitude
    mask_0 = mask_0_1 & mask_0_2

    mask_1_1 = df.washington_linkoln_path_pick >= df.pickup_latitude
    mask_1_2 = df.washington_linkoln_path_drop >= df.dropoff_latitude
    mask_1 = mask_1_1 & mask_1_2

    mask_2_1 = df.linkoln_holland_path_pick >= df.pickup_latitude
    mask_2_2 = df.linkoln_holland_path_drop >= df.dropoff_latitude
    mask_2 = mask_2_1 & mask_2_2

    mask_3_1 = df.brooklyn_williamsburg_path_pick <= df.pickup_latitude
    mask_3_2 = df.brooklyn_williamsburg_path_drop <= df.dropoff_latitude
    mask_3 = mask_3_1 & mask_3_2

    mask_4_1 = df.williamsburg_midtown_path_pick <= df.pickup_latitude
    mask_4_2 = df.williamsburg_midtown_path_drop <= df.dropoff_latitude
    mask_4 = mask_4_1 & mask_4_2

    mask_5_1 = df.midtown_quins_path_pick <= df.pickup_latitude
    mask_5_2 = df.midtown_quins_path_drop <= df.dropoff_latitude
    mask_5 = mask_5_1 & mask_5_2

    mask_6_1 = df.kennedy_avenue_path_pick >= df.pickup_latitude
    mask_6_2 = df.kennedy_avenue_path_drop >= df.dropoff_latitude
    mask_6 = mask_6_1 & mask_6_2

    mask_7_1 = df.avenue_river_path_pick >= df.pickup_latitude
    mask_7_2 = df.avenue_river_path_drop >= df.dropoff_latitude
    mask_7 = mask_7_1 & mask_7_2

    mask_8_1 = df.river_heights_path_pick <= df.pickup_latitude
    mask_8_2 = df.river_heights_path_drop <= df.dropoff_latitude
    mask_8 = mask_8_1 & mask_8_2

    mask = mask_0 & mask_1 & mask_2 & mask_3 & (mask_4 | mask_5) & ((mask_6 | mask_7 | mask_8))
    df = df[mask].reset_index(drop=True)
    cols = ['pickup_longitude', 'pickup_latitude', 'dropoff_longitude', 'dropoff_latitude',
            'trip_duration', 'pickup_datetime', 'passenger_count',]
    return df[cols]


def get_data(path: str) -> pd.DataFrame:
    """Function loads Data from specific path.
    :param path: data path.
    :return: DataFrame.
    """
    data = pd.read_csv(path)
    return data