# from loguru import logger
# import logging
# import time
# from datetime import datetime, timedelta


# # @logger.catch
# if __name__ == '__main__':
#     logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
#     path = '..\\train.csv'
#     logging.info('Training a model for taxi route clustering!')
#     # Data: get, clean
#     start_time = time.time()
#     logging.info(f'Data preparation starts. {datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}')
#     df = clean_data(get_data(path))
#     end_time = time.time()
#     execution_time = str(timedelta(seconds=(end_time - start_time))).split('.')[0]
#     logging.info(f'Execution time = {execution_time}')
#     logging.info(f'Dataset is ready for training. {datetime.utcfromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")}') 
#     logging.info('Next Stage')
#     # K-means fitting & saving
#     start_time = time.time()
#     logging.info(f'Fitting K-means. {datetime.utcfromtimestamp(start_time).strftime("%Y-%m-%d %H:%M:%S")}')
#     model = train_kmeans(df)
#     pickle.dump(model, open('models/kmeans.pkl', 'wb'))
#     end_time = time.time()
#     execution_time = str(timedelta(seconds=(end_time - start_time))).split('.')[0]
#     logging.info(f'Execution time = {execution_time}')
#     logging.info(f'K-means Trained & Saved. {datetime.utcfromtimestamp(end_time).strftime("%Y-%m-%d %H:%M:%S")}')

from preprocessing import get_data, add_features, clean_data
from config.import_config import import_config
from models.train_kmeans import train_kmeans
import pandas as pd

df = get_data('train.csv')
df = clean_data(df)

model = train_kmeans(df)
print(df.shape)