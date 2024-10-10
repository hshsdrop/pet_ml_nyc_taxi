from backend.preprocessing import get_data, add_features, clean_data
# from models.train_kmeans import train_kmeans
from models.train_catboost import train_catboost
# import pandas as pd

# df = get_data('train.csv')
# df = clean_data(df)
# kmeans = train_kmeans(df)
# df = add_features(df, path_kmeans='models/kmeans.pkl')
# print(df.shape)
# print(df.head(2))

path = 'https://media.githubusercontent.com/media/hshsdrop/df_for_downloads/refs/heads/main/train.csv'
df = add_features(clean_data(get_data(path)), path_kmeans='models/kmeans.pkl', purpose='tuning')
# Catboost fitting & saving
model = train_catboost(df)
model.save_model('models/catboost.bin', format='cbm')