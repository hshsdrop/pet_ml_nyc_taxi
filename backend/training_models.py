from backend.preprocessing import get_data, add_features, clean_data
from backend.models.train_kmeans import train_kmeans
from backend.models.train_catboost import train_catboost

path = 'https://media.githubusercontent.com/media/hshsdrop/df_for_downloads/refs/heads/main/train.csv'

df = clean_data(get_data(path))
# saving params for kmeans
train_kmeans(df)
df = add_features(clean_data(get_data(path)), path_kmeans='backend/models/kmeans.pkl', purpose='tuning')
# Catboost fitting & saving
model = train_catboost(df)
model.save_model('models/catboost.bin', format='cbm')