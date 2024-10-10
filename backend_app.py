import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from backend.preprocessing import add_features
from models import predict


app = FastAPI(title='NYC Manhattan Taxi Trip Duration',
              description='Backend server for Predicting Taxi Trip Duration based on incoming data from Frontend')

class TripConfigure(BaseModel):
    """
    Input Features Validation for the ml model
    """
    pickup_latitude: float
    pickup_longitude: float
    dropoff_latitude: float
    dropoff_longitude: float
    pickup_datetime: str
    passenger_count: int


@app.post('/predict')
async def trip_duration(trip: TripConfigure):
    trip_params = pd.DataFrame({key: [value] for key, value in dict(trip).items()})
    trip_params = add_features(trip_params, path_kmeans='model/kmeans.pkl', purpose='predict')
    prediction = predict(trip_params)
    prediction = {'prediction': prediction}
    return prediction
    


