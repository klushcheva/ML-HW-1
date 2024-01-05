from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import uvicorn

app = FastAPI()


class CarData(BaseModel):
    year: int
    selling_price: int
    km_driven: int
    fuel: str
    seller_type: str
    transmission: str
    owner: str
    mileage: str
    engine: str
    max_power: str
    torque: str
    seats: float


class CarDataCollection(BaseModel):
    objects: List[CarData]


model = joblib.load('trained_model.pkl')
pf = joblib.load('pf.pkl')

exp_cols = ['fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual', 'seller_type_Trustmark Dealer',
            'transmission_Manual', 'owner_Fourth & Above Owner', 'owner_Second Owner', 'owner_Test Drive Car',
            'owner_Third Owner']
drop_cols = ['owner_First Owner', 'transmission_Automatic', 'fuel_CNG', 'seller_type_Dealer']


def preprocess(input_json: CarData):
    input_dict = input_json.dict()
    input_dict.pop("selling_price", None)
    input_dict.pop("torque", None)
    input_dict.pop("name", None)
    input_df = pd.DataFrame([input_dict])
    input_code = pd.get_dummies(input_df, columns=['fuel', 'seller_type', 'transmission', 'owner'])

    for col in exp_cols:
        if col not in input_code.columns:
            input_code[col] = 0

    for col in drop_cols:
        if col in input_code.columns:
            input_code = input_code.drop(columns=col, axis=1)

    input_code = input_code.reindex(columns=['year', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
                                             'fuel_Diesel', 'fuel_LPG', 'fuel_Petrol', 'seller_type_Individual',
                                             'seller_type_Trustmark Dealer', 'transmission_Manual',
                                             'owner_Fourth & Above Owner', 'owner_Second Owner',
                                             'owner_Test Drive Car', 'owner_Third Owner'])

    input_transformed = pf.transform(input_code)
    return input_transformed


def preprocess_list(car_data_list):
    processed_data = []
    for input_json in car_data_list:
        processed_data.append(preprocess(input_json))
    return processed_data


car_object = CarData(
    year=2020,
    selling_price=8000,
    km_driven=50000,
    fuel='Petrol',
    seller_type='Dealer',
    transmission='Automatic',
    owner='First',
    mileage='18',
    engine='1498',
    max_power='120',
    torque='115 Nm',
    seats=5.0
)

car_list = [CarData(
    year=2020,
    selling_price=8000,
    km_driven=50000,
    fuel='Petrol',
    seller_type='Dealer',
    transmission='Automatic',
    owner='First Owner',
    mileage='18',
    engine='1498',
    max_power='120',
    torque='115 Nm',
    seats=5.0),

    CarData(
        year=2019,
        selling_price=7000,
        km_driven=12000,
        fuel='LPG',
        seller_type='Dealer',
        transmission='Automatic',
        owner='First Owner',
        mileage='100',
        engine='1498',
        max_power='120',
        torque='115 Nm',
        seats=5.0
    )]


@app.post("/predict_item")
def predict_item(item: CarData):
    data = preprocess(item)
    prediction = model.predict(data)
    return prediction[0]


@app.post("/predict_items")
def predict_items(items: CarDataCollection):
    data = preprocess_list(items)
    predictions = model.predict(data)
    return predictions
