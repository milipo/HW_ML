from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pickle
from typing import List
import pandas as pd
from fastapi.responses import FileResponse
from fastapi import UploadFile
df_train = pd.read_csv('../downloads/HW_ML1/Hometasks/fill_nan.csv')

app = FastAPI()

pkl_filename = "../downloads/HW_ML1/Hometasks/car.pkl"
with open(pkl_filename, 'rb') as file:
    model = pickle.load(file)

scaler_pickle = "../downloads/HW_ML1/Hometasks/scaler.pkl"
with open(scaler_pickle, 'rb') as file:
    scaler = pickle.load(file)

encoder_pickle = "../downloads/HW_ML1/Hometasks/encoder.pkl"
with open(encoder_pickle, 'rb') as file:
    encoder = pickle.load(file)


class Item(BaseModel):
    name: str
    year: int
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


class Items(BaseModel):
    objects: List[Item]

@app.get("/")
def say_hi():
    return {"Message": 'Hi, mom!'}


@app.post("/predict_item")
def predict_item(item: Item) -> float:

    input_frame = pd.DataFrame([item.dict()])


    to_clean = ['mileage', 'engine', 'max_power']
    for i in to_clean:
        input_frame[i] = input_frame[i].str.extract(r'(\d)')
        input_frame[i] = input_frame[i].astype('float64')

    input_frame.drop(labels=['torque', 'name'], axis=1, inplace=True)

    to_clean = ['engine', 'seats']
    for i in to_clean:
        input_frame[i] = input_frame[i].astype('int')

    num_d = input_frame.select_dtypes(exclude=['object'])
    input_frame[num_d.columns] = pd.DataFrame(data=scaler.transform(input_frame[num_d.columns]), columns=num_d.columns)


    cat_cols = (input_frame.select_dtypes(include=['object']).columns.values).copy()
    encoder_df = pd.DataFrame(encoder.transform(input_frame[cat_cols]).toarray())
    input_frame = input_frame.join(encoder_df)
    input_frame.drop(cat_cols, axis=1, inplace=True)

    input_frame.rename(columns={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                                9: '9', 10: '10', 11: '11', 12: '12', 13: '13'}, inplace=True)

    input_frame['year_sq'] = input_frame['year'] * input_frame['year']
    input_frame['power/mileage'] = input_frame['max_power'] / input_frame['mileage']

    pred = model.predict(input_frame.values)[0]
    return pred


@app.post("/upload_predict_items", response_class=FileResponse)
def predict_items(file: UploadFile):
    input_frame  = pd.read_csv(file.file)
    input_frame.drop(labels=['Unnamed: 0'], axis=1, inplace=True)

    to_clean = ['mileage', 'engine', 'max_power']
    for i in to_clean:
        input_frame[i] = input_frame[i].str.extract(r'(\d)')
        input_frame[i] = input_frame[i].astype('float64')

    input_frame.drop(labels=['torque'], axis=1, inplace=True)
    input_frame.fillna(df_train.median(), inplace=True)
    to_clean = ['engine', 'seats']
    for i in to_clean:
        input_frame[i] = input_frame[i].astype('int')

    num_d = input_frame.select_dtypes(exclude=['object'])
    input_frame.drop(labels=['name'], axis=1, inplace=True)
    b = input_frame[num_d.columns]
    input_frame[num_d.columns] = pd.DataFrame(data=scaler.transform(input_frame[num_d.columns]), columns=num_d.columns)
    cat_cols = (input_frame.select_dtypes(include=['object']).columns.values).copy()
    encoder_df = pd.DataFrame(encoder.transform(input_frame[cat_cols]).toarray())
    input_frame = input_frame.join(encoder_df)
    input_frame.drop(cat_cols, axis=1, inplace=True)

    input_frame.rename(columns={0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8',
                                9: '9', 10: '10', 11: '11', 12: '12', 13: '13'}, inplace=True)

    input_frame['year_sq'] = input_frame['year'] * input_frame['year']
    input_frame['power/mileage'] = input_frame['max_power'] / input_frame['mileage']

    input_frame['results'] = model.predict(input_frame)
    input_frame.to_csv('results.csv')
    return 'results.csv'

if __name__ == "__main__":
    uvicorn.run(app)