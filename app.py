from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI()

STUDENT_NAME = "Nandana"
ROLL_NO = "2022BCS0005"

model = None

def load_model():
    global model
    if os.path.exists("models/model.pkl"):
        model = joblib.load("models/model.pkl")

load_model()

class PredictRequest(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.get("/")
def health():
    return {
        "status": "healthy",
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO
    }

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO
    }

@app.post("/predict")
def predict(request: PredictRequest):
    if model is None:
        return {"error": "Model not loaded", "name": STUDENT_NAME, "roll_no": ROLL_NO}
    
    features = np.array([[
        request.sepal_length,
        request.sepal_width,
        request.petal_length,
        request.petal_width
    ]])
    
    prediction = model.predict(features)[0]
    species = {0: "setosa", 1: "versicolor", 2: "virginica"}
    
    return {
        "prediction": int(prediction),
        "species": species[int(prediction)],
        "name": STUDENT_NAME,
        "roll_no": ROLL_NO
    }