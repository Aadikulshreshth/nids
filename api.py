from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import requests
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = "nids_rf_model.pkl"
MODEL_URL = "https://drive.google.com/uc?export=download&id=1HF5gDrxY99h2YLllnDL_I-t16OVluKcH"

# 🔥 Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    r = requests.get(MODEL_URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(r.content)

# Load model
model = joblib.load(MODEL_PATH)
le = joblib.load("nids_label_encoder.pkl")
features = joblib.load("nids_features_list.pkl")

@app.get("/")
def home():
    return {"status": "API is live"}

@app.post("/predict_live")
async def predict(request: Request):
    try:
        data = await request.json()

        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        X_input = pd.DataFrame()
        for f in features:
            X_input[f] = df.get(f, 0)

        X_input.fillna(0, inplace=True)

        preds = model.predict(X_input)
        decoded = le.inverse_transform(preds)

        return {"predictions": decoded.tolist()}

    except Exception as e:
        return {"error": str(e)}