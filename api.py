from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import requests
import os

app = FastAPI()

#Enable CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================================
# MODEL DOWNLOAD (Google Drive)
# ================================

MODEL_PATH = "nids_rf_model.pkl"
FILE_ID = "1HF5gDrxY99h2YLllnDL_I-t16OVluKcH"

def download_from_drive(file_id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params={'id': file_id}, stream=True)

    #Handle large file confirmation
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            params = {'id': file_id, 'confirm': value}
            response = session.get(URL, params=params, stream=True)
            break

    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

#Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model...")
    download_from_drive(FILE_ID, MODEL_PATH)

# ================================
# LOAD MODEL + ASSETS
# ================================

model = joblib.load(MODEL_PATH)
le = joblib.load("nids_label_encoder.pkl")
features = joblib.load("nids_features_list.pkl")

# ================================
#ROUTES
# ================================

@app.get("/")
def home():
    return {"status": "API is live 🚀"}

@app.post("/predict_live")
async def predict(request: Request):
    try:
        data = await request.json()

        #Handle both dict and list input
        if isinstance(data, dict):
            data = [data]

        df = pd.DataFrame(data)

        #Match features expected by model
        X_input = pd.DataFrame()
        for f in features:
            X_input[f] = df.get(f, 0)

        #Clean data
        X_input.replace([np.inf, -np.inf], 0, inplace=True)
        X_input.fillna(0, inplace=True)

        #Predict
        preds = model.predict(X_input)
        decoded = le.inverse_transform(preds)

        return {
            "status": "success",
            "predictions": decoded.tolist()
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }