from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np

app = FastAPI()

# ✅ CORS (fixes your error)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model files
model = joblib.load("nids_rf_model.pkl")
le = joblib.load("nids_label_encoder.pkl")
features = joblib.load("nids_features_list.pkl")

@app.post("/predict_live")
async def predict(request: Request):
    data = await request.json()

    df = pd.DataFrame(data)

    X_input = pd.DataFrame()
    for feature in features:
        if feature in df.columns:
            X_input[feature] = df[feature]
        else:
            X_input[feature] = 0

    X_input.replace([np.inf, -np.inf], 0, inplace=True)
    X_input.fillna(0, inplace=True)

    preds = model.predict(X_input)
    decoded = le.inverse_transform(preds)

    return {"predictions": decoded.tolist()}