from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import threading

from sniffer import flows, start_sniffing
from features import extract_features

app = FastAPI()

#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load model
model = joblib.load("nids_rf_model.pkl")
le = joblib.load("nids_label_encoder.pkl")
features = joblib.load("nids_features_list.pkl")

#Start packet sniffer in background
threading.Thread(target=start_sniffing, daemon=True).start()

@app.get("/")
def home():
    return {"status": "NIDS running 🚀"}

@app.get("/live_detect")
def live_detect():
    results = []

    for key, flow in list(flows.items()):

        if len(flow) < 5:
            continue

        features_dict = extract_features(flow)

        df = pd.DataFrame([features_dict])

        X_input = pd.DataFrame()
        for f in features:
            X_input[f] = df.get(f, 0)

        X_input.replace([np.inf, -np.inf], 0, inplace=True)
        X_input.fillna(0, inplace=True)

        pred = model.predict(X_input)
        decoded = le.inverse_transform(pred)

        results.append({
            "flow": str(key),
            "prediction": decoded[0]
        })

        #clear processed flow
        flows[key] = []

    return {"results": results}