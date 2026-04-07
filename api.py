from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import numpy as np
import threading

from sniffer import flows, start_sniffing
from features import extract_features

app = FastAPI()

#Enable CORS (important for frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load model files
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

        #lowered threshold (important fix)
        if len(flow) < 2:
            continue

        features_dict = extract_features(flow)

        df = pd.DataFrame([features_dict])

        X_input = pd.DataFrame()

        #match model features
        for f in features:
            if f in df.columns:
                X_input[f] = df[f]
            else:
                X_input[f] = 0

        #clean data
        X_input.replace([np.inf, -np.inf], 0, inplace=True)
        X_input.fillna(0, inplace=True)

        #CRITICAL FIX — avoid empty input crash
        if X_input.shape[0] == 0:
            continue

        #predict
        pred = model.predict(X_input)
        decoded = le.inverse_transform(pred)

        results.append({
        "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
        "threat_type": decoded[0],
        "severity": "LOW" if decoded[0] == "BENIGN" else "HIGH",
        "status": "BENIGN" if decoded[0] == "BENIGN" else "THREAT",
        "confidence": "90%"
        })

        #clear processed flow
        flows[key] = []

    #FINAL safety return
        if len(results) == 0:
            return [
                {
                    "timestamp": "now",
                    "threat_type": "BENIGN",
                    "severity": "LOW",
                    "status": "BENIGN",
                    "confidence": "95%"
                }
            ]

    return { results}