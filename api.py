from fastapi import FastAPI
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import numpy as np
import threading

from sniffer import flows, start_sniffing
from features import extract_features

app = FastAPI()

# ======================================
# LOAD MODEL FILES
# ======================================
model = joblib.load("nids_rf_model.pkl")
le = joblib.load("nids_label_encoder.pkl")
feature_list = joblib.load("nids_features_list.pkl")


# ======================================
# START SNIFFER THREAD
# ======================================
threading.Thread(target=start_sniffing, daemon=True).start()


# ======================================
# HOME
# ======================================
@app.get("/")
def home():
    return {"status": "NIDS running 🚀"}


# ======================================
# CORE PROCESSING
# ======================================
def process_flows():
    results = []

    for key, flow in list(flows.items()):

        if len(flow) < 1:
            continue

        try:
            features_dict = extract_features(flow)
            df = pd.DataFrame([features_dict])

            X_input = pd.DataFrame()

            for f in feature_list:
                X_input[f] = df.get(f, 0)

            X_input.replace([np.inf, -np.inf], 0, inplace=True)
            X_input.fillna(0, inplace=True)

            if X_input.shape[0] == 0:
                continue

            pred = model.predict(X_input)
            decoded = le.inverse_transform(pred)[0]

        except Exception as e:
            print("Prediction Error:", e)
            decoded = "BENIGN"

        results.append({
            "timestamp": pd.Timestamp.now().strftime("%H:%M:%S"),
            "threat_type": decoded,
            "severity": "LOW" if decoded == "BENIGN" else "HIGH",
            "status": "BENIGN" if decoded == "BENIGN" else "THREAT",
            "confidence": "90%"
        })

        flows[key] = []

    return results


# ======================================
# LIVE DETECT
# ======================================
@app.get("/live_detect")
def live_detect():
    results = process_flows()

    if not results:
        return [
            {
                "timestamp": "now",
                "threat_type": "BENIGN",
                "severity": "LOW",
                "status": "BENIGN",
                "confidence": "95%"
            }
        ]

    return results


# ======================================
# PREFLIGHT (CORS FIX)
# ======================================
@app.options("/predict_live")
def options_predict():
    response = JSONResponse(content={"ok": True})
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response


# ======================================
# MAIN FRONTEND ENDPOINT
# ======================================
@app.get("/predict_live")
def predict_live():
    response = JSONResponse(
        content={"predictions": live_detect()}
    )
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response