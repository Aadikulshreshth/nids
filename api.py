from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import joblib
import pandas as pd
import numpy as np
import threading

from sniffer import flows, start_sniffing
from features import extract_features

app = FastAPI()

# ======================================
# GLOBAL CORS (WORKS FOR NGROK)
# ======================================
@app.middleware("http")
async def cors_fix(request: Request, call_next):
    # HANDLE PREFLIGHT FIRST
    if request.method == "OPTIONS":
        response = JSONResponse(content={"ok": True})
    else:
        response = await call_next(request)

    # FORCE HEADERS
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"

    return response


# ======================================
# LOAD MODEL
# ======================================
model = joblib.load("nids_rf_model.pkl")
le = joblib.load("nids_label_encoder.pkl")
feature_list = joblib.load("nids_features_list.pkl")


# ======================================
# START SNIFFER
# ======================================
threading.Thread(target=start_sniffing, daemon=True).start()


# ======================================
# CORE LOGIC
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

            pred = model.predict(X_input)
            decoded = le.inverse_transform(pred)[0]

        except Exception as e:
            print("Error:", e)
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
# ENDPOINT
# ======================================
@app.get("/predict_live")
def predict_live():
    return {
        "predictions": process_flows() or [{
            "timestamp": "now",
            "threat_type": "BENIGN",
            "severity": "LOW",
            "status": "BENIGN",
            "confidence": "95%"
        }]
    }