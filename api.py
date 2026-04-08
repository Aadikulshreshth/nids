from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

app = FastAPI()

@app.middleware("http")
async def cors_fix(request: Request, call_next):
    if request.method == "OPTIONS":
        response = JSONResponse(content={"ok": True})
    else:
        response = await call_next(request)

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"

    return response


@app.get("/predict_live")
def predict_live():
    return {
        "predictions": [
            {
                "timestamp": "now",
                "threat_type": "BENIGN",
                "severity": "LOW",
                "status": "BENIGN",
                "confidence": "99%"
            }
        ]
    }