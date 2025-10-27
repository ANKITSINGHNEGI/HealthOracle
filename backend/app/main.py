from fastapi import FastAPI
from app.api.v1.routes_predict import router as predict_router

app = FastAPI(title="HealthOracle", version="0.1.0")
app.include_router(predict_router)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
