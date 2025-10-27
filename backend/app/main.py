from fastapi import FastAPI
from contextlib import asynccontextmanager
import joblib

@asynccontextmanager
async def lifespan(app: FastAPI):
    bundle = joblib.load("ml/models/diabetes.pkl")
    app.state.model = bundle["model"]
    app.state.model_version = bundle.get("version", "v0.1.0")
    app.state.features = bundle.get("features", [])
    yield

app = FastAPI(title="HealthOracle", version="0.1.0", lifespan=lifespan)

from app.api.v1.routes_predict import router as predict_router
from app.api.v1.routes_explain import router as explain_router

app.include_router(predict_router)
app.include_router(explain_router)

@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}
