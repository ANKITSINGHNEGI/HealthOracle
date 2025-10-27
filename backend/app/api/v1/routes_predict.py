from fastapi import APIRouter
from app.models.pydantic_schemas import PredictRequest, PredictResponse

router = APIRouter(prefix="/v1", tags=["predict"])

@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    return PredictResponse(
        risk={"diabetes": 0.35, "heart_disease": 0.28, "hypertension": 0.31},
        top_features={"bmi": 0.12, "systolic_bp": 0.09, "cholesterol": 0.07},
        recommendations=[
            "Increase weekly activity toward 150 minutes.",
            "Reduce sodium and saturated fat intake.",
            "Monitor blood pressure and lipids in 3 months."
        ],
        model_version="v0.1.0"
    )
