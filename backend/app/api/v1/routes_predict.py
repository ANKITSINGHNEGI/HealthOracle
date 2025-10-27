from fastapi import APIRouter, Request, HTTPException
from app.models.pydantic_schemas import PredictRequest, PredictResponse
from app.services.inference import predict_diabetes

router = APIRouter(prefix="/v1", tags=["predict"])

@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest, request: Request):
    try:
        risk = predict_diabetes(request.app.state, payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference error: {e}")
    return PredictResponse(
        risk=risk,
        top_features={},
        recommendations=[],
        model_version=request.app.state.model_version
    )
