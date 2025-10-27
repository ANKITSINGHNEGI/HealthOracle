from fastapi import APIRouter, Request, HTTPException
from app.models.pydantic_schemas import PredictRequest
from app.services.inference import payload_to_features
from app.services.shap_explain import explain_row

router = APIRouter(prefix="/v1", tags=["explain"])

@router.post("/explain")
def explain(payload: PredictRequest, request: Request):
    try:
        df = payload_to_features(payload)
        if getattr(request.app.state, "features", None):
            df = df.reindex(columns=request.app.state.features, fill_value=0)
        contrib = explain_row(request.app.state, df, top_n=5)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Explain error: {e}")
    return {"top_features": contrib, "model_version": request.app.state.model_version}
