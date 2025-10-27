# backend/app/services/inference.py
from typing import Dict
import pandas as pd

def payload_to_features(payload) -> pd.DataFrame:
    d = payload.model_dump()
    feats = {
        "Pregnancies": 0,
        "Glucose": d["vitals"].get("glucose") or 0,
        "BloodPressure": d["vitals"].get("diastolic_bp") or 0,
        "SkinThickness": 0,
        "Insulin": 0,
        "BMI": d["vitals"].get("bmi") or 0.0,
        "DiabetesPedigreeFunction": 0.5,
        "Age": d["demographics"].get("age") or 0
    }
    return pd.DataFrame([feats])

def predict_diabetes(app_state, payload) -> Dict[str, float]:
    df = payload_to_features(payload)
    if getattr(app_state, "features", None):
        df = df.reindex(columns=app_state.features, fill_value=0)
    proba = float(app_state.model.predict_proba(df)[:, 1])
    return {"diabetes": proba}
