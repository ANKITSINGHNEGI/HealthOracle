from pydantic import BaseModel
from typing import Dict, List, Optional

class Demographics(BaseModel):
    age: int
    gender: str

class Vitals(BaseModel):
    bmi: float
    systolic_bp: int
    diastolic_bp: int
    cholesterol: float
    glucose: Optional[float] = None

class Lifestyle(BaseModel):
    exercise_hours: float
    diet_score: int
    smoking: bool
    alcohol: str
    sleep_hours: float
    stress_level: str

class MedicalHistory(BaseModel):
    family_diabetes: bool
    hypertension: bool

class PredictRequest(BaseModel):
    demographics: Demographics
    vitals: Vitals
    lifestyle: Lifestyle
    medical_history: MedicalHistory

class PredictResponse(BaseModel):
    risk: Dict[str, float]
    top_features: Dict[str, float]
    recommendations: List[str]
    model_version: str
