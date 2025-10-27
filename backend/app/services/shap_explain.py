# backend/app/services/shap_explain.py
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import shap

def explain_row(app_state, df_one: pd.DataFrame, top_n: int = 5) -> Dict[str, float]:
    """
    Compute local SHAP attributions for a single-row dataframe.
    Returns a dict of feature -> shap_value (sorted by |value| desc, top_n).
    """
    # Use model's predict_proba so we explain the probability of class 1
    f = lambda X: app_state.model.predict_proba(X)[:, 1]
    # A small background set can be the same row or a zero/mean reference if available
    background = df_one  # for a quick start; replace with a small sample from training
    explainer = shap.Explainer(f, background, feature_names=df_one.columns.tolist())
    sv = explainer(df_one)
    values = sv.values[0] if hasattr(sv, "values") else np.array(sv[0])
    names = df_one.columns.to_list()
    order = np.argsort(-np.abs(values))[:top_n]
    return {names[i]: float(values[i]) for i in order}
