# ml/src/train_diabetes.py
import os, joblib, pandas as pd, numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import os

basepath = os.getcwd()

DATA_PATH = os.path.join(basepath, "ml","data","diabetes.csv")
ARTIFACT_PATH = os.path.join(basepath,"ml","models","diabetes.pkl")
MODEL_VERSION = "v0.1.0"

df = pd.read_csv(DATA_PATH)

# Rename columns if needed to match canonical names
# Expected Pima columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI,
# DiabetesPedigreeFunction, Age, Outcome

# Treat domain-impossible zeros as missing for key vitals and impute
for col in ["Glucose", "BloodPressure", "BMI"]:
    df[col] = df[col].replace(0, np.nan)

# Simple median imputation
for col in ["Glucose", "BloodPressure", "BMI", "SkinThickness", "Insulin"]:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

y = df["Outcome"].astype(int)
X = df.drop(columns=["Outcome"])

# Define numeric / categorical splits (Pima is mostly numeric)
num_cols = [c for c in X.columns if X[c].dtype != "object"]
cat_cols = [c for c in X.columns if X[c].dtype == "object"]

pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

clf = Pipeline([
    ("pre", pre),
    ("lr", LogisticRegression(max_iter=500))
])

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf.fit(X_tr, y_tr)
proba = clf.predict_proba(X_te)[:, 1]
pred = (proba >= 0.5).astype(int)

metrics = {
    "roc_auc": float(roc_auc_score(y_te, proba)),
    "accuracy": float(accuracy_score(y_te, pred)),
    "f1": float(f1_score(y_te, pred))
}
print("Metrics:", metrics)

# Ensure artifact folder exists
os.makedirs(os.path.dirname(ARTIFACT_PATH), exist_ok=True)
joblib.dump({"model": clf, "version": MODEL_VERSION, "features": X.columns.tolist()}, ARTIFACT_PATH)
print(f"Saved model to {ARTIFACT_PATH} (version {MODEL_VERSION})")
