import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

df = pd.DataFrame([
    {"age":45,"bmi":29,"glucose":115,"cholesterol":230,"systolic_bp":140,"diastolic_bp":90,"gender":"male","smoking":True,"alcohol":"occasional","diet_type":"mixed","stress_level":"medium","target":1},
    {"age":30,"bmi":22,"glucose":90,"cholesterol":180,"systolic_bp":118,"diastolic_bp":76,"gender":"female","smoking":False,"alcohol":"none","diet_type":"mixed","stress_level":"low","target":0}
])
y = df.pop("target")
num = ["age","bmi","glucose","cholesterol","systolic_bp","diastolic_bp"]
cat = ["gender","smoking","alcohol","diet_type","stress_level"]
pre = ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(handle_unknown="ignore"), cat)])
pipe = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))])
X_tr, X_te, y_tr, y_te = train_test_split(df, y, test_size=0.5, random_state=42)
pipe.fit(X_tr, y_tr)
print("AUC:", roc_auc_score(y_te, pipe.predict_proba(X_te)[:,1]))
joblib.dump(pipe, "ml/models/diabetes.pkl")
