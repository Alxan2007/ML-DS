from fastapi import FastAPI
import joblib
import pandas as pd

# Загрузка моделей
model = joblib.load("xgb_model.pkl")
regressor = joblib.load("rf_model.pkl")
imputer = joblib.load("imputer.pkl")
scaler = joblib.load("scaler.pkl")
X_columns = joblib.load("X_columns.pkl")

app = FastAPI()

@app.get("/")
def health_check():
    return {"status": "OK", "model": "XGBoost + RandomForest", "ready": True}

@app.post("/predict")
def predict(features: dict):
    input_df = pd.DataFrame([features])
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=X_columns, fill_value=0)
    input_df = imputer.transform(input_df)
    input_scaled = scaler.transform(input_df)
    pred_xgb = model.predict(input_scaled)[0]
    pred_rf = regressor.predict(input_df)[0]
    ensemble = (pred_xgb + pred_rf) / 2
    return {
        "prediction_xgb": round(pred_xgb, 2),
        "prediction_rf": round(pred_rf, 2),
        "average_ensemble": round(ensemble, 2)
    }

print(app)