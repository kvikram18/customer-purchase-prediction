"""Make a purchase prediction for one example customer."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
model = joblib.load(ROOT / "models" / "best_model.joblib")

customer = pd.DataFrame([{
    "age": 29,
    "annual_income": 72000,
    "website_time_minutes": 12.5,
    "previous_purchases": 3,
    "website_visits": 6,
    "discount_received": 1
}])

prediction = int(model.predict(customer)[0])
probability = float(model.predict_proba(customer)[0, 1])
print("Prediction:", "Likely to purchase" if prediction else "Not likely to purchase")
print(f"Estimated purchase probability: {probability:.1%}")
