import joblib
import pandas as pd

MODEL_PATH = "best_model.joblib"
SCALER_PATH = "scaler.joblib"
COLUMNS_PATH = "model_columns.joblib"

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
model_columns = joblib.load(COLUMNS_PATH)

required = ["Gender", "Platform", "Age", "Daily_Usage_Time (minutes)", "Posts_Per_Day", "Likes_Received_Per_Day", "Comments_Received_Per_Day", "Messages_Sent_Per_Day"]

print("Model:", type(model).__name__)
print("Features:", len(model_columns))
print("Scaler features:", scaler.n_features_in_)
print("Model features:", model.n_features_in_)
assert scaler.n_features_in_ == len(model_columns) == model.n_features_in_
print("Artifact compatibility check: PASSED")
