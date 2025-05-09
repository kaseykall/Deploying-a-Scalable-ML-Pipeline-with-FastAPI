import os
import pandas as pd
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ml.data import process_data
from ml.model import apply_label, inference

#  Load model and encoder paths
model_path = "model/model.pkl"
encoder_path = "model/encoder.pkl"
lb_path = "model/lb.pkl"
column_order_path = "model/column_order.pkl"

model = joblib.load(model_path)
encoder = joblib.load(encoder_path)
lb = joblib.load(lb_path)
column_order = joblib.load(column_order_path)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country"
]

# Confirm successful loads
print("Model loaded:", model is not None)
print("Encoder loaded:", encoder is not None)
print("Label binarizer loaded:", lb is not None)
print("Column order loaded:", column_order is not None)

# === FastAPI setup ===
app = FastAPI()

# DO NOT MODIFY
class Data(BaseModel):
    age: int = Field(..., example=37)
    workclass: str = Field(..., example="Private")
    fnlgt: int = Field(..., example=178356)
    education: str = Field(..., example="HS-grad")
    education_num: int = Field(..., example=10, alias="education-num")
    marital_status: str = Field(
        ..., example="Married-civ-spouse", alias="marital-status"
    )
    occupation: str = Field(..., example="Prof-specialty")
    relationship: str = Field(..., example="Husband")
    race: str = Field(..., example="White")
    sex: str = Field(..., example="Male")
    capital_gain: int = Field(..., example=0, alias="capital-gain")
    capital_loss: int = Field(..., example=0, alias="capital-loss")
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., example="United-States", alias="native-country")

# TODO: create a GET on the root giving a welcome message
@app.get("/")
async def get_root():
    return {"message": "Welcome to the ML Inference API"}

# TODO: create a POST on a different path that does model inference
@app.post("/data/")
async def post_inference(data: Data):
    # DO NOT MODIFY: turn the Pydantic model into a dict.
    data_dict = data.dict()
    # DO NOT MODIFY: clean up the dict to turn it into a Pandas DataFrame.
    # The data has names with hyphens and Python does not allow those as variable names.
    # Here it uses the functionality of FastAPI/Pydantic/etc to deal with this.
    data_dict_clean = {k.replace("_", "-"): [v] for k, v in data_dict.items()}
    data_df = pd.DataFrame(data_dict_clean) # Create the DataFrame
    # Preprocess the data just like in training
    X, _, _, _ = process_data(
        data_df,
        categorical_features=cat_features,
        training=False,
        encoder=encoder,
        lb=lb
    )
    # Reorder columns to match training
    X = X[column_order]

    prediction = model.predict(X)
    
    label = apply_label(prediction, lb)
    # Convert from bytes to string if needed
    label_str = label[0].decode() if isinstance(label[0], bytes) else str(label[0])
    return {"prediction": int(prediction[0]), "label": label_str}

    # return {"prediction": int(prediction[0]), "label": label[0]}
