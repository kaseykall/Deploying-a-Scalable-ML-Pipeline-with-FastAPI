import os
import pandas as pd
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    performance_on_categorical_slice,
    train_model,
)

# TODO: load the census.csv data
project_path = "/Users/kaseykallevig/Deploying-a-Scalable-ML-Pipeline-with-FastAPI"
data_path = os.path.join(project_path, "data", "census.csv")
model_dir = os.path.join(project_path, "model")
os.makedirs(model_dir, exist_ok=True)

# Load data
print(data_path)
data = pd.read_csv(data_path)

# TODO: split the provided data to have a train dataset and a test dataset
train, test = train_test_split(data, test_size=0.2, random_state=42)  # 80% train, 20% test

# DO NOT MODIFY
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

# TODO: use the process_data function provided to process the data.
X_train, y_train, encoder, lb = process_data(
    train,
    categorical_features=cat_features,
    label="salary",
    training=True
)
print("Shape of X_train:", X_train.shape)

# Process test data
X_test, y_test, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False, # Indicating that we are not training (using processed data)
    encoder=encoder, # Using the encoder from the training phase
    lb=lb, # Using the label binarizer from the training phase
)

# TODO: use the train_model function to train the model on the training dataset
model = train_model(X_train, y_train)

# Save artifacts using joblib
joblib.dump(model, os.path.join(model_dir, "model.pkl"))
joblib.dump(encoder, os.path.join(model_dir, "encoder.joblib"))
joblib.dump(lb, os.path.join(model_dir, "lb.joblib"))
joblib.dump(X_train.columns.tolist(), os.path.join(model_dir, "column_order.pkl"))

# Load the model (to validate saving worked)
model = joblib.load(os.path.join(model_dir, "model.pkl"))

# Inference
preds = inference(model, X_test)

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# TODO: compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
for col in cat_features:
    # Iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
             test, col, slicevalue, cat_features, label="salary", encoder=encoder, lb=lb, model=model
        )
        with open("slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)
