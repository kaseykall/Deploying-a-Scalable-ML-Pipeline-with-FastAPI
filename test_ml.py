import pytest
# TODO: add necessary imports
import pandas as pd
import numpy as np
from ml.model import train_model, inference, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from ml.data import process_data
from sklearn.model_selection import train_test_split

# 1 TODO: implement the first test. Change the function name and input as needed
def test_train_model():
    """
    Assuming 'train_data' is a valid input to 'train_model'
    """
    train_data = pd.read_csv("data/census.csv")  # or any other data file
    
    # Assume `process_data` works and provides the correct input for training
    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    X_train, y_train, encoder, lb = process_data(
        train_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    # Call train_model and check the type of model returned
    model = train_model(X_train, y_train)
    assert isinstance(model, RandomForestClassifier), f"Expected RandomForestClassifier, got {type(model)}"


# 2 TODO: implement the second test. Change the function name and input as needed
def test_inference():
    """
    Mock some test data or use the processed data
    """
    test_data = pd.read_csv("data/census.csv")

    cat_features = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
    X_train, y_train, encoder, lb = process_data(
        test_data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )
    
    model = train_model(X_train, y_train)
    
    # Assuming also processed the test data for inference
    X_test, y_test, _, _ = process_data(test_data, categorical_features=cat_features, label="salary", training=False, encoder=encoder, lb=lb)
    
    # Call inference and check if predictions are returned as an array
    predictions = inference(model, X_test)
    assert isinstance(predictions, (np.ndarray)), f"Expected np.ndarray, got {type(predictions)}"


# 3 TODO: implement the third test. Change the function name and input as needed
def test_compute_model_metrics():
    """
    Generate some mock y_test and predictions
    """
    y_test = [0, 0, 1, 1, 0, 1, 0, 1] # Example binary labels
    preds = [0, 1, 1, 1, 0, 0, 0, 1] # Predicted labels 
    
    # Call compute_model_metrics
    precision, recall, f1 = compute_model_metrics(y_test, preds)
    
    # Check the precision, recall, and F1 score are within the expected range
    assert isinstance(precision, float), f"Expected float, got {type(precision)}"
    assert isinstance(recall, float), f"Expected float, got {type(recall)}"
    assert isinstance(f1, float), f"Expected float, got {type(f1)}"
    
    # Optionally check that the F1 score is reasonable 
    assert 0 <= precision <= 1, f"Precision out of bounds: {precision}"
    assert 0 <= recall <= 1, f"Recall out of bounds: {recall}"
    assert 0 <= f1 <= 1, f"F1 out of bounds: {f1}"
