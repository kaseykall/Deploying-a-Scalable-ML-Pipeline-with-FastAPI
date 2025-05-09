import pickle
from sklearn.metrics import fbeta_score, precision_score, recall_score
from ml.data import process_data
# TODO: add necessary import
import numpy as np
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.
    """
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.
    """

    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """
    Run model inferences and return the predictions.
    """
    preds = model.predict(X)
    return preds

def save_model(model, path):
    """
    Serializes model to a file.
    """
    with open(path, "wb") as f:
        pickle.dump(model, f)

def load_model(path):
    """
    Loads pickle file from `path` and returns it.
    """
    print(f"Loading model from: {path}")  # Debugging line
    with open(path, "rb") as f:
        return pickle.load(f)


def performance_on_categorical_slice(
    data, column_name, slice_value, categorical_features, label, encoder, lb, model
):
    """
    Computes the model metrics on a slice of the data specified by a column name and
    """
    data_slice = data[data[column_name] == slice_value]

    if data_slice.empty:
        return None, None, None  # Avoid error if slice has no data

    # TODO: implement the function
    X_slice, y_slice, _, _ = process_data(
        data_slice,
        categorical_features=categorical_features,
        label=label,
        training=False,
        encoder=encoder,
        lb=lb,
    )
    preds = inference(model, X_slice)
    precision, recall, fbeta = compute_model_metrics(y_slice, preds)
    return precision, recall, fbeta

def apply_label(prediction, lb):
    """Converts a numeric prediction back into the original string label."""
    print("Prediction (raw):", prediction)
    label = lb.inverse_transform(prediction)
    print("Decoded label:", label)
    return label
    # return lb.inverse_transform(prediction)
