import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_data(file_path):
    """Load the test data from CSV file."""
    try:
        data = pd.read_csv(file_path)
        if data.empty:
            raise ValueError("Test data is empty.")
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y
    except FileNotFoundError:
        raise FileNotFoundError(f"Data file not found: {file_path}")
    except pd.errors.EmptyDataError:
        raise ValueError("The file is empty or corrupted.")
    except Exception as e:
        raise RuntimeError(f"Failed to load test data: {e}")


def load_model(model_path):
    """Load the trained model from a pickle file."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    except pickle.UnpicklingError:
        raise ValueError("Model file is corrupted or incompatible.")
    except Exception as e:
        raise RuntimeError(f"Failed to load model: {e}")


def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return the metrics as a dictionary."""
    try:
        y_pred = model.predict(X_test)
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1-score': f1_score(y_test, y_pred, zero_division=0)
        }
        return metrics
    except Exception as e:
        raise RuntimeError(f"Model evaluation failed: {e}")


def save_metrics(metrics, output_path):
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
    except Exception as e:
        raise RuntimeError(f"Failed to save metrics to {output_path}: {e}")


def main():
    try:
        # Define paths
        data_path = 'data/preprocessed/test_processed.csv'
        model_path = 'model.pkl'
        metrics_path = 'metrics.json'

        # Pipeline steps
        X_test, y_test = load_data(data_path)
        model = load_model(model_path)
        metrics = evaluate_model(model, X_test, y_test)
        save_metrics(metrics, metrics_path)

        print("✅ Evaluation completed successfully.")
        print(json.dumps(metrics, indent=4))

    except Exception as e:
        print(f"❌ Error in evaluation pipeline: {e}")


if __name__ == "__main__":
    main()
