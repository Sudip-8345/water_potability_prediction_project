import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import dvclive
import mlflow
import dagshub
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

dagshub.init(repo_owner='Sudip-8345', repo_name='water_potability_prediction_project', mlflow=True)

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
        # Log confusion matrix and classification report
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred, output_dict=True)
        print("Confusion Matrix:\n", cm)
        print("Classification Report:\n", cr)
        # Save confusion matrix as a heatmap
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=np.unique(y_test), yticklabels=np.unique(y_test))
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        mlflow.log_artifact('confusion_matrix.png')
        
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
# def logging_metrics(metrics):
#     """Log metrics using DVC Live."""
#     try:
#         with dvclive.Live() as live:
#             for key, value in metrics.items():
#                 live.log_metric(key, value)
#             live.next_step()
#     except Exception as e:
#         raise RuntimeError(f"Failed to log metrics: {e}")

def main():
    try:
        # Set MLflow tracking server
        mlflow.set_tracking_uri("https://dagshub.com/Sudip-8345/water_potability_prediction_project.mlflow")
        mlflow.set_experiment("water_potability_evaluation")

        # Define paths
        data_path = 'data/preprocessed/test_processed.csv'
        model_path = 'model.pkl'
        metrics_path = 'metrics.json'

        # Start MLflow run BEFORE evaluation
        with mlflow.start_run():
            # Load data and model
            X_test, y_test = load_data(data_path)
            model = load_model(model_path)

            # Evaluate model and log artifact
            metrics = evaluate_model(model, X_test, y_test)

            # Save metrics to file
            save_metrics(metrics, metrics_path)
            mlflow.log_artifact(metrics_path)  # Optional: log JSON file

            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            mlflow.set_tag("model_version", "v1.0")
            mlflow.log_artifact(__file__)

        print("Evaluation completed successfully.")
        print(json.dumps(metrics, indent=4))

    except Exception as e:
        print(f"Error in evaluation pipeline: {e}")




if __name__ == "__main__":
    main()
