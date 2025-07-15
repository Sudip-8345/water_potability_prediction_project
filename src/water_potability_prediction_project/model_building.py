import pandas as pd
import numpy as np
import yaml
import pickle
# from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
# import dvclive
import mlflow
import dagshub

dagshub.init(repo_owner='Sudip-8345', repo_name='water_potability_prediction_project', mlflow=True)

def load_params(params_path: str) -> dict:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        return params.get('model_building', {})
    except Exception as e:
        raise Exception(f"Error loading parameters from {params_path}: {e}")


def load_data(filepath: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath)
    except Exception as e:
        raise Exception(f"Error loading data from {filepath}: {e}")


def prepare_data(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    try:
        X_train = data.iloc[:, :-1].values
        y_train = data.iloc[:, -1].values
        return X_train, y_train
    except Exception as e:
        raise Exception(f"Error preparing data: {e}")


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> GradientBoostingClassifier:
    try:
        mlflow.set_tracking_uri("https://dagshub.com/Sudip-8345/water_potability_prediction_project.mlflow")
        mlflow.set_experiment("water_potability_prediction_using_GB")
        with mlflow.start_run():
            n_estimators = params.get('n_estimators', 100)
            max_depth = params.get('max_depth', None)
            clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth)
            clf.fit(X_train, y_train)
            mlflow.log_params({'n_estimators': n_estimators, 'max_depth': max_depth})
            import joblib
            joblib.dump(clf, "model.pkl")
            mlflow.log_artifact("model.pkl")

            return clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")


def save_model(model: GradientBoostingClassifier, file_path: str) -> None:
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        raise Exception(f"Error saving model to {file_path}: {e}")

# def logging_parameters(params):
#     """Log parameters using DVC Live."""
#     try:
#         with dvclive.Live() as live:
#             for key, value in params.items():
#                 live.log_param(key, value)
#             live.next_step()
#     except Exception as e:
#         raise RuntimeError(f"Failed to log parameters: {e}")

def main():
    try:
        params_path = "params.yaml"
        data_path = './data/preprocessed/train_processed.csv'
        model_save_path = 'model.pkl'

        params = load_params(params_path)
        data = load_data(data_path)
        X_train, y_train = prepare_data(data)
        model = train_model(X_train, y_train, params)
        save_model(model, model_save_path)
        # logging_parameters(params)
        print("Model trained and saved successfully.")
    except Exception as e:
        print(f"Error occurred: {e}")


if __name__ == '__main__':
    main()
