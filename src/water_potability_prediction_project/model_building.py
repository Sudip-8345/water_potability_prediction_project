import pandas as pd
import numpy as np
import yaml
import pickle
from sklearn.ensemble import RandomForestClassifier


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


def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    try:
        n_estimators = params.get('n_estimators', 100)
        max_depth = params.get('max_depth', None)
        clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        raise Exception(f"Error training model: {e}")


def save_model(model: RandomForestClassifier, file_path: str) -> None:
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(model, f)
    except Exception as e:
        raise Exception(f"Error saving model to {file_path}: {e}")


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

        print("✅ Model trained and saved successfully.")
    except Exception as e:
        print(f"❌ Error occurred: {e}")


if __name__ == '__main__':
    main()
