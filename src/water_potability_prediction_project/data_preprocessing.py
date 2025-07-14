import os
import pandas as pd
import numpy as np


def load_csv(file_path):
    """Load a CSV file into a pandas DataFrame."""
    try:
        df = pd.read_csv(file_path)
        if df.empty:
            raise ValueError(f"{file_path} is empty.")
        return df
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    except pd.errors.ParserError:
        raise ValueError(f"Failed to parse CSV file: {file_path}")
    except Exception as e:
        raise RuntimeError(f"Error loading {file_path}: {e}")


def fill_missing_values_median(df):
    """Fill missing values with median for each column."""
    try:
        for column in df.columns:
            if df[column].isna().any():
                mean = df[column].median()
                df[column] = df[column].fillna(mean)

        return df
    except Exception as e:
        raise RuntimeError(f"Error while filling missing values: {e}")


def save_csv(df, file_path):
    """Save a DataFrame to a CSV file."""
    try:
        df.to_csv(file_path, index=False)
    except Exception as e:
        raise RuntimeError(f"Failed to save file: {file_path}. Error: {e}")


def ensure_directory(path):
    """Ensure that a directory exists."""
    try:
        os.makedirs(path, exist_ok=True)
    except Exception as e:
        raise RuntimeError(f"Could not create directory {path}: {e}")


def main():
    try:
        # Paths
        raw_data_path = os.path.join('data', 'raw')
        preprocessed_data_path = os.path.join('data', 'preprocessed')
        train_file = os.path.join(raw_data_path, 'train.csv')
        test_file = os.path.join(raw_data_path, 'test.csv')

        # Load data
        train_data = load_csv(train_file)
        test_data = load_csv(test_file)

        # Process missing values
        train_processed = fill_missing_values_median(train_data)
        test_processed = fill_missing_values_median(test_data)

        # Ensure output directory
        ensure_directory(preprocessed_data_path)

        # Save processed data
        save_csv(train_processed, os.path.join(preprocessed_data_path, 'train_processed.csv'))
        save_csv(test_processed, os.path.join(preprocessed_data_path, 'test_processed.csv'))

        print("Data preprocessing completed successfully.")

    except Exception as e:
        print(f"Error in preprocessing pipeline: {e}")


if __name__ == "__main__":
    main()
