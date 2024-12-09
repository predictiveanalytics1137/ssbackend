from joblib import load
import pandas as pd
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import predict_new_data

if __name__ == "__main__":
    # new_csv_path = 'data/Test.csv'
    new_csv_path = 'C:/sandy/ssbackend/automation/data/Test.csv'
    predictions = predict_new_data(new_csv_path)
    print("Predictions:")
    print(predictions)
