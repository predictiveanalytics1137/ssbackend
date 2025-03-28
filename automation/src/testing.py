import os
import sys

import requests




sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import joblib
import io
from datetime import datetime, timedelta
from src.time_series_pipeline import create_lag_features
from src.feature_engineering import feature_engineering_timeseries
from src.s3_operations import load_from_s3
from src.logging_config import get_logger
from sklearn.preprocessing import StandardScaler  # Assuming this was used
from typing import Optional, List, Tuple
from src.logging_config import get_logger

logger = get_logger(__name__)


# Assuming you have a function to download from S3
# def download_from_s3(bucket_name: str, key: str) -> io.BytesIO:
#     # Placeholder: Replace with your actual S3 download logic
#     import boto3
#     s3 = boto3.client('s3')
#     obj = s3.get_object(Bucket=bucket_name, Key=key)
#     return io.BytesIO(obj['Body'].read())

# def predict_next_30_days(
#     # historical_df: pd.DataFrame,
#     bucket_name: str = "artifacts1137",
#     prefix: str = "ml-artifacts/default/",  # Adjust chat_id as needed
#     today: Optional[str] = None,
#     entity_column: str = "store_id",
#     time_column: str = "analysis_time",
#     target_column: str = "target_within_30_days_after"
# ) -> pd.DataFrame:
#     """
#     Predicts target_within_30_days_after for the next 30 days using saved artifacts.

#     Parameters:
#     - historical_df: DataFrame with historical data (e.g., up to 2013-03-31).
#     - bucket_name: S3 bucket name.
#     - prefix: S3 prefix including chat_id (e.g., "ml-artifacts/123/").
#     - today: Date to treat as "today" (default: current date).
#     - entity_column: Column for store identifier.
#     - time_column: Column for timestamps.
#     - target_column: Target column to predict.

#     Returns:
#     - DataFrame with predictions (store_id, analysis_time, predicted).
#     """
#     try:
#         # Set today's date
#         today = pd.to_datetime(today) if today else pd.to_datetime("2013-04-01")  # Example date
#         logger.info(f"Using today’s date: {today}")

#         # Load artifacts from S3
#         final_model = joblib.load(load_from_s3(bucket_name, f"{prefix}final_model.joblib"))
#         imputers = joblib.load(load_from_s3(bucket_name, f"{prefix}imputers.joblib"))
#         encoders = joblib.load(load_from_s3(bucket_name, f"{prefix}encoders.joblib"))
#         feature_defs = joblib.load(load_from_s3(bucket_name, f"{prefix}feature_defs.joblib"))
#         selected_features = joblib.load(load_from_s3(bucket_name, f"{prefix}selected_features.pkl"))
#         outlier_bounds = joblib.load(load_from_s3(bucket_name, f"{prefix}outlier_bounds.pkl"))
#         saved_column_names = joblib.load(load_from_s3(bucket_name, f"{prefix}saved_column_names.pkl"))
#         historical_df = joblib.load(load_from_s3(bucket_name, f"{prefix}historical_data.joblib"))
#         logger.info("Loaded all artifacts from S3.")

#         # Ensure historical data has datetime
#         historical_df[time_column] = pd.to_datetime(historical_df[time_column])
#         last_date = historical_df[time_column].max()
#         logger.info(f"Last historical date: {last_date}, Historical shape: {historical_df.shape}")

#         # Generate future dates (next 30 days from today)
#         future_dates = pd.date_range(start=today + timedelta(days=1), periods=30, freq='D')
#         unique_stores = historical_df[entity_column].unique()

#         # Create future dataset skeleton
#         future_df = pd.DataFrame({
#             entity_column: np.repeat(unique_stores, len(future_dates)),
#             time_column: future_dates.tolist() * len(unique_stores)
#         })
#         logger.info(f"Future dataset skeleton shape: {future_df.shape}")

#         # Combine historical and future data
#         full_data = pd.concat([historical_df, future_df], ignore_index=True)

#         # Apply imputers (if applicable)
#         for col, imputer in imputers.items():
#             if col in full_data.columns:
#                 full_data[col] = imputer.transform(full_data[[col]])

#         # Apply encoders to categorical columns
#         for col, encoder in encoders.items():
#             if col in full_data.columns and col != target_column:
#                 full_data[col] = full_data[col].apply(
#                     lambda x: encoder.transform([x])[0] if x in encoder.classes_ else -1
#                 )

#         # Feature Engineering
#         logger.info("Performing feature engineering for future predictions...")
#         future_engineered = feature_engineering_timeseries(
#             full_data,
#             target_column=target_column,
#             id_column=entity_column,
#             time_column=time_column,
#             training=False,
#             feature_defs=feature_defs
#         )
#         future_engineered = future_engineered.reset_index(drop=True)

#         # Filter to future dates only
#         future_engineered = future_engineered[future_engineered[time_column] > today]
#         logger.info(f"Future engineered shape: {future_engineered.shape}")

#         # Select features used in training
#         X_future = future_engineered[[col for col in selected_features if col in future_engineered.columns]]

#         # Handle missing columns
#         missing_cols = set(selected_features) - set(X_future.columns)
#         if missing_cols:
#             logger.warning(f"Missing columns in X_future: {missing_cols}")
#             for col in missing_cols:
#                 X_future[col] = 0  # Default to 0 (or use imputers if specified)
#         X_future = X_future[selected_features]  # Ensure order matches training

#         # Scale features (assuming StandardScaler was used)
#         scaler = StandardScaler()  # Load actual scaler if saved separately
#         # Note: In practice, load the trained scaler from S3 if saved
#         X_future_numeric = X_future.select_dtypes(include=["float64", "int64"])
#         X_future_scaled = pd.DataFrame(
#             scaler.fit_transform(X_future_numeric),  # Replace fit_transform with transform if scaler is loaded
#             columns=X_future_numeric.columns,
#             index=X_future_numeric.index
#         )
#         X_future_non_numeric = X_future.select_dtypes(exclude=["float64", "int64"])
#         X_future = pd.concat([X_future_scaled, X_future_non_numeric], axis=1)[selected_features]

#         # Predict
#         future_predictions = final_model.predict(X_future)
#         logger.info(f"Generated {len(future_predictions)} predictions.")

#         # Create predictions dataset
#         predictions_df = pd.DataFrame({
#             entity_column: future_engineered[entity_column].reset_index(drop=True),
#             time_column: future_engineered[time_column].reset_index(drop=True),
#             "predicted": future_predictions
#         })

#         # Clip predictions to avoid negatives (if target is non-negative)
#         predictions_df["predicted"] = np.clip(predictions_df["predicted"], 0, None)
#         logger.info(f"Predictions sample:\n{predictions_df.head()}")

#         return predictions_df

#     except Exception as e:
#         logger.error(f"Prediction failed: {str(e)}", exc_info=True)
#         raise

# # Example usage
# if __name__ == "__main__":
#     # Sample historical data
#     # historical_df = pd.DataFrame({
#     #     "store_id": [8023] * 90,
#     #     "analysis_time": pd.date_range("2013-01-01", "2013-03-31"),
#     #     "target_within_30_days_after": [100.0, 120.0, 110.0] + [150.0] * 87,
#     #     "base_price": [50.0, 52.0, 51.0] + [55.0] * 86 + [56.0],
#     #     "is_featured_sku": [False, False, True] + [False] * 86 + [True],
#     #     "is_display_sku": [False, True, False] + [True] * 86 + [False]
#     # })

#     # Predict
#     chat_id = "TEST130705e-5548-457e-9e0f-b74d8ac3c86d91"  # Replace with actual chat_id
#     prefix = f"ml-artifacts/{chat_id}/"
#     import pdb; pdb.set_trace()
#     predictions = predict_next_30_days(
#         # historical_df,
#         bucket_name="artifacts1137",
#         prefix=prefix,
#         today="2013-04-01"
#     )
#     print(predictions.head())


import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from io import StringIO
import boto3
import pandas as pd
import numpy as np
import joblib
import logging
from src.s3_operations import get_s3_client, load_from_s3, upload_to_s3
from src.data_preprocessing import handle_categorical_features
from src.feature_engineering import feature_engineering_timeseries
from src.helper import normalize_column_names
from src.logging_config import get_logger
import time


def predict_future_timeseries(
    chat_id: str,
    prediction_id: str,
    user_id: str,
    time_column: str,
    entity_column: str,
    target_column: str,
    new_target_column: str
) -> pd.DataFrame:
    try:
        logger.info(f"Starting future predictions for chat_id: {chat_id}, prediction_id: {prediction_id}")
        metadata_api_url = "http://127.0.0.1:8000/api/update_prediction_status/"
        start_time = datetime.now()
        start_timer = time.time()
        bucket_name = "artifacts1137"
        prefix = f"ml-artifacts/{chat_id}/"

        # Load artifacts
        logger.info(f"Loading artifacts from s3://{bucket_name}/{prefix}...")
        model = joblib.load(load_from_s3(bucket_name, f"{prefix}final_model.joblib"))
        imputers = joblib.load(load_from_s3(bucket_name, f"{prefix}imputers.joblib"))
        encoders = joblib.load(load_from_s3(bucket_name, f"{prefix}encoders.joblib"))
        feature_defs = joblib.load(load_from_s3(bucket_name, f"{prefix}feature_defs.joblib"))
        selected_features = joblib.load(load_from_s3(bucket_name, f"{prefix}selected_features.pkl"))
        saved_column_names = joblib.load(load_from_s3(bucket_name, f"{prefix}saved_column_names.pkl"))
        historical_data = joblib.load(load_from_s3(bucket_name, f"{prefix}historical_data.joblib"))
        try:
            scaler = joblib.load(load_from_s3(bucket_name, f"{prefix}scaler.joblib"))
            logger.info("Loaded trained scaler from S3.")
        except:
            scaler = StandardScaler()  # Fallback if not saved yet
            logger.warning("Scaler not found; using default (fix by saving in training).")

        entity_count = len(historical_data[entity_column].unique())
        # response = requests.post(metadata_api_url, json={
        #     'prediction_id': prediction_id,
        #     'chat_id': chat_id,
        #     'user_id': user_id,
        #     'status': 'inprogress',
        #     'entity_count': entity_count,
        #     'start_time': start_time.isoformat(),
        #     "workflow": "prediction",
        # })
        # if response.status_code != 201:
        #     logger.error(f"Failed to create metadata: {response.json()}")
        #     raise RuntimeError("Initial metadata creation failed.")

        # Prepare historical data
        historical_data[time_column] = pd.to_datetime(historical_data[time_column])
        last_date = historical_data[time_column].max()
        logger.info(f"Last historical date: {last_date}, Historical shape: {historical_data.shape}")

        # Generate future dates
        import pdb; pdb.set_trace()
        future_dates = pd.date_range(start=last_date + timedelta(days=7), periods=4, freq='W')
        unique_entities = historical_data[entity_column].unique()
        future_df = pd.DataFrame({
            entity_column: np.repeat(unique_entities, len(future_dates)),
            time_column: future_dates.tolist() * len(unique_entities)
        })

        # Combine and preserve original columns
        full_data = pd.concat([historical_data, future_df], ignore_index=True)
        columns_to_lag = [col for col in historical_data.columns if col not in [time_column, entity_column, new_target_column] and 'lag' not in col and historical_data[col].dtype in ['int64', 'float64']]
        full_data = create_lag_features(full_data, entity_column, columns_to_lag, time_column, lags=3)

        # Apply imputers
        for col, imputer in imputers.items():
            if col in full_data.columns:
                full_data[col] = imputer.transform(full_data[[col]])

        # Apply encoders
        full_encoded, _ = handle_categorical_features(
            df=full_data,
            target_column=new_target_column,
            id_column=entity_column,
            encoders=encoders,
            training=False,
            cardinality_threshold=3,
            saved_column_names=saved_column_names
        )

        # Feature engineering
        logger.info("Performing feature engineering...")
        future_engineered = feature_engineering_timeseries(
            full_encoded,
            target_column=new_target_column,
            id_column=entity_column,
            time_column=time_column,
            training=False,
            feature_defs=feature_defs
        )
        normalize_column_names
        
        future_engineered = future_engineered.reset_index(drop=True)
        future_engineered = future_engineered[future_engineered[time_column] > last_date]
        logger.info(f"Future engineered shape: {future_engineered.shape}, columns: {future_engineered.columns.tolist()}")

        # Select features
        X_predict = future_engineered.reindex(columns=selected_features, fill_value=0)
        logger.info(f"X_predict columns: {X_predict.columns.tolist()}")

        # Scale features
        X_predict_numeric = X_predict.select_dtypes(include=["float64", "int64"])
        X_predict_scaled = pd.DataFrame(
            scaler.transform(X_predict_numeric),
            columns=X_predict_numeric.columns,
            index=X_predict_numeric.index
        )
        X_predict_non_numeric = X_predict.select_dtypes(exclude=["float64", "int64"])
        X_predict = pd.concat([X_predict_scaled, X_predict_non_numeric], axis=1)[selected_features]

        # Predict
        logger.info("Generating predictions...")
        predictions = model.predict(X_predict)

        # Create predictions DataFrame
        predictions_df = pd.DataFrame({
            entity_column: future_engineered[entity_column].reset_index(drop=True),
            time_column: future_engineered[time_column].reset_index(drop=True),
            'predicted': predictions
        })
        predictions_df['predicted'] = np.clip(predictions_df['predicted'], 0, None)
        logger.info(f"Prediction shape: {predictions_df.shape}")
        logger.info(f"Sample predictions:\n{predictions_df.head()}")

        # Save to S3
        prediction_key = f"{prefix}future_predictions_{chat_id}_{prediction_id}.csv"
        with io.StringIO() as buffer:
            predictions_df.to_csv(buffer, index=False)
            buffer.seek(0)
            upload_to_s3(io.BytesIO(buffer.getvalue().encode()), bucket_name, prediction_key)
        logger.info(f"Predictions saved to s3://{bucket_name}/{prediction_key}")

        # Update metadata
        predictions_df[time_column] = predictions_df[time_column].astype(str)
        predictions_json = predictions_df.to_dict(orient='records')
        duration = time.time() - start_timer
        response = requests.patch(f"{metadata_api_url}{prediction_id}/", json={
            'status': 'success',
            'duration': duration,
            'predictions_csv_path': prediction_key,
            'predictions_data': predictions_json
        })
        if response.status_code != 200:
            logger.error(f"Failed to update metadata: {response.json()}")
            raise RuntimeError("Final metadata update failed.")

        return predictions_df

    except Exception as e:
        duration = time.time() - start_timer if 'start_timer' in locals() else 0
        # response = requests.patch(f"{metadata_api_url}{prediction_id}/", json={
        #     'status': 'failed',
        #     'duration': duration,
        #     'predictions_csv_path': None,
        #     'predictions_data': None
        # })
        # if response.status_code != 200:
        #     logger.error(f"Failed to update metadata on failure: {response.json()}")
        logger.error(f"Error during future predictions: {str(e)}", exc_info=True)
        raise




# Example usage
if __name__ == "__main__":
    # input_df = pd.DataFrame({
    #     "store_id": [8023] * 90,
    #     "analysis_time": pd.date_range("2013-01-01", "2013-03-31"),
    #     "target_within_30_days_after": [100.0, 120.0] + [150.0] * 88,
    #     "base_price": [50.0, 52.0] + [55.0] * 87 + [56.0],
    #     "is_featured_sku": [False, False] + [True] * 88,
    #     "is_display_sku": [False, True] + [False] * 88
    # })
    predictions = predict_future_timeseries(
        # input_df=input_df,
        chat_id="ZCXTTC30705e-548-47e-9e0f-b74dac3c86d91",
        prediction_id="pred_123",
        user_id="9",
        time_column="analysis_time",
        entity_column="store_id",
        target_column="total_price",
        new_target_column="target_within_30_days_after"
    )
    print(predictions.head())