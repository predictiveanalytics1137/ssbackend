

import datetime
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import re
import time
import io
import joblib
import logging
import requests
import featuretools as ft
import io
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.outlier_handling import apply_outlier_bounds, detect_and_handle_outliers_train
from src.s3_operations import get_s3_client, load_from_s3, upload_to_s3
from src.data_preprocessing import handle_categorical_features
from src.finalization import finalize_and_evaluate_model, finalize_and_evaluate_model_timeseries
from src.model_selection import train_test_model_selection, train_test_model_selection_timeseries
from src.hyperparameter_tuning import hyperparameter_tuning
from src.utils import automatic_imputation
from src.feature_engineering import feature_engineering, feature_engineering_timeseries
from src.feature_selection import feature_selection
from src.helper import normalize_column_names

# Setup logging
logger = logging.getLogger(__name__)

# =============================================================================
# # -------------------------------
# # Time Series Utilities
# # -------------------------------
# def parse_horizon(horizon_str):
#     match = re.match(r'(\d+)\s*(day|days|month|months)', horizon_str, re.IGNORECASE)
#     if match:
#         value = int(match.group(1))
#         unit = match.group(2).lower()
#         if 'day' in unit:
#             return timedelta(days=value)
#         elif 'month' in unit:
#             return relativedelta(months=value)
#     raise ValueError("Horizon must be like '30 days' or '6 months'")
# 
# def next_aligned(date, frequency):
#     if frequency.lower().startswith('week'):
#         days_ahead = (7 - date.weekday()) % 7
#         if days_ahead == 0:
#             days_ahead = 7
#         return date + timedelta(days=days_ahead)
#     elif frequency.lower().startswith('month'):
#         return (date.replace(day=1) + relativedelta(months=1))
#     raise ValueError(f"Unsupported frequency: {frequency}")
# 
# def generate_forecast_dates(start_date, horizon_delta, frequency):
#     dates = []
#     current = start_date
#     increment = timedelta(weeks=1) if frequency.lower().startswith('week') else relativedelta(months=1)
#     end_date = start_date + horizon_delta
#     while current <= end_date:
#         dates.append(current)
#         current += increment
#     return dates
# 
# def create_time_based_features(df, time_column):
#     df = df.copy()
#     df[time_column] = pd.to_datetime(df[time_column])
#     df["month"] = df[time_column].dt.month
#     df["day_of_week"] = df[time_column].dt.dayofweek
#     df["week_of_year"] = df[time_column].dt.isocalendar().week.astype(int)
#     return df
# 
# def create_lag_features(df, group_column, columns_to_lag, time_column, lags=3):
#     df = df.copy()
#     if time_column not in df.columns:
#         raise ValueError(f"Time column '{time_column}' not found in DataFrame.")
#     df = df.sort_values([group_column, time_column])
#     for col in columns_to_lag:
#         for i in range(1, lags + 1):
#             df[f"{col}_lag_{i}"] = df.groupby(group_column)[col].shift(i)
#     return df
# 
# # -------------------------------
# # Static Feature Aggregation
# # -------------------------------
# def aggregate_static_features(df, entity_col, static_cols):
#     available_cols = [col for col in static_cols if col in df.columns]
#     if not available_cols:
#         logger.warning("No static columns available for aggregation.")
#         return pd.DataFrame({entity_col: df[entity_col].unique()})
#     agg_dict = {}
#     for col in available_cols:
#         if pd.api.types.is_numeric_dtype(df[col]):
#             agg_dict[col] = 'mean'
#         else:
#             agg_dict[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
#     agg_df = df.groupby(entity_col)[available_cols].agg(agg_dict).reset_index()
#     return agg_df
# 
# # -------------------------------
# # Main Training Pipeline
# # -------------------------------
# def train_pipeline_timeseries(
#     df: pd.DataFrame,
#     target_column: str,
#     user_id: str,
#     chat_id: str,
#     column_id: str,
#     time_column: str = None,
#     freq: str = "weekly",
#     forecast_horizon: str = "30 days",
#     use_time_series: bool = False
# ):
#     try:
#         # import pdb; pdb.set_trace()
#         logger.info("Dataset received successfully.")
#         logger.info(f"Dataset shape: {df.shape}")
#         if df.shape[0] == 0:
#             raise ValueError("DataFrame is empty.")
#         if target_column not in df.columns:
#             raise ValueError(f"Target column '{target_column}' not found.")
#         if column_id not in df.columns:
#             raise ValueError(f"ID column '{column_id}' not found.")
# 
#         training_id = chat_id
#         start_time = time.time()
#         metadata_api_url = "http://127.0.0.1:8000/api/update_prediction_status/"
# 
#         # 1) Preprocess Entire Dataset
#         if use_time_series and time_column:
#             logger.info("Performing time-series preprocessing...")
#             if time_column not in df.columns:
#                 raise ValueError(f"Time column '{time_column}' not found.")
#             df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
#             df = create_time_based_features(df, time_column)
#             columns_to_lag = [col for col in df.columns if col not in [time_column, column_id] and 'lag' not in col]
#             df = create_lag_features(df, column_id, columns_to_lag, time_column)
# 
#         logger.info("Performing missing value imputation...")
#         df_imputed, imputers = automatic_imputation(df, target_column=target_column)
#         logger.info("Handling outliers...")
#         df_outlier_fixed, outlier_bounds = detect_and_handle_outliers_train(df_imputed, factor=1.5)
#         logger.info("Encoding categorical features...")
#         df_encoded, encoders = handle_categorical_features(
#             df=df_outlier_fixed,
#             target_column=target_column,
#             id_column=column_id,
#             cardinality_threshold=3
#         )
# 
#         # Update columns_to_lag to include encoded features
#         encoded_columns = [col for col in df_encoded.columns if col not in [time_column, column_id, target_column] and 'lag' not in col]
#         df_encoded = create_lag_features(df_encoded, column_id, encoded_columns, time_column)
# 
#         # 2) Single Training on Entire Dataset
#         horizon_delta = parse_horizon(forecast_horizon)
#         global_max_date = df[time_column].max()
#         train_cutoff = global_max_date - horizon_delta
#         train_data = df_encoded[df_encoded[time_column] <= train_cutoff].copy()
#         logger.info(f"Training set shape: {train_data.shape}")
# 
#         logger.info("Performing feature engineering on training data...")
#         train_data = train_data.rename(columns={time_column: "analysis_time"})
#         df_engineered, feature_defs = feature_engineering_timeseries(
#             train_data,
#             target_column=target_column,
#             id_column=column_id,
#             time_column="analysis_time",
#             training=True
#         )
#         logger.info(f"Training set shape after feature engineering: {df_engineered.shape}")
#         logger.info("feature_defs: " + str(feature_defs))
#         df_engineered = normalize_column_names(df_engineered)
# 
#         logger.info("Performing feature selection...")
#         df_selected, selected_features = feature_selection(
#             df_engineered,
#             target_column=target_column,
#             task="regression",
#             id_column=column_id
#         )
#         logger.info(f"Selected features: {selected_features}")
#         X_train = df_selected.drop(columns=[target_column, column_id], errors='ignore')
#         y_train = df_selected[target_column]
# 
#         logger.info("Selecting best model...")
#         best_model_name, _, _, _, _, _ = train_test_model_selection_timeseries(
#             df_selected,
#             target_column=target_column,
#             id_column=column_id,
#             time_column="analysis_time",
#             task='regression'
#         )
#         logger.info(f"Tuning hyperparameters for {best_model_name}...")
#         best_model, best_params = hyperparameter_tuning(
#             best_model_name=best_model_name,
#             X_train=X_train,
#             y_train=y_train,
#             X_test=None,
#             y_test=None,
#             task='regression'
#         )
# 
#         # Train the final model directly in the pipeline
#         logger.info("Training final model with best hyperparameters...")
#         final_model = best_model.__class__(**best_params)
#         final_model.fit(X_train, y_train)
# 
# 
#         # 3) Validation Predictions Per Product
#         logger.info("Generating validation predictions...")
#         products = df[column_id].unique()
#         validation_metrics = {}
#         predictions = []
# 
#         for product in products:
#             product_data = df_encoded[df_encoded[column_id] == product].copy()
#             max_date = product_data[time_column].max()
#             val_data = product_data[product_data[time_column] > max_date - horizon_delta].copy()
# 
#             if len(val_data) == 0:
#                 logger.info(f"Skipping validation for {product} due to no validation data")
#                 continue
# 
#             val_data = create_lag_features(val_data, column_id, encoded_columns, time_column)
#             logger.info(f"Validation set shape for {product}: {val_data.shape}")
# 
#             val_engineered = feature_engineering_timeseries(
#                 val_data.rename(columns={time_column: "analysis_time"}),
#                 target_column=target_column,
#                 id_column=column_id,
#                 time_column="analysis_time",
#                 training=False,
#                 feature_defs=feature_defs
#             )
#             logger.info(f"Validation set shape for {product}: {val_engineered.shape}")
#             val_engineered = normalize_column_names(val_engineered)
#             X_val = val_engineered.reindex(columns=X_train.columns, fill_value=0)
#             val_predictions = best_model.predict(X_val)
# 
#             val_df = pd.DataFrame({
#                 'analysis_time': val_data[time_column],
#                 column_id: product,
#                 'actual': val_data[target_column],
#                 'predicted': val_predictions
#             })
#             aggregated_preds = val_df.groupby('analysis_time').agg({
#                 column_id: 'first',
#                 'actual': 'mean',
#                 'predicted': 'mean'
#             }).reset_index()
#             predictions.append(aggregated_preds)
#             logger.info(f"Validation predictions shape for {product}: {aggregated_preds.shape}")
#             
#             
# 
#             # Calculate validation metrics for this product
#             mse = mean_squared_error(aggregated_preds['actual'], aggregated_preds['predicted'])
#             mae = mean_absolute_error(aggregated_preds['actual'], aggregated_preds['predicted'])
#             r2 = r2_score(aggregated_preds['actual'], aggregated_preds['predicted']) if len(aggregated_preds['actual']) > 1 else 0.0
#             validation_metrics[product] = {'RMSE': mse, 'MAE': mae, 'R2': r2}
#             logger.info(f"Product {product} - Validation MSE: {mse:.2f}")
#         predictions_df = pd.concat(predictions, ignore_index=True)
#         logger.info("Validation predictions shape: " + str(predictions_df.shape))
#         logger.info("Validation predictions sample:\n" + str(predictions_df.head()))
#             
# 
#         print(predictions_df)
#         # import pdb; pdb.set_trace()
# 
#         # 4) Future Predictions
#         # static_cols = [col for col in df_encoded.columns if col not in [time_column, column_id, target_column] and 'lag' not in col]
#         # entity_static = aggregate_static_features(df_encoded, column_id, static_cols)
#         # forecast_list = []
# 
#         # for product in products:
#         #     product_data = df_encoded[df_encoded[column_id] == product].copy()
#         #     max_date = product_data[time_column].max()
#         #     forecast_start = next_aligned(max_date, freq)
#         #     forecast_dates = generate_forecast_dates(forecast_start, horizon_delta, freq)
# 
#         #     for future_date in forecast_dates:
#         #         future_df = pd.DataFrame({
#         #             column_id: [product],
#         #             time_column: [future_date]
#         #         })
#         #         future_df = future_df.merge(entity_static, on=column_id, how='left')
#         #         future_df = create_time_based_features(future_df, time_column)
# 
#         #         # Add lag features using last known values
#         #         last_train = product_data.sort_values(time_column).tail(1)
#         #         for col in encoded_columns:
#         #             if col in last_train.columns:
#         #                 future_df[f"{col}_lag_1"] = last_train[col].values[0]
#         #             future_df[f"{col}_lag_2"] = 0
#         #             future_df[f"{col}_lag_3"] = 0
#                     
#         #         # Apply encoding to future_df using saved encoders
#         #         future_encoded, _ = handle_categorical_features(
#         #             df=future_df,
#         #             target_column=target_column,
#         #             id_column=column_id,
#         #             encoders=encoders,
#         #             cardinality_threshold=3,
#         #             saved_column_names=encoded_columns + [time_column, column_id]
#         #         )
#         #         # Ensure time_column is preserved
#         #         if time_column not in future_encoded.columns:
#         #             future_encoded[time_column] = future_df[time_column]
#         #         if column_id not in future_encoded.columns:
#         #             future_encoded[column_id] = future_df[column_id]
#                 
#         #         future_encoded = create_lag_features(future_encoded, column_id, encoded_columns, time_column)
# 
#         #         future_engineered = feature_engineering_timeseries(
#         #             future_encoded.rename(columns={time_column: "analysis_time"}),
#         #             target_column=target_column,
#         #             id_column=column_id,
#         #             time_column="analysis_time",
#         #             training=False,
#         #             feature_defs=feature_defs
#         #         )
#         #         future_engineered = normalize_column_names(future_engineered)
#         #         X_future = future_engineered.reindex(columns=X_train.columns, fill_value=0)
#         #         future_pred = best_model.predict(X_future)[0]
#         #         forecast_list.append({
#         #             'analysis_time': future_date,
#         #             column_id: product,
#         #             'predicted': future_pred
#         #         })
# 
#         # 5) Combine and Format Output
#         validation_preds = pd.concat(predictions) if predictions else pd.DataFrame()
#         # future_preds = pd.DataFrame(forecast_list)
#         # forecast_output = pd.concat([
#         #     validation_preds.rename(columns={'analysis_time': 'Marker', 'actual': 'Actual', 'predicted': 'Prediction'}),
#         #     future_preds.rename(columns={'analysis_time': 'Marker', 'predicted': 'Prediction'})
#         # ])
#         # forecast_output['pecan_id'] = range(864, 864 + len(forecast_output))
#         # forecast_output['pecan_model_id'] = best_model_name
#         # forecast_output['sampled_date'] = forecast_output['Marker']
#         # forecast_output = forecast_output[['pecan_id', column_id, 'sampled_date', 'Marker', 'Prediction', 'pecan_model_id', 'Actual']]
# 
# 
# 
#         # 6) Finalize and Evaluate Model (using validation metrics)
#         # logger.info("Finalizing and evaluating the model with validation metrics...")
#         # final_model, final_metrics = finalize_and_evaluate_model_timeseries(
#         #     best_model_class=final_model.__class__,
#         #     best_params=best_params,
#         #     X_train=X_train,
#         #     y_train=y_train,
#         #     validation_metrics=validation_metrics,  # Pass validation metrics dictionary
#         #     user_id=user_id,
#         #     chat_id=chat_id
#         # )
#         # In train_pipeline_timeseries (replace the finalize_and_evaluate_model call)
#         # 6) Finalize and Evaluate Model (using validation predictions)
#         logger.info("Finalizing and evaluating the model with validation predictions...")
#         # import pdb; pdb.set_trace()
#         final_model, final_metrics = finalize_and_evaluate_model_timeseries(
#             final_model=final_model,  # Pass the trained model from the pipeline
#             X_train=X_train,
#             predictions_df=predictions_df,  # Pass the concatenated validation predictions
#             user_id=user_id,
#             chat_id=chat_id,
#             best_params=best_params
#         )
# 
#         # 7) Save Artifacts
#         logger.info("Saving artifacts to S3...")
#         bucket_name = "artifacts1137"
#         prefix = f"ml-artifacts/{chat_id}/"
#         def save_to_s3(obj, filename):
#             with io.BytesIO() as f:
#                 joblib.dump(obj, f)
#                 f.seek(0)
#                 upload_to_s3(f, bucket_name, f"{prefix}{filename}")
#         save_to_s3(best_model, 'best_model.joblib')
#         save_to_s3(imputers, 'imputers.joblib')
#         save_to_s3(encoders, 'encoder.joblib')
#         save_to_s3(feature_defs, 'feature_defs.joblib')
#         save_to_s3(selected_features, 'selected_features.pkl')
#         save_to_s3(outlier_bounds, 'outlier_bounds.pkl')
#         save_to_s3(X_train.columns.tolist(), 'saved_column_names.pkl')
#         # save_to_s3(forecast_output, 'forecast_output.pkl')
# 
#         # 7) Final Metadata Update and Return
#         duration = time.time() - start_time
#         requests.patch(f"{metadata_api_url}{training_id}/", json={
#             'status': 'training_completed',
#             'duration': duration
#         })
#         # logger.info("Forecast output shape: " + str(forecast_output.shape))
#         # logger.info("Sample forecast:\n" + str(forecast_output.head()))
#         return best_model, best_params
# 
#     except Exception as e:
#         logger.error(f"Pipeline failed: {e}")
#         raise
#         
# =============================================================================


# =============================================================================
# 
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from io import StringIO
# import boto3
# import pandas as pd
# import numpy as np
# import joblib
# import logging
# from src.s3_operations import get_s3_client, download_from_s3
# from src.data_preprocessing import handle_categorical_features
# from src.feature_engineering import feature_engineering_timeseries
# from src.helper import normalize_column_names
# from src.logging_config import get_logger
# logger = get_logger(__name__)
# 
# def predict_future_timeseries(input_df, chat_id, time_column="analysis_time", column_id="product_id", target_column="daily_demand"):
#     """
#     Loads a trained model and artifacts from S3 and generates future predictions for the input dataset.
# 
#     Parameters:
#     - input_df (pd.DataFrame): Input dataset with product_id, sampled_date, and features (e.g., category, daily_demand, is_holiday, is_promotion).
#     - chat_id (str): Chat identifier to load corresponding artifacts from S3.
#     - time_column (str): Column name for the time index (default: "sampled_date").
#     - column_id (str): Column name for the entity ID (default: "product_id").
#     - target_column (str): Column name for the target variable (default: "daily_demand").
# 
#     Returns:
#     - predictions_df (pd.DataFrame): DataFrame with predictions, including 'product_id', 'sampled_date', and 'predicted' values.
#     """
#     try:
#         logger.info(f"Starting future predictions for chat_id: {chat_id}")
#         # import pdb; pdb.set_trace()
#         
#         # S3 bucket and prefix
#         bucket_name = "artifacts1137"
#         prefix = f"ml-artifacts/{chat_id}/"
# 
#         # Load artifacts from S3
#         logger.info("Loading model and artifacts from S3...")
#         model = joblib.load(load_from_s3(bucket_name, f"{prefix}final_model.joblib"))
#         encoders = joblib.load(load_from_s3(bucket_name, f"{prefix}encoder.joblib"))
#         feature_defs = joblib.load(load_from_s3(bucket_name, f"{prefix}feature_defs.joblib"))
#         selected_features = joblib.load(load_from_s3(bucket_name, f"{prefix}selected_features.pkl"))
#         saved_column_names = joblib.load(load_from_s3(bucket_name, f"{prefix}saved_column_names.pkl"))
#         # imputers        = joblib.load(load_from_s3(bucket_name, prefix + "imputers.joblib"))
# 
# 
#         import pdb; pdb.set_trace()
# 
#         # Preprocess input dataset
#         input_df[time_column] = pd.to_datetime(input_df[time_column], errors="coerce")
#         input_df = create_time_based_features(input_df, time_column)
# 
#         # Identify columns to lag (excluding time_column, column_id, and target_column)
#         columns_to_lag = [col for col in input_df.columns if col not in [time_column, column_id, target_column] and 'lag' not in col]
#         input_df = create_lag_features(input_df, column_id, columns_to_lag, time_column)
# 
#         # Apply categorical encoding using saved encoders
#         input_encoded, _ = handle_categorical_features(
#             df=input_df,
#             target_column=target_column,
#             id_column=column_id,
#             encoders=encoders,
#             cardinality_threshold=3,
#             saved_column_names=columns_to_lag + [time_column, column_id]
#         )
# 
#         # Ensure time_column and column_id are preserved
#         if time_column not in input_encoded.columns:
#             input_encoded[time_column] = input_df[time_column]
#         if column_id not in input_encoded.columns:
#             input_encoded[column_id] = input_df[column_id]
# 
#         input_encoded = create_lag_features(input_encoded, column_id, columns_to_lag, time_column)
# 
#         # Perform feature engineering
#         input_engineered = feature_engineering_timeseries(
#             input_encoded.rename(columns={time_column: "analysis_time"}),
#             target_column=target_column,
#             id_column=column_id,
#             time_column="analysis_time",
#             training=False,
#             feature_defs=feature_defs
#         )
#         input_engineered = normalize_column_names(input_engineered)
#         X_predict = input_engineered.reindex(columns=saved_column_names, fill_value=0)
# 
#         # Generate predictions
#         logger.info("Generating future predictions...")
#         predictions = model.predict(X_predict)
#         
#         # Create predictions DataFrame
#         predictions_df = pd.DataFrame({
#             column_id: input_df[column_id],
#             time_column: input_df[time_column],
#             'predicted': predictions
#         })
# 
#         logger.info(f"Prediction shape: {predictions_df.shape}")
#         logger.info(f"Sample predictions:\n{predictions_df.head()}")
# 
#         # Save predictions to S3 (optional)
#         prediction_key = f"{prefix}future_predictions_{chat_id}.csv"
#         with io.StringIO() as buffer:
#             predictions_df.to_csv(buffer, index=False)
#             buffer.seek(0)
#             upload_to_s3(io.BytesIO(buffer.getvalue().encode()), bucket_name, prediction_key)
#         logger.info(f"Predictions saved to s3://{bucket_name}/{prediction_key}")
# 
#         return predictions_df
# 
#     except Exception as e:
#         logger.error(f"Error during future predictions: {e}")
#         raise
#      
#         
#         
# # Define input data
# input_data = {
#     "file_url": "s3://testingfiles-pacx/predict_f7.csv",
#     "column_id": "product_id",
#     "user_id": "17236",
#     "chat_id": "7236389",
#     "ml_type": True
# }
# 
# def fetch_csv_from_s3(s3_path):
#     """
#     Fetches a CSV file from S3 with error handling for empty data.
#     """
#     try:
#         if not s3_path.startswith("s3://"):
#             raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'.")
#         
#         s3_parts = s3_path.replace("s3://", "").split("/")
#         bucket_name = s3_parts[0]
#         object_key = "/".join(s3_parts[1:])
#         
#         logger.info(f"Fetching CSV from S3 bucket: {bucket_name}, key: {object_key}")
#         
#         s3 = boto3.client(
#             's3',
#             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#             region_name=os.getenv('AWS_S3_REGION_NAME'),
#         )
# 
#         obj = s3.get_object(Bucket=bucket_name, Key=object_key)
#         csv_content = obj['Body'].read().decode('utf-8')
#         df = pd.read_csv(StringIO(csv_content))
# 
#         if df.shape[0] == 0:
#             raise ValueError("Fetched CSV is empty. Cannot proceed with training.")
# 
#         logger.info(f"CSV fetched successfully. DataFrame shape: {df.shape}")
#         return df
# 
#     except Exception as e:
#         logger.error(f"Error fetching CSV from S3: {e}")
#         raise
# 
# df = fetch_csv_from_s3(input_data["file_url"])
# # Drop unwanted columns
# df = df.drop(columns=["entity_id", "date","target_within_30_days_after"], errors="ignore")
# # Call prediction function
# predictions_df = predict_future_timeseries(
#     input_df=df,
#     chat_id=input_data["chat_id"],
#     time_column="analysis_time",
#     column_id=input_data["column_id"],
#     target_column="daily_demand"
# )
# 
# 
# =============================================================================





import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import re
import time
import io
import joblib
import logging
import requests
import featuretools as ft
from sklearn.preprocessing import StandardScaler
from src.outlier_handling import apply_outlier_bounds, detect_and_handle_outliers_train
from src.s3_operations import get_s3_client, upload_to_s3
from src.data_preprocessing import handle_categorical_features
from src.finalization import finalize_and_evaluate_model_timeseries
from src.model_selection import train_test_model_selection_timeseries
from src.hyperparameter_tuning import hyperparameter_tuning
from src.utils import automatic_imputation
from src.feature_engineering import feature_engineering_timeseries
from src.feature_selection import feature_selection
from src.helper import normalize_column_names

logger = logging.getLogger(__name__)
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from datetime import timedelta
from dateutil.relativedelta import relativedelta
import re
import time
import io
import joblib
import logging
import requests
import featuretools as ft
from sklearn.preprocessing import StandardScaler
from src.outlier_handling import apply_outlier_bounds, detect_and_handle_outliers_train
from src.s3_operations import get_s3_client, upload_to_s3
from src.data_preprocessing import handle_categorical_features
from src.finalization import finalize_and_evaluate_model_timeseries
from src.model_selection import train_test_model_selection_timeseries
from src.hyperparameter_tuning import hyperparameter_tuning
from src.utils import automatic_imputation
from src.feature_engineering import feature_engineering_timeseries
from src.feature_selection import feature_selection
from src.helper import normalize_column_names

logger = logging.getLogger(__name__)

# Time Series Utilities
def parse_horizon(horizon_str):
    match = re.match(r'(\d+)\s*(day|days|month|months)', horizon_str, re.IGNORECASE)
    if match:
        value = int(match.group(1))
        unit = match.group(2).lower()
        if 'day' in unit:
            return timedelta(days=value)
        elif 'month' in unit:
            return relativedelta(months=value)
    raise ValueError("Horizon must be like '30 days' or '6 months'")

def next_aligned(date, frequency):
    if frequency.lower().startswith('week'):
        days_ahead = (7 - date.weekday()) % 7
        if days_ahead == 0:
            days_ahead = 7
        return date + timedelta(days=days_ahead)
    elif frequency.lower().startswith('month'):
        return (date.replace(day=1) + relativedelta(months=1))
    raise ValueError(f"Unsupported frequency: {frequency}")

def generate_forecast_dates(start_date, horizon_delta, frequency):
    dates = []
    current = start_date
    increment = timedelta(weeks=1) if frequency.lower().startswith('week') else relativedelta(months=1)
    end_date = start_date + horizon_delta
    while current <= end_date:
        dates.append(current)
        current += increment
    return dates

def create_time_based_features(df, time_column):
    df = df.copy()
    df[time_column] = pd.to_datetime(df[time_column])
    df["month"] = df[time_column].dt.month
    df["day_of_week"] = df[time_column].dt.dayofweek
    df["week_of_year"] = df[time_column].dt.isocalendar().week.astype(int)
    return df

def create_lag_features(df, group_column, columns_to_lag, time_column, lags=3):
    df = df.copy()
    if time_column not in df.columns:
        raise ValueError(f"Time column '{time_column}' not found in DataFrame.")
    df = df.sort_values([group_column, time_column])
    for col in columns_to_lag:
        for i in range(1, lags + 1):
            df[f"{col}_lag_{i}"] = df.groupby(group_column)[col].shift(i)
    return df

# Static Feature Aggregation
def aggregate_static_features(df, entity_col, static_cols):
    available_cols = [col for col in static_cols if col in df.columns]
    if not available_cols:
        logger.warning("No static columns available for aggregation.")
        return pd.DataFrame({entity_col: df[entity_col].unique()})
    agg_dict = {}
    for col in available_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            agg_dict[col] = 'mean'
        else:
            agg_dict[col] = lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan
    agg_df = df.groupby(entity_col)[available_cols].agg(agg_dict).reset_index()
    return agg_df

# Main Training Pipeline
# def train_pipeline_timeseries(
#     df: pd.DataFrame,
#     target_column: str,
#     user_id: str,
#     chat_id: str,
#     entity_column: str,
#     time_column: str,
#     freq: str,
#     forecast_horizon: str = "30 days",
#     use_time_series: bool = False,
#     machine_learning_type: str = "regression",
#     new_target_column: str = "target_within_90_days_after"
# ):
#     try:
#         logger.info("Dataset received successfully.")
#         logger.info(f"Dataset shape: {df.shape}")
#         if df.shape[0] == 0:
#             raise ValueError("DataFrame is empty.")
#         if new_target_column not in df.columns:
#             raise ValueError(f"Target column '{new_target_column}' not found.")
#         if entity_column not in df.columns:
#             raise ValueError(f"ID column '{entity_column}' not found.")

#         training_id = chat_id
#         start_time = time.time()
#         start_time_iso = datetime.datetime.fromtimestamp(start_time).isoformat()  # Convert to ISO format
#         entity_count = df.shape[0]
#         metadata_api_url = "http://127.0.0.1:8000/api/update_prediction_status/"

#         response = requests.post(metadata_api_url, json={
#             'prediction_id': training_id,
#             'chat_id': chat_id,
#             'user_id': user_id,
#             'status': 'inprogress',
#             'entity_count': entity_count,
#             'start_time': start_time_iso
#         })
#         if response.status_code != 201:
#             logger.error(f"Failed to create metadata: {response.json()}")
#             raise RuntimeError("Initial metadata creation failed.")

#         # 1) Preprocess Entire Dataset
#         if use_time_series and time_column:
#             logger.info("Performing time-series preprocessing...")
#             if time_column not in df.columns:
#                 raise ValueError(f"Time column '{time_column}' not found.")
#             df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
#             df = create_time_based_features(df, time_column)
#             columns_to_lag = [col for col in df.columns if col not in [time_column, entity_column, new_target_column] and 'lag' not in col]
#             df = create_lag_features(df, entity_column, columns_to_lag, time_column)

#         logger.info("Performing missing value imputation...")
#         df_imputed, imputers = automatic_imputation(df, target_column=new_target_column)
#         logger.info("Handling outliers...")
#         df_outlier_fixed, outlier_bounds = detect_and_handle_outliers_train(df_imputed, factor=1.5)
#         logger.info("Encoding categorical features...")
#         df_encoded, encoders = handle_categorical_features(
#             df=df_outlier_fixed,
#             target_column=new_target_column,
#             id_column=entity_column,
#             cardinality_threshold=3
#         )

#         # Update columns_to_lag to include encoded features
#         encoded_columns = [col for col in df_encoded.columns if col not in [time_column, entity_column, new_target_column] and 'lag' not in col]
#         df_encoded = create_lag_features(df_encoded, entity_column, encoded_columns, time_column)

#         # 2) Single Training on Entire Dataset with sampled_date
#         horizon_delta = parse_horizon(forecast_horizon)
#         df_encoded[time_column] = pd.to_datetime(df_encoded[time_column])
#         sampled_dates = df_encoded.groupby(entity_column)[time_column].max().reset_index()
#         sampled_dates.columns = [entity_column, 'sampled_date']
#         train_data = pd.merge(df_encoded, sampled_dates, on=entity_column, how='left')
#         train_data = train_data[train_data[time_column] <= train_data['sampled_date'] - horizon_delta].copy()
#         logger.info(f"Training set shape: {train_data.shape}")

#         logger.info("Performing feature engineering on training data...")
#         # train_data = train_data.rename(columns={time_column: "analysis_time"})
#         df_engineered, feature_defs = feature_engineering_timeseries(
#             train_data,
#             target_column=new_target_column,
#             id_column=entity_column,
#             # time_column="analysis_time",
#             time_column=time_column,
#             training=True
#         )
#         requests.patch(f"{metadata_api_url}{training_id}/", json={'status': 'feature_engineering_completed'})
#         logger.info(f"Generated features in df_engineered: {df_engineered.columns.tolist()}")
#         df_engineered = normalize_column_names(df_engineered)
#         df_engineered = df_engineered.drop(columns=['date'], errors='ignore')  # Remove date after feature engineering

#         logger.info("Performing feature selection...")
#         df_selected, selected_features = feature_selection(
#             df_engineered,
#             target_column=new_target_column,
#             task="regression",
#             id_column=entity_column
#         )
#         # Ensure analysis_time is preserved for model selection but not as a feature
#         if 'analysis_time' in df_selected.columns:
#             analysis_time = df_selected['analysis_time'].copy()
#             df_selected = df_selected.drop(columns=['analysis_time'])
#         else:
#             analysis_time = None
#         X_train = df_selected.drop(columns=[new_target_column, entity_column], errors='ignore')
#         y_train = df_selected[new_target_column]

#         logger.info("Selecting best model...")
#         best_model_name, _, _, _, _, _ = train_test_model_selection_timeseries(
#             df_selected.assign(analysis_time=analysis_time) if analysis_time is not None else df_selected,
#             target_column=new_target_column,
#             id_column=entity_column,
#             time_column="analysis_time",
#             task='regression'
#         )
#         logger.info(f"Tuning hyperparameters for {best_model_name}...")
#         best_model, best_params = hyperparameter_tuning(
#             best_model_name=best_model_name,
#             X_train=X_train,
#             y_train=y_train,
#             X_test=None,
#             y_test=None,
#             task='regression'
#         )
#         requests.patch(f"{metadata_api_url}{training_id}/", json={'status': 'hyperparameter_tuning_completed'})

#         # Train the final model directly in the pipeline
#         logger.info("Training final model with best hyperparameters...")
#         final_model = best_model.__class__(**best_params)
#         final_model.fit(X_train, y_train)

#         # 3) Validation Predictions Per Product
#         logger.info("Generating validation predictions...")
#         requests.patch(f"{metadata_api_url}{training_id}/", json={'status': 'validation_predictions_started'})
#         products = df[entity_column].unique()
#         validation_metrics = {}
#         predictions = []

#         for product in products:
#             product_data = df_encoded[df_encoded[entity_column] == product].copy()
#             sampled_date = sampled_dates[sampled_dates[entity_column] == product]['sampled_date'].iloc[0]
#             val_data = product_data[(product_data[time_column] > sampled_date - horizon_delta) & 
#                                   (product_data[time_column] <= sampled_date)].copy()

#             if len(val_data) == 0:
#                 logger.info(f"Skipping validation for {product} due to no validation data")
#                 continue

#             val_data = create_lag_features(val_data, entity_column, encoded_columns, time_column)
#             logger.info(f"Validation set shape for {product}: {val_data.shape}")

#             val_engineered = feature_engineering_timeseries(
#                 # val_data.rename(columns={time_column: "analysis_time"}),
#                 val_data,
#                 target_column=target_column,
#                 id_column=entity_column,
#                 time_column=time_column,
#                 training=False,
#                 feature_defs=feature_defs
#             )
#             requests.patch(f"{metadata_api_url}{training_id}/", json={'status': 'validation_feature_engineering_completed'})
#             val_engineered = normalize_column_names(val_engineered)
#             # import pdb; pdb.set_trace()
#             X_val = val_engineered.reindex(columns=X_train.columns, fill_value=0)
#             val_predictions = final_model.predict(X_val)

#             val_df = pd.DataFrame({
#                 'analysis_time': val_data[time_column],
#                 entity_column: product,
#                 'actual': val_data[new_target_column],
#                 'predicted': val_predictions
#             })
#             aggregated_preds = val_df.groupby('analysis_time').agg({
#                 entity_column: 'first',
#                 'actual': 'mean',
#                 'predicted': 'mean'
#             }).reset_index()
#             predictions.append(aggregated_preds)
#             logger.info(f"Validation predictions shape for {product}: {aggregated_preds.shape}")
            
#             mse = mean_squared_error(aggregated_preds['actual'], aggregated_preds['predicted'])
#             mae = mean_absolute_error(aggregated_preds['actual'], aggregated_preds['predicted'])
#             r2 = r2_score(aggregated_preds['actual'], aggregated_preds['predicted']) if len(aggregated_preds['actual']) > 1 else 0.0
#             validation_metrics[product] = {'RMSE': mse, 'MAE': mae, 'R2': r2}
#             logger.info(f"Product {product} - Validation MSE: {mse:.2f}")

#         predictions_df = pd.concat(predictions, ignore_index=True)
#         logger.info("Validation predictions shape: " + str(predictions_df.shape))
#         logger.info("Validation predictions sample:\n" + str(predictions_df.head()))

#         # 6) Finalize and Evaluate Model (using validation predictions)
#         logger.info("Finalizing and evaluating the model with validation predictions...")
#         final_model, final_metrics = finalize_and_evaluate_model_timeseries(
#             final_model=final_model,
#             X_train=X_train,
#             predictions_df=predictions_df,
#             user_id=user_id,
#             chat_id=chat_id,
#             best_params=best_params
#         )

#         # 7) Save Artifacts
#         logger.info("Saving artifacts to S3...")
#         bucket_name = "artifacts1137"
#         prefix = f"ml-artifacts/{chat_id}/"
#         def save_to_s3(obj, filename):
#             with io.BytesIO() as f:
#                 joblib.dump(obj, f)
#                 f.seek(0)
#                 upload_to_s3(f, bucket_name, f"{prefix}{filename}")
#         save_to_s3(final_model, "final_model.joblib")
#         save_to_s3(imputers, "imputers.joblib")
#         save_to_s3(encoders, "encoders.joblib")
#         save_to_s3(feature_defs, "feature_defs.joblib")
#         save_to_s3(selected_features, "selected_features.pkl")
#         save_to_s3(outlier_bounds, "outlier_bounds.pkl")
#         save_to_s3(X_train.columns.tolist(), "saved_column_names.pkl")
#         save_to_s3(df_encoded, "historical_data.joblib")  # Save historical data for lag fallback

#         # 8) Final Metadata Update and Return
#         duration = time.time() - start_time
#         requests.patch(f"{metadata_api_url}{training_id}/", json={
#             "status": "training_completed",
#             "duration": duration
#         })
#         logger.info(f"Training completed. Total duration: {duration:.2f} seconds")
#         return final_model, best_params

#     except Exception as e:
#         logger.error(f"Pipeline failed: {e}")
#         raise


def train_pipeline_timeseries(
    df: pd.DataFrame,
    target_column: str,
    user_id: str,
    chat_id: str,
    entity_column: str,
    time_column: str,
    freq: str,
    forecast_horizon: str = "30 days",
    use_time_series: bool = False,
    machine_learning_type: str = "regression",
    new_target_column: str = "target_within_90_days_after"
):
    start_time = time.time()
    start_time_iso = datetime.datetime.fromtimestamp(start_time).isoformat()
    training_id = chat_id
    entity_count = df.shape[0]
    metadata_api_url = "http://127.0.0.1:8000/api/update_prediction_status/"
    current_status = "inprogress"

    def update_status(status: str, error: str = None, duration: float = None):
        """Helper function to update prediction status via API."""
        payload = {
            "prediction_id": training_id,
            "chat_id": chat_id,
            "user_id": user_id,
            "status": status,
            "entity_count": entity_count,
            "duration": duration if duration is not None else None,
            "error": error if error else None,
        }
        # Include start_time only for the initial creation (handled by POST), omit for updates
        if status == "inprogress":
            payload["start_time"] = start_time_iso
            try:
                response = requests.post(metadata_api_url, json=payload)
                if response.status_code != 201:
                    logger.error(f"Failed to create metadata: {response.status_code} - {response.text}")
                    raise RuntimeError(f"Initial metadata creation failed: {response.text}")
                logger.info(f"Metadata created successfully with status: {status}")
            except Exception as e:
                logger.error(f"Initial status creation failed: {e}")
                raise
        else:
            try:
                response = requests.patch(f"{metadata_api_url}{training_id}/", json=payload)
                if response.status_code != 200:
                    logger.error(f"Failed to update status to {status}: {response.status_code} - {response.text}")
                    raise RuntimeError(f"Status update failed: {response.text}")
                logger.info(f"Status updated to {status} successfully")
            except Exception as e:
                logger.error(f"Error updating status to {status}: {e}")
                raise

    # Initial status update with POST to create the record
    update_status("inprogress")

    try:
        logger.info("Dataset received successfully.")
        logger.info(f"Dataset shape: {df.shape}")
        if df.shape[0] == 0:
            raise ValueError("DataFrame is empty.")
        if new_target_column not in df.columns:
            raise ValueError(f"Target column '{new_target_column}' not found.")
        if entity_column not in df.columns:
            raise ValueError(f"ID column '{entity_column}' not found.")

        # 1) Preprocess Entire Dataset
        if use_time_series and time_column:
            logger.info("Performing time-series preprocessing...")
            if time_column not in df.columns:
                raise ValueError(f"Time column '{time_column}' not found.")
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
            df = create_time_based_features(df, time_column)
            columns_to_lag = [col for col in df.columns if col not in [time_column, entity_column, new_target_column] and 'lag' not in col]
            df = create_lag_features(df, entity_column, columns_to_lag, time_column)

        logger.info("Performing missing value imputation...")
        df_imputed, imputers = automatic_imputation(df, target_column=new_target_column)
        logger.info("Handling outliers...")
        df_outlier_fixed, outlier_bounds = detect_and_handle_outliers_train(df_imputed, factor=1.5)
        logger.info("Encoding categorical features...")
        df_encoded, encoders = handle_categorical_features(
            df=df_outlier_fixed,
            target_column=new_target_column,
            id_column=entity_column,
            cardinality_threshold=3
        )

        # Update columns_to_lag to include encoded features
        encoded_columns = [col for col in df_encoded.columns if col not in [time_column, entity_column, new_target_column] and 'lag' not in col]
        df_encoded = create_lag_features(df_encoded, entity_column, encoded_columns, time_column)

        # 2) Single Training on Entire Dataset with sampled_date
        horizon_delta = parse_horizon(forecast_horizon)
        df_encoded[time_column] = pd.to_datetime(df_encoded[time_column])
        sampled_dates = df_encoded.groupby(entity_column)[time_column].max().reset_index()
        sampled_dates.columns = [entity_column, 'sampled_date']
        train_data = pd.merge(df_encoded, sampled_dates, on=entity_column, how='left')
        #train_data = train_data[train_data[time_column] <= train_data['sampled_date'] - horizon_delta].copy()
        train_data = train_data[train_data[time_column] <= train_data['sampled_date'].apply(lambda x: x - horizon_delta)].copy()
        logger.info(f"Training set shape: {train_data.shape}")

        logger.info("Performing feature engineering on training data...")
        df_engineered, feature_defs = feature_engineering_timeseries(
            train_data,
            target_column=new_target_column,
            id_column=entity_column,
            time_column=time_column,
            training=True
        )
        current_status = "feature_engineering_completed"
        update_status(current_status)

        logger.info(f"Generated features in df_engineered: {df_engineered.columns.tolist()}")
        df_engineered = normalize_column_names(df_engineered)
        df_engineered = df_engineered.drop(columns=['date'], errors='ignore')

        logger.info("Performing feature selection...")
        df_selected, selected_features = feature_selection(
            df_engineered,
            target_column=new_target_column,
            task="regression",
            id_column=entity_column
        )
        if 'analysis_time' in df_selected.columns:
            analysis_time = df_selected['analysis_time'].copy()
            df_selected = df_selected.drop(columns=['analysis_time'])
        else:
            analysis_time = None
        X_train = df_selected.drop(columns=[new_target_column, entity_column], errors='ignore')
        y_train = df_selected[new_target_column]

        logger.info("Selecting best model...")
        best_model_name, _, _, _, _, _ = train_test_model_selection_timeseries(
            df_selected.assign(analysis_time=analysis_time) if analysis_time is not None else df_selected,
            target_column=new_target_column,
            id_column=entity_column,
            time_column="analysis_time",
            task='regression'
        )
        logger.info(f"Tuning hyperparameters for {best_model_name}...")
        best_model, best_params = hyperparameter_tuning(
            best_model_name=best_model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=None,
            y_test=None,
            task='regression'
        )
        current_status = "hyperparameter_tuning_completed"
        update_status(current_status)

        logger.info("Training final model with best hyperparameters...")
        final_model = best_model.__class__(**best_params)
        final_model.fit(X_train, y_train)

        logger.info("Generating validation predictions...")
        current_status = "validation_predictions_inprogress"
        update_status(current_status)
        products = df[entity_column].unique()
        validation_metrics = {}
        predictions = []

        for product in products:
            product_data = df_encoded[df_encoded[entity_column] == product].copy()
            sampled_date = sampled_dates[sampled_dates[entity_column] == product]['sampled_date'].iloc[0]
            val_data = product_data[(product_data[time_column] > sampled_date - horizon_delta) & 
                                  (product_data[time_column] <= sampled_date)].copy()

            if len(val_data) == 0:
                logger.info(f"Skipping validation for {product} due to no validation data")
                continue

            val_data = create_lag_features(val_data, entity_column, encoded_columns, time_column)
            logger.info(f"Validation set shape for {product}: {val_data.shape}")

            val_engineered = feature_engineering_timeseries(
                val_data,
                target_column=target_column,
                id_column=entity_column,
                time_column=time_column,
                training=False,
                feature_defs=feature_defs
            )
            current_status = "validation_feature_engineering_completed"
            update_status(current_status)
            val_engineered = normalize_column_names(val_engineered)
            X_val = val_engineered.reindex(columns=X_train.columns, fill_value=0)
            val_predictions = final_model.predict(X_val)

            val_df = pd.DataFrame({
                'analysis_time': val_data[time_column],
                entity_column: product,
                'actual': val_data[new_target_column],
                'predicted': val_predictions
            })
            aggregated_preds = val_df.groupby('analysis_time').agg({
                entity_column: 'first',
                'actual': 'mean',
                'predicted': 'mean'
            }).reset_index()
            predictions.append(aggregated_preds)
            logger.info(f"Validation predictions shape for {product}: {aggregated_preds.shape}")
            
            mse = mean_squared_error(aggregated_preds['actual'], aggregated_preds['predicted'])
            mae = mean_absolute_error(aggregated_preds['actual'], aggregated_preds['predicted'])
            r2 = r2_score(aggregated_preds['actual'], aggregated_preds['predicted']) if len(aggregated_preds['actual']) > 1 else 0.0
            validation_metrics[product] = {'RMSE': mse, 'MAE': mae, 'R2': r2}
            logger.info(f"Product {product} - Validation MSE: {mse:.2f}")

        predictions_df = pd.concat(predictions, ignore_index=True)
        logger.info("Validation predictions shape: " + str(predictions_df.shape))
        logger.info("Validation predictions sample:\n" + str(predictions_df.head()))

        logger.info("Finalizing and evaluating the model with validation predictions...")
        final_model, final_metrics = finalize_and_evaluate_model_timeseries(
            final_model=final_model,
            X_train=X_train,
            predictions_df=predictions_df,
            user_id=user_id,
            chat_id=chat_id,
            best_params=best_params,
            entity_column=entity_column,
        )

        logger.info("Saving artifacts to S3...")
        bucket_name = "artifacts1137"
        prefix = f"ml-artifacts/{chat_id}/"
        def save_to_s3(obj, filename):
            with io.BytesIO() as f:
                joblib.dump(obj, f)
                f.seek(0)
                upload_to_s3(f, bucket_name, f"{prefix}{filename}")
        save_to_s3(final_model, "final_model.joblib")
        save_to_s3(imputers, "imputers.joblib")
        save_to_s3(encoders, "encoders.joblib")
        save_to_s3(feature_defs, "feature_defs.joblib")
        save_to_s3(selected_features, "selected_features.pkl")
        save_to_s3(outlier_bounds, "outlier_bounds.pkl")
        save_to_s3(X_train.columns.tolist(), "saved_column_names.pkl")
        # save_to_s3(df_encoded, "historical_data.joblib")

        
        # Avoid duplication by filtering out existing entity_column
        other_columns = [col for col in df_encoded.columns if col != entity_column]
        df_encoded_with_entity = df_encoded[[entity_column] + other_columns]
        save_to_s3(df_encoded_with_entity, "historical_data.joblib")

        # Final status update
        duration = time.time() - start_time
        current_status = "training_completed"
        update_status(current_status, duration=duration)

        logger.info(f"Training completed. Total duration: {duration:.2f} seconds")
        return final_model, best_params

    except Exception as e:
        error_message = str(e)
        logger.error(f"Pipeline failed: {error_message}")
        current_status = "failed"
        update_status(current_status, error=error_message)
        raise

        
        
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

logger = get_logger(__name__)

# def predict_future_timeseries(input_df, chat_id, time_column="analysis_time", column_id="product_id", target_column="target_within_30_days_after"):
#     """
#     Loads a trained model and artifacts from S3 and generates future predictions for the input dataset,
#     using historical data for lags.

#     Parameters:
#     - input_df (pd.DataFrame): Input dataset with product_id, analysis_time, and features (e.g., category, is_holiday, is_promotion).
#     - chat_id (str): Chat identifier to load corresponding artifacts from S3.
#     - time_column (str): Column name for the time index (default: "analysis_time").
#     - column_id (str): Column name for the entity ID (default: "product_id").
#     - target_column (str): Column name for the target variable (default: "daily_demand", ignored for prediction).

#     Returns:
#     - predictions_df (pd.DataFrame): DataFrame with predictions, including 'product_id', 'analysis_time', and 'predicted' values.
#     """
#     try:
#         logger.info(f"Starting future predictions for chat_id: {chat_id}")
        
#         # S3 bucket and prefix
#         bucket_name = "artifacts1137"
#         prefix = f"ml-artifacts/{chat_id}/"

#         # Load artifacts from S3
#         logger.info("Loading model and artifacts from S3...")
#         model = joblib.load(load_from_s3(bucket_name, f"{prefix}final_model.joblib"))
#         encoders = joblib.load(load_from_s3(bucket_name, f"{prefix}encoder.joblib"))
#         feature_defs = joblib.load(load_from_s3(bucket_name, f"{prefix}feature_defs.joblib"))
#         selected_features = joblib.load(load_from_s3(bucket_name, f"{prefix}selected_features.pkl"))
#         saved_column_names = joblib.load(load_from_s3(bucket_name, f"{prefix}saved_column_names.pkl"))
#         historical_data = joblib.load(load_from_s3(bucket_name, f"{prefix}historical_data.joblib"))

#         # Preprocess input dataset
#         input_df = input_df.copy()
#         input_df[time_column] = pd.to_datetime(input_df[time_column], errors="coerce")
#         input_df = create_time_based_features(input_df, time_column)

#         # Drop unnecessary columns (e.g., target for prediction)
#         input_df = input_df.drop(columns=[target_column], errors="ignore")

#         # Identify columns to lag (excluding time_column, column_id, target_column, and daily_demand)
#         columns_to_lag = [col for col in input_df.columns if col not in [time_column, column_id, target_column] and 'lag' not in col]
        
#         # Use historical data to populate lags
#         if columns_to_lag and not historical_data.empty:
#             last_historical = historical_data.groupby(column_id).tail(1).set_index(column_id)
#             for col in columns_to_lag:
#                 input_df = pd.merge(input_df, last_historical[[col]], on=column_id, how='left', suffixes=('', '_hist'))
#                 input_df[f"{col}_lag_1"] = input_df[f"{col}_hist"].fillna(0.0)
#                 input_df[f"{col}_lag_2"] = 0.0
#                 input_df[f"{col}_lag_3"] = 0.0
#                 input_df.drop(columns=[f"{col}_hist"], inplace=True)
#         else:
#             logger.warning("No historical data or columns to lag; using zeros for lags.")
#             for col in columns_to_lag:
#                 input_df[f"{col}_lag_1"] = 0.0
#                 input_df[f"{col}_lag_2"] = 0.0
#                 input_df[f"{col}_lag_3"] = 0.0

#         # Apply categorical encoding using saved encoders
#         input_encoded, _ = handle_categorical_features(
#             df=input_df,
#             target_column=target_column,
#             id_column=column_id,
#             encoders=encoders,
#             training=False,
#             cardinality_threshold=3,
#             saved_column_names=columns_to_lag + [time_column, column_id]
#         )

#         # Ensure time_column and column_id are preserved
#         if time_column not in input_encoded.columns:
#             input_encoded[time_column] = input_df[time_column]
#         if column_id not in input_encoded.columns:
#             input_encoded[column_id] = input_df[column_id]

#         # Recreate lags on encoded data using historical fallback
#         if columns_to_lag and not historical_data.empty:
#             for col in columns_to_lag:
#                 input_encoded = pd.merge(input_encoded, last_historical[[col]], on=column_id, how='left', suffixes=('', '_hist'))
#                 input_encoded[f"{col}_lag_1"] = input_encoded[f"{col}_hist"].fillna(0.0)
#                 input_encoded[f"{col}_lag_2"] = 0.0
#                 input_encoded[f"{col}_lag_3"] = 0.0
#                 input_encoded.drop(columns=[f"{col}_hist"], inplace=True)
#         else:
#             for col in columns_to_lag:
#                 input_encoded[f"{col}_lag_1"] = 0.0
#                 input_encoded[f"{col}_lag_2"] = 0.0
#                 input_encoded[f"{col}_lag_3"] = 0.0          
                
                
#         # Filter feature_defs to include only computable features
#         available_cols = set(input_encoded.columns)
#         computable_defs = [
#             f for f in feature_defs if all(
#                 isinstance(dep, str) and dep in available_cols and dep != target_column
#                 for dep in f.get_dependencies(deep=True) or []
#             )
#         ]
#         if not computable_defs:
#             logger.warning("No computable Featuretools features; falling back to input data.")
#             input_engineered = input_encoded
#         else:
#             # Perform feature engineering
#             input_engineered = feature_engineering_timeseries(
#                 input_encoded.rename(columns={time_column: "analysis_time"}),
#                 target_column=target_column,
#                 id_column=column_id,
#                 time_column="analysis_time",
#                 training=False,
#                 feature_defs=computable_defs
#             )
#             input_engineered = normalize_column_names(input_engineered)
#         X_predict = input_engineered.reindex(columns=saved_column_names, fill_value=0)

#         # Generate predictions
#         logger.info("Generating future predictions...")
#         predictions = model.predict(X_predict)
        
#         # Create predictions DataFrame
#         predictions_df = pd.DataFrame({
#             column_id: input_df[column_id],
#             time_column: input_df[time_column],
#             'predicted': predictions
#         })
        
#         # Aggregate predictions by mean per product_id and analysis_time
#         predictions_df = predictions_df.groupby([column_id, time_column]).agg({'predicted': 'mean'}).reset_index()

#         logger.info(f"Prediction shape: {predictions_df.shape}")
#         logger.info(f"Sample predictions:\n{predictions_df.head()}")

#         # Save predictions to S3 (optional)
#         prediction_key = f"{prefix}future_predictions_{chat_id}.csv"
#         with io.StringIO() as buffer:
#             predictions_df.to_csv(buffer, index=False)
#             buffer.seek(0)
#             upload_to_s3(io.BytesIO(buffer.getvalue().encode()), bucket_name, prediction_key)
#         logger.info(f"Predictions saved to s3://{bucket_name}/{prediction_key}")

#         return predictions_df

#     except Exception as e:
#         logger.error(f"Error during future predictions: {e}")
#         raise


def predict_future_timeseries(
    input_df, 
    chat_id, 
    prediction_id,  # New parameter from frontend
    user_id,        # New parameter from frontend
    time_column, 
    entity_column, 
    target_column,
    new_target_column
):
    """
    Loads a trained model and artifacts from S3 and generates future predictions for the input dataset,
    using historical data for lags.

    Parameters:
    - input_df (pd.DataFrame): Input dataset with product_id, analysis_time, and features.
    - chat_id (str): Chat identifier to load corresponding artifacts from S3.
    - prediction_id (str): Unique ID from frontend to track prediction status.
    - user_id (str): User identifier from frontend.
    - time_column (str): Column name for the time index (default: "analysis_time").
    - column_id (str): Column name for the entity ID (default: "product_id").
    - target_column (str): Column name for the target variable (default: "daily_demand", ignored for prediction).

    Returns:
    - predictions_df (pd.DataFrame): DataFrame with predictions, including 'product_id', 'analysis_time', and 'predicted' values.
    """
    try:
        logger.info(f"Starting future predictions for chat_id: {chat_id}, prediction_id: {prediction_id}")
        

        # Metadata API endpoint
        metadata_api_url = "http://127.0.0.1:8000/api/update_prediction_status/"

        # Step 1: POST initial metadata with "inprogress" status
        start_time = datetime.datetime.now()
        entity_count = input_df.shape[0]
        print("------------------------------------------------------------")
        print(f"Starting prediction process for {prediction_id}...")
        print(f"chat_id: {chat_id}, user_id: {user_id}, entity_count: {entity_count}, start_time: {start_time},prediction_id: {prediction_id}")
        response = requests.post(metadata_api_url, json={
            'prediction_id': prediction_id,
            # 'prediction_id': "1234566",  # Placeholder for now
            'chat_id': chat_id,
            # 'chat_id': "123455",  # Placeholder for now
            'user_id': user_id,
            'status': 'inprogress',
            'entity_count': entity_count,
            'start_time': start_time.isoformat()
        })
        if response.status_code != 201:
            logger.error(f"Failed to create metadata: {response.json()}")
            raise RuntimeError("Initial metadata creation failed.")

        logger.info("Starting prediction process...")
        start_timer = time.time()

        # S3 bucket and prefix
        bucket_name = "artifacts1137"
        prefix = f"ml-artifacts/{chat_id}/"
        
        logger.info(f"Loading model and artifacts from S3 location {bucket_name}/{prefix}...")
        logger.info(f"{bucket_name}/{prefix}final_model.joblib")
        encoder_key = f"{prefix}encoder.joblib"
        logger.info(f"Attempting to load: {encoder_key}")

        # Load artifacts from S3
        logger.info("Loading model and artifacts from S3...")
        model = joblib.load(load_from_s3(bucket_name, f"{prefix}final_model.joblib"))
        # model = joblib.load(load_from_s3(bucket_name,"ml-artifacts/8c30705e-5548-457e-9e0f-b74d8ac3c86d90/final_model.joblib"))
        encoders = joblib.load(load_from_s3(bucket_name, f"{prefix}encoders.joblib"))
        feature_defs = joblib.load(load_from_s3(bucket_name, f"{prefix}feature_defs.joblib"))
        selected_features = joblib.load(load_from_s3(bucket_name, f"{prefix}selected_features.pkl"))
        saved_column_names = joblib.load(load_from_s3(bucket_name, f"{prefix}saved_column_names.pkl"))
        historical_data = joblib.load(load_from_s3(bucket_name, f"{prefix}historical_data.joblib"))
        
        

        # Preprocess input dataset
        input_df = input_df.copy()
        input_df[time_column] = pd.to_datetime(input_df[time_column], errors="coerce")
        input_df = create_time_based_features(input_df, time_column)
        input_df = input_df.drop(columns=[new_target_column], errors="ignore")

        # Identify columns to lag
        columns_to_lag = [col for col in input_df.columns if col not in [time_column, entity_column, new_target_column] and 'lag' not in col]
        
        # Use historical data to populate lags
        if columns_to_lag and not historical_data.empty:
            last_historical = historical_data.groupby(entity_column).tail(1).set_index(entity_column)
            for col in columns_to_lag:
                input_df = pd.merge(input_df, last_historical[[col]], on=entity_column, how='left', suffixes=('', '_hist'))
                input_df[f"{col}_lag_1"] = input_df[f"{col}_hist"].fillna(0.0)
                input_df[f"{col}_lag_2"] = 0.0
                input_df[f"{col}_lag_3"] = 0.0
                input_df.drop(columns=[f"{col}_hist"], inplace=True)
        else:
            logger.warning("No historical data or columns to lag; using zeros for lags.")
            for col in columns_to_lag:
                input_df[f"{col}_lag_1"] = 0.0
                input_df[f"{col}_lag_2"] = 0.0
                input_df[f"{col}_lag_3"] = 0.0

        # Apply categorical encoding
        input_encoded, _ = handle_categorical_features(
            df=input_df,
            target_column=target_column,
            id_column=entity_column,
            encoders=encoders,
            training=False,
            cardinality_threshold=3,
            saved_column_names=columns_to_lag + [time_column, entity_column]
        )

        if time_column not in input_encoded.columns:
            input_encoded[time_column] = input_df[time_column]
        if entity_column not in input_encoded.columns:
            input_encoded[entity_column] = input_df[entity_column]

        # Recreate lags on encoded data
        if columns_to_lag and not historical_data.empty:
            for col in columns_to_lag:
                input_encoded = pd.merge(input_encoded, last_historical[[col]], on=entity_column, how='left', suffixes=('', '_hist'))
                input_encoded[f"{col}_lag_1"] = input_encoded[f"{col}_hist"].fillna(0.0)
                input_encoded[f"{col}_lag_2"] = 0.0
                input_encoded[f"{col}_lag_3"] = 0.0
                input_encoded.drop(columns=[f"{col}_hist"], inplace=True)
        else:
            for col in columns_to_lag:
                input_encoded[f"{col}_lag_1"] = 0.0
                input_encoded[f"{col}_lag_2"] = 0.0
                input_encoded[f"{col}_lag_3"] = 0.0

        # Feature engineering
        available_cols = set(input_encoded.columns)
        computable_defs = [
            f for f in feature_defs if all(
                isinstance(dep, str) and dep in available_cols and dep != target_column
                for dep in f.get_dependencies(deep=True) or []
            )
        ]
        if not computable_defs:
            logger.warning("No computable Featuretools features; falling back to input data.")
            input_engineered = input_encoded
        else:
            input_engineered = feature_engineering_timeseries(
                input_encoded.rename(columns={time_column: "analysis_time"}),
                target_column=target_column,
                id_column=entity_column,
                time_column="analysis_time",
                training=False,
                feature_defs=computable_defs
            )
            input_engineered = normalize_column_names(input_engineered)
        X_predict = input_engineered.reindex(columns=saved_column_names, fill_value=0)

        # Generate predictions
        logger.info("Generating future predictions...")
        predictions = model.predict(X_predict)
        
        predictions_df = pd.DataFrame({
            entity_column: input_df[entity_column],
            time_column: input_df[time_column],
            'predicted': predictions
        })
        predictions_df = predictions_df.groupby([entity_column, time_column]).agg({'predicted': 'mean'}).reset_index()

        logger.info(f"Prediction shape: {predictions_df.shape}")
        logger.info(f"Sample predictions:\n{predictions_df.head()}")

        # Save predictions to S3
        prediction_key = f"{prefix}future_predictions_{chat_id}_{prediction_id}.csv"
        with io.StringIO() as buffer:
            predictions_df.to_csv(buffer, index=False)
            buffer.seek(0)
            upload_to_s3(io.BytesIO(buffer.getvalue().encode()), bucket_name, prediction_key)
        logger.info(f"Predictions saved to s3://{bucket_name}/{prediction_key}")

        # Step 2: PATCH metadata with "success" status
        duration = time.time() - start_timer
        response = requests.patch(f"{metadata_api_url}{prediction_id}/", json={
            'status': 'success',
            'duration': duration,
            'predictions_csv_path': prediction_key
        })
        if response.status_code != 200:
            logger.error(f"Failed to update metadata: {response.json()}")
            raise RuntimeError("Final metadata update failed.")

        return predictions_df

    except Exception as e:
        # Step 3: PATCH metadata with "failed" status on error
        duration = time.time() - start_timer if 'start_timer' in locals() else 0
        response = requests.patch(f"{metadata_api_url}{prediction_id}/", json={
            'status': 'failed',
            'duration': duration,
            'predictions_csv_path': None
        })
        if response.status_code != 200:
            logger.error(f"Failed to update metadata on failure: {response.json()}")
        logger.error(f"Error during future predictions: {e}")
        raise    



def fetch_csv_from_s3(s3_path):
    """
    Fetches a CSV file from S3 with error handling for empty data.
    """
    try:
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'.")
        
        s3_parts = s3_path.replace("s3://", "").split("/")
        bucket_name = s3_parts[0]
        object_key = "/".join(s3_parts[1:])
        
        logger.info(f"Fetching CSV from S3 bucket: {bucket_name}, key: {object_key}")
        
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_S3_REGION_NAME'),
        )

        obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        csv_content = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))

        if df.shape[0] == 0:
            raise ValueError("Fetched CSV is empty. Cannot proceed with training.")

        logger.info(f"CSV fetched successfully. DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error fetching CSV from S3: {e}")
        raise

#this is for testing
# Define input data
# input_data = {
#     "file_url": "s3://testingfiles-pacx/predict_f7_2.csv",
#     "column_id": "product_id",
#     "user_id": "17236",
#     "chat_id": "7236389",
#     "ml_type": True
# }
# df = fetch_csv_from_s3(input_data["file_url"])
# # Drop unwanted columns
# df = df.drop(columns=["entity_id"], errors="ignore")
# # Call prediction function
# predictions_df = predict_future_timeseries(
#     input_df=df,
#     chat_id=input_data["chat_id"],
#     time_column="analysis_time",
#     column_id=input_data["column_id"],
#     target_column="target_within_30_days_after"
# )




#this is training
# =============================================================================
# 
# # Define input parameters
# input_data = {
#     "file_url": "s3://testingfiles-pacx/cell_8_2c46f7.csv",
#     "target_column": "target_within_30_days_after",
#     "user_id": "17236",
#     "chat_id": "7236389",
#     "column_id": "product_id",
#     "ml_type": True
# }
# 
# # Fetch the dataset from S3
# df = fetch_csv_from_s3(input_data["file_url"])
# df = df.drop(columns=["entity_id"], errors="ignore")
# # Call the training function
# train_pipeline_timeseries(
#     df=df,
#     target_column=input_data["target_column"],
#     user_id=input_data["user_id"],
#     chat_id=input_data["chat_id"],
#     column_id=input_data["column_id"],
#     time_column="analysis_time",  # Update if you have a time column
#     freq="weekly",
#     forecast_horizon="30 days",
#     use_time_series=input_data["ml_type"]
# )
# 
# =============================================================================
