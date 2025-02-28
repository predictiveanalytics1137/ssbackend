

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
from automation.src.outlier_handling import apply_outlier_bounds, detect_and_handle_outliers_train
from src.s3_operations import get_s3_client, upload_to_s3
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

# -------------------------------
# Time Series Utilities
# -------------------------------
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

# -------------------------------
# Static Feature Aggregation
# -------------------------------
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

# -------------------------------
# Main Training Pipeline
# -------------------------------
def train_pipeline_timeseries(
    df: pd.DataFrame,
    target_column: str,
    user_id: str,
    chat_id: str,
    column_id: str,
    time_column: str = None,
    freq: str = "weekly",
    forecast_horizon: str = "30 days",
    use_time_series: bool = False
):
    try:
        # import pdb; pdb.set_trace()
        logger.info("Dataset received successfully.")
        logger.info(f"Dataset shape: {df.shape}")
        if df.shape[0] == 0:
            raise ValueError("DataFrame is empty.")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
        if column_id not in df.columns:
            raise ValueError(f"ID column '{column_id}' not found.")

        training_id = chat_id
        start_time = time.time()
        metadata_api_url = "http://127.0.0.1:8000/api/update_prediction_status/"

        # 1) Preprocess Entire Dataset
        if use_time_series and time_column:
            logger.info("Performing time-series preprocessing...")
            if time_column not in df.columns:
                raise ValueError(f"Time column '{time_column}' not found.")
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
            df = create_time_based_features(df, time_column)
            columns_to_lag = [col for col in df.columns if col not in [time_column, column_id] and 'lag' not in col]
            df = create_lag_features(df, column_id, columns_to_lag, time_column)

        logger.info("Performing missing value imputation...")
        df_imputed, imputers = automatic_imputation(df, target_column=target_column)
        logger.info("Handling outliers...")
        df_outlier_fixed, outlier_bounds = detect_and_handle_outliers_train(df_imputed, factor=1.5)
        logger.info("Encoding categorical features...")
        df_encoded, encoders = handle_categorical_features(
            df=df_outlier_fixed,
            target_column=target_column,
            id_column=column_id,
            cardinality_threshold=3
        )

        # Update columns_to_lag to include encoded features
        encoded_columns = [col for col in df_encoded.columns if col not in [time_column, column_id, target_column] and 'lag' not in col]
        df_encoded = create_lag_features(df_encoded, column_id, encoded_columns, time_column)

        # 2) Single Training on Entire Dataset
        horizon_delta = parse_horizon(forecast_horizon)
        global_max_date = df[time_column].max()
        train_cutoff = global_max_date - horizon_delta
        train_data = df_encoded[df_encoded[time_column] <= train_cutoff].copy()
        logger.info(f"Training set shape: {train_data.shape}")

        logger.info("Performing feature engineering on training data...")
        train_data = train_data.rename(columns={time_column: "analysis_time"})
        df_engineered, feature_defs = feature_engineering_timeseries(
            train_data,
            target_column=target_column,
            id_column=column_id,
            time_column="analysis_time",
            training=True
        )
        df_engineered = normalize_column_names(df_engineered)

        logger.info("Performing feature selection...")
        df_selected, selected_features = feature_selection(
            df_engineered,
            target_column=target_column,
            task="regression",
            id_column=column_id
        )
        X_train = df_selected.drop(columns=[target_column, column_id], errors='ignore')
        y_train = df_selected[target_column]

        logger.info("Selecting best model...")
        best_model_name, _, _, _, _, _ = train_test_model_selection_timeseries(
            df_selected,
            target_column=target_column,
            id_column=column_id,
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

        # Train the final model directly in the pipeline
        logger.info("Training final model with best hyperparameters...")
        final_model = best_model.__class__(**best_params)
        final_model.fit(X_train, y_train)


        
        # logger.info("Finalizing and evaluating the model...")
        # best_model, final_metrics = finalize_and_evaluate_model(
        #     best_model_class=best_model.__class__,
        #     best_params=best_params,
        #     X_train=X_train,
        #     y_train=y_train,
        #     user_id=user_id,
        #     chat_id=chat_id
        # )

        # 3) Validation Predictions Per Product
        logger.info("Generating validation predictions...")
        products = df[column_id].unique()
        validation_metrics = {}
        predictions = []

        for product in products:
            product_data = df_encoded[df_encoded[column_id] == product].copy()
            max_date = product_data[time_column].max()
            val_data = product_data[product_data[time_column] > max_date - horizon_delta].copy()

            if len(val_data) == 0:
                logger.info(f"Skipping validation for {product} due to no validation data")
                continue

            val_data = create_lag_features(val_data, column_id, encoded_columns, time_column)
            logger.info(f"Validation set shape for {product}: {val_data.shape}")

            val_engineered = feature_engineering_timeseries(
                val_data.rename(columns={time_column: "analysis_time"}),
                target_column=target_column,
                id_column=column_id,
                time_column="analysis_time",
                training=False,
                feature_defs=feature_defs
            )
            logger.info(f"Validation set shape for {product}: {val_engineered.shape}")
            val_engineered = normalize_column_names(val_engineered)
            X_val = val_engineered.reindex(columns=X_train.columns, fill_value=0)
            val_predictions = best_model.predict(X_val)

            val_df = pd.DataFrame({
                'analysis_time': val_data[time_column],
                column_id: product,
                'actual': val_data[target_column],
                'predicted': val_predictions
            })
            aggregated_preds = val_df.groupby('analysis_time').agg({
                column_id: 'first',
                'actual': 'mean',
                'predicted': 'mean'
            }).reset_index()
            predictions.append(aggregated_preds)
            logger.info(f"Validation predictions shape for {product}: {aggregated_preds.shape}")
            
            

            # Calculate validation metrics for this product
            mse = mean_squared_error(aggregated_preds['actual'], aggregated_preds['predicted'])
            mae = mean_absolute_error(aggregated_preds['actual'], aggregated_preds['predicted'])
            r2 = r2_score(aggregated_preds['actual'], aggregated_preds['predicted']) if len(aggregated_preds['actual']) > 1 else 0.0
            validation_metrics[product] = {'RMSE': mse, 'MAE': mae, 'R2': r2}
            logger.info(f"Product {product} - Validation MSE: {mse:.2f}")
        predictions_df = pd.concat(predictions, ignore_index=True)
        logger.info("Validation predictions shape: " + str(predictions_df.shape))
        logger.info("Validation predictions sample:\n" + str(predictions_df.head()))
            

        print(predictions_df)
        # import pdb; pdb.set_trace()

        # # 4) Future Predictions
        # static_cols = [col for col in df_encoded.columns if col not in [time_column, column_id, target_column] and 'lag' not in col]
        # entity_static = aggregate_static_features(df_encoded, column_id, static_cols)
        # forecast_list = []

        # for product in products:
        #     product_data = df_encoded[df_encoded[column_id] == product].copy()
        #     max_date = product_data[time_column].max()
        #     forecast_start = next_aligned(max_date, freq)
        #     forecast_dates = generate_forecast_dates(forecast_start, horizon_delta, freq)

        #     for future_date in forecast_dates:
        #         future_df = pd.DataFrame({
        #             column_id: [product],
        #             time_column: [future_date]
        #         })
        #         future_df = future_df.merge(entity_static, on=column_id, how='left')
        #         future_df = create_time_based_features(future_df, time_column)

        #         # Add lag features using last known values
        #         last_train = product_data.sort_values(time_column).tail(1)
        #         for col in encoded_columns:
        #             if col in last_train.columns:
        #                 future_df[f"{col}_lag_1"] = last_train[col].values[0]
        #             future_df[f"{col}_lag_2"] = 0
        #             future_df[f"{col}_lag_3"] = 0
                    
        #         # Apply encoding to future_df using saved encoders
        #         future_encoded, _ = handle_categorical_features(
        #             df=future_df,
        #             target_column=target_column,
        #             id_column=column_id,
        #             encoders=encoders,
        #             cardinality_threshold=3,
        #             saved_column_names=encoded_columns + [time_column, column_id]
        #         )
        #         # Ensure time_column is preserved
        #         if time_column not in future_encoded.columns:
        #             future_encoded[time_column] = future_df[time_column]
        #         if column_id not in future_encoded.columns:
        #             future_encoded[column_id] = future_df[column_id]
                
        #         future_encoded = create_lag_features(future_encoded, column_id, encoded_columns, time_column)

        #         future_engineered = feature_engineering_timeseries(
        #             future_encoded.rename(columns={time_column: "analysis_time"}),
        #             target_column=target_column,
        #             id_column=column_id,
        #             time_column="analysis_time",
        #             training=False,
        #             feature_defs=feature_defs
        #         )
        #         future_engineered = normalize_column_names(future_engineered)
        #         X_future = future_engineered.reindex(columns=X_train.columns, fill_value=0)
        #         future_pred = best_model.predict(X_future)[0]
        #         forecast_list.append({
        #             'analysis_time': future_date,
        #             column_id: product,
        #             'predicted': future_pred
        #         })

        # 5) Combine and Format Output
        validation_preds = pd.concat(predictions) if predictions else pd.DataFrame()
        # future_preds = pd.DataFrame(forecast_list)
        # forecast_output = pd.concat([
        #     validation_preds.rename(columns={'analysis_time': 'Marker', 'actual': 'Actual', 'predicted': 'Prediction'}),
        #     future_preds.rename(columns={'analysis_time': 'Marker', 'predicted': 'Prediction'})
        # ])
        # forecast_output['pecan_id'] = range(864, 864 + len(forecast_output))
        # forecast_output['pecan_model_id'] = best_model_name
        # forecast_output['sampled_date'] = forecast_output['Marker']
        # forecast_output = forecast_output[['pecan_id', column_id, 'sampled_date', 'Marker', 'Prediction', 'pecan_model_id', 'Actual']]



        # 6) Finalize and Evaluate Model (using validation metrics)
        # logger.info("Finalizing and evaluating the model with validation metrics...")
        # final_model, final_metrics = finalize_and_evaluate_model_timeseries(
        #     best_model_class=final_model.__class__,
        #     best_params=best_params,
        #     X_train=X_train,
        #     y_train=y_train,
        #     validation_metrics=validation_metrics,  # Pass validation metrics dictionary
        #     user_id=user_id,
        #     chat_id=chat_id
        # )
        # In train_pipeline_timeseries (replace the finalize_and_evaluate_model call)
        # 6) Finalize and Evaluate Model (using validation predictions)
        logger.info("Finalizing and evaluating the model with validation predictions...")
        # import pdb; pdb.set_trace()
        final_model, final_metrics = finalize_and_evaluate_model_timeseries(
            final_model=final_model,  # Pass the trained model from the pipeline
            X_train=X_train,
            predictions_df=predictions_df,  # Pass the concatenated validation predictions
            user_id=user_id,
            chat_id=chat_id,
            best_params=best_params
        )

        # 7) Save Artifacts
        logger.info("Saving artifacts to S3...")
        bucket_name = "artifacts1137"
        prefix = f"ml-artifacts/{chat_id}/"
        def save_to_s3(obj, filename):
            with io.BytesIO() as f:
                joblib.dump(obj, f)
                f.seek(0)
                upload_to_s3(f, bucket_name, f"{prefix}{filename}")
        save_to_s3(best_model, 'best_model.joblib')
        save_to_s3(imputers, 'imputers.joblib')
        save_to_s3(encoders, 'encoder.joblib')
        save_to_s3(feature_defs, 'feature_defs.joblib')
        save_to_s3(selected_features, 'selected_features.pkl')
        save_to_s3(outlier_bounds, 'outlier_bounds.pkl')
        save_to_s3(X_train.columns.tolist(), 'saved_column_names.pkl')
        # save_to_s3(forecast_output, 'forecast_output.pkl')

        # 7) Final Metadata Update and Return
        duration = time.time() - start_time
        requests.patch(f"{metadata_api_url}{training_id}/", json={
            'status': 'training_completed',
            'duration': duration
        })
        # logger.info("Forecast output shape: " + str(forecast_output.shape))
        # logger.info("Sample forecast:\n" + str(forecast_output.head()))
        return best_model, best_params

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise
        