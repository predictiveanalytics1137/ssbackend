


# v3


import pandas as pd
import numpy as np
import joblib
import io
import shap
import datetime
import requests
import time
import uuid

from sklearn.model_selection import train_test_split

from src.logging_config import get_logger
from src.s3_operations import upload_to_s3, load_from_s3
from src.feature_engineering import feature_engineering
from src.feature_selection import feature_selection
from src.helper import normalize_column_names
from src.data_preprocessing import handle_categorical_features
from src.finalization import finalize_and_evaluate_model
from src.hyperparameter_tuning import hyperparameter_tuning
from src.model_selection import train_test_model_selection
from src.utils import automatic_imputation
from src.outlier_handling import detect_and_handle_outliers_train, apply_outlier_bounds

logger = get_logger(__name__)



############################################################
# -- SHAP Explanation Helper
############################################################

def generate_shap_summary(model, X_sample):
    """
    Generates a SHAP summary plot for interpretability (optional).
    """
    try:
        explainer = shap.Explainer(model, X_sample)
        shap_values = explainer(X_sample)

        # Summaries (not displayed in console)
        shap.summary_plot(shap_values, X_sample, show=False)

        shap_importance = pd.DataFrame({
            "Feature": X_sample.columns,
            "Mean_SHAP_Value": np.abs(shap_values.values).mean(axis=0)
        }).sort_values(by="Mean_SHAP_Value", ascending=False)
        logger.info("SHAP summary plot generated. (Not displayed in console)")
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")


############################################################
# -- TRAIN PIPELINE (with optional time-series logic)
############################################################

def train_pipeline(
    df: pd.DataFrame,
    target_column: str,
    user_id: str,
    chat_id: str,
    column_id: str,
):
    """
    Complete machine learning pipeline, extended to handle time-series data if requested.
    """

    try:
        logger.info("Dataset received successfully.")
        logger.info(f"Dataset shape: {df.shape}")

        # Basic checks
        if df.shape[0] == 0:
            raise ValueError("DataFrame is empty. Cannot proceed with training.")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        if column_id not in df.columns:
            raise ValueError(f"ID column '{column_id}' not found in DataFrame.")
        if df[target_column].isnull().all():
            raise ValueError(f"Target column '{target_column}' is entirely null.")

        # Setup metadata logging
        training_id = chat_id
        start_time = datetime.datetime.now()
        start_timer = time.time()
        metadata_api_url = "http://127.0.0.1:8000/api/update_prediction_status/"
        entity_count = df.shape[0]


# =============================================================================
        # Create the initial metadata
        response = requests.post(metadata_api_url, json={
            'prediction_id': training_id,
            'chat_id': chat_id,
            'user_id': user_id,
            'status': 'inprogress',
            'entity_count': entity_count,
            'start_time': start_time.isoformat()
        })
        if response.status_code != 201:
            logger.error(f"Failed to create metadata: {response.json()}")
            raise RuntimeError("Initial metadata creation failed.")


        # Patch metadata
        requests.patch(f"{metadata_api_url}{training_id}/", json={'status': 'time_preprocessing_done'})

        logger.info("Using random train/test split.")
        id_data = df[column_id].copy()
        df_no_id = df.drop(columns=[column_id])

        X = df_no_id.drop(columns=[target_column])
        y = df_no_id[target_column]
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        test_ids = id_data.loc[X_test_raw.index]
        # Recombine to get train_df / test_df
        train_df = pd.concat([X_train_raw, y_train_raw], axis=1)
        train_df[column_id] = id_data.loc[train_df.index]
        test_df  = pd.concat([X_test_raw, y_test_raw], axis=1)
        test_df[column_id]  = test_ids

        # For clarity:
        logger.info(f"Train set shape: {train_df.shape}, Test set shape: {test_df.shape}")

        ##########################################
        # (5) MISSING VALUE IMPUTATION
        ##########################################
        logger.info("Starting missing value imputation on TRAIN set...")
        train_df_imputed, imputers = automatic_imputation(train_df, target_column=target_column)
        logger.info("Starting missing value imputation on TEST set...")
        test_df_imputed, _ = automatic_imputation(test_df, target_column=target_column, imputers=imputers)

        ##########################################
        # (6) OUTLIER DETECTION & HANDLING
        ##########################################
        logger.info("Computing outlier bounds and capping on TRAIN set (IQR-based).")
        train_df_outlier_fixed, outlier_bounds = detect_and_handle_outliers_train(
            train_df_imputed,
            factor=1.5
        )
        logger.info("Applying outlier bounds to TEST set.")
        test_df_outlier_fixed = apply_outlier_bounds(
            test_df_imputed,
            outlier_bounds
        )

        ##########################################
        # (7) CATEGORICAL ENCODING
        ##########################################
        logger.info("Handling categorical features on TRAIN set...")
        train_df_encoded, encoders = handle_categorical_features(
            df=train_df_outlier_fixed,
            target_column=target_column,
            id_column=column_id,
            cardinality_threshold=3
        )
        logger.info("Handling categorical features on TEST set...")
        test_df_encoded, _ = handle_categorical_features(
            df=test_df_outlier_fixed,
            target_column=target_column,
            encoders=encoders,
            id_column=column_id,
            cardinality_threshold=3,
            saved_column_names=train_df_encoded.columns.tolist()  # align columns
        )

        # Update metadata
        requests.patch(f"{metadata_api_url}{training_id}/", json={'status': 'future_engineering'})

        ##########################################
        # (8) FEATURE ENGINEERING (FeatureTools)
        ##########################################
        logger.info("Performing feature engineering on TRAIN set...")
        train_engineered, feature_defs, potential_binary_cols = feature_engineering(
            train_df_encoded,
            target_column=target_column,
            training=True,
            id_column=column_id
        )
        train_engineered = normalize_column_names(train_engineered)

        logger.info("Performing feature engineering on TEST set...")
        test_engineered = feature_engineering(
            df=test_df_encoded,
            training=False,
            feature_defs=feature_defs,
            id_column=column_id,
            fixed_binary_cols=potential_binary_cols  # Pass the binary columns from training
        )
        test_engineered = normalize_column_names(test_engineered)

        ##########################################
        # (9) FEATURE SELECTION
        ##########################################
        logger.info("Performing feature selection on TRAIN set...")
        train_selected, selected_features = feature_selection(
            train_engineered,
            target_column=target_column,
            task="regression",
            id_column=column_id
        )
        test_selected = test_engineered[selected_features]
        
        
        
        # After you get train_selected and test_selected from feature_selection:
        X_train = train_selected.drop(columns=[target_column], errors='ignore')
        y_train = train_selected[target_column]
        
        # ---- FIX: Drop Dealer_ID from X_train if present ----
        if column_id in X_train.columns:
            X_train = X_train.drop(columns=[column_id])
        
        X_test = test_selected.drop(columns=[target_column], errors='ignore')
        y_test = test_engineered[target_column] if target_column in test_engineered.columns else test_df[target_column]
        
        # ---- FIX: Drop Dealer_ID from X_test if present ----
        if column_id in X_test.columns:
            X_test = X_test.drop(columns=[column_id])
        


# =============================================================================
#         # Separate final X/y for train & test
#         X_train = train_selected.drop(columns=[target_column])
#         y_train = train_selected[target_column]
#         X_test  = test_selected.drop(columns=[target_column], errors="ignore") if target_column in test_selected.columns else test_selected
#         y_test  = test_engineered[target_column] if target_column in test_engineered.columns else test_df[target_column]
# =============================================================================





        # Also retrieve test IDs if needed
        # (They might exist in test_selected if id_column was re-added)
        test_ids = test_selected[column_id] if column_id in test_selected.columns else None

        ##########################################
        # (10) MODEL SELECTION (Baseline)
        ##########################################
        logger.info("Selecting best model from baseline pool...")
        # For model_selection, pass a combined DF with target
        # We'll create a dummy ID column for compatibility
        train_for_selection = train_selected.copy()
        train_for_selection['ID_dummy'] = np.arange(len(train_for_selection))

        best_model_name, _, _, _, _, _ = train_test_model_selection(
            train_for_selection,
            target_column=target_column,
            id_column=column_id,
            #id_column='ID_dummy',
            task='regression'
        )

        ##########################################
        # (11) HYPERPARAMETER TUNING
        ##########################################
        logger.info(f"Performing hyperparameter tuning for: {best_model_name}")
        best_model, best_params = hyperparameter_tuning(
            best_model_name=best_model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task='regression'
        )

        ##########################################
        # (12) FINALIZE & EVALUATE
        ##########################################
        logger.info("Finalizing and evaluating the model...")
        final_metrics = finalize_and_evaluate_model(
            best_model_class=best_model.__class__,
            best_params=best_params,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            user_id=user_id,
            chat_id=chat_id,
            test_ids=test_ids
        )

        # SHAP
        sample_size = min(100, X_train.shape[0])
        X_sample = X_train.sample(sample_size, random_state=42)
        generate_shap_summary(best_model, X_sample)

        ##########################################
        # (13) SAVE ARTIFACTS TO S3
        ##########################################
        logger.info("Uploading artifacts to S3...")
        bucket_name = "artifacts1137"
        prefix = f"ml-artifacts/{chat_id}/"

        def save_to_s3(obj, filename):
            with io.BytesIO() as f:
                joblib.dump(obj, f)
                f.seek(0)
                s3_key = f"{prefix}{filename}"
                upload_to_s3(f, bucket_name, s3_key)
                logger.info(f"Uploaded {filename} to s3://{bucket_name}/{s3_key}")

        # Save model and other pipeline objects
        save_to_s3(best_model, 'best_model.joblib')
        save_to_s3(imputers, 'imputers.joblib')
        save_to_s3(encoders, 'encoder.joblib')
        save_to_s3(feature_defs, 'feature_defs.joblib')
        save_to_s3(selected_features, 'selected_features.pkl')
        save_to_s3(outlier_bounds, 'outlier_bounds.pkl')

        # Save the columns used for encoding
        saved_column_names = train_df_encoded.columns.tolist()
        if target_column in saved_column_names:
            saved_column_names.remove(target_column)
        save_to_s3(saved_column_names, 'saved_column_names.pkl')

        logger.info("Artifacts uploaded successfully.")
        logger.info(f"Final metrics: {final_metrics}")

        # Metadata final update
        duration = time.time() - start_timer
        response = requests.patch(f"{metadata_api_url}{training_id}/", json={
            'status': 'training_completed',
            'duration': duration
        })
        if response.status_code != 200:
            logger.error(f"Failed to update metadata: {response.json()}")
            raise RuntimeError("Final metadata update failed.")

        return best_model, best_params

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise


############################################################
# PREDICT NEW DATA (TIME SERIES)
############################################################

def predict_new_data(new_data, bucket_name="artifacts1137", id_column=None, chat_id=None, user_id=None):
    """
    Loads trained model and artifacts to predict on new data.
    With data schema checks & improved handling.
    (You can optionally do time-series alignment if needed.)
    """
    try:
        logger.info(f"Loaded new data for prediction. Shape: {new_data.shape}")
        if new_data.shape[0] == 0:
            raise ValueError("New data is empty; cannot predict.")
        if not chat_id:
            raise ValueError("chat_id is required to locate the S3 folder for artifacts.")

        if id_column and id_column in new_data.columns:
            ids = new_data[id_column].copy()
            new_data = new_data.drop(columns=[id_column])
        else:
            ids = None

        # Generate unique prediction ID
        prediction_id = str(uuid.uuid4())[:8]

        # Log start
        start_time = datetime.datetime.now()
        metadata_api_url = "http://127.0.0.1:8000/api/update_prediction_status/"
        entity_count = new_data.shape[0]
        response = requests.post(metadata_api_url, json={
            'prediction_id': prediction_id,
            'chat_id': chat_id,
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

        prefix = f"ml-artifacts/{chat_id}/"
        model           = joblib.load(load_from_s3(bucket_name, prefix + "best_model.joblib"))
        imputers        = joblib.load(load_from_s3(bucket_name, prefix + "imputers.joblib"))
        encoders        = joblib.load(load_from_s3(bucket_name, prefix + "encoder.joblib"))
        selected_feats  = joblib.load(load_from_s3(bucket_name, prefix + "selected_features.pkl"))
        feature_defs    = joblib.load(load_from_s3(bucket_name, prefix + "feature_defs.joblib"))
        saved_col_names = joblib.load(load_from_s3(bucket_name, prefix + "saved_column_names.pkl"))
        outlier_bounds  = joblib.load(load_from_s3(bucket_name, prefix + "outlier_bounds.pkl"))

        # 1) Basic imputation
        new_data_imputed, _ = automatic_imputation(new_data, target_column=None, imputers=imputers)

        # 2) Outlier capping
        logger.info("Applying stored outlier bounds to new data...")
        new_data_outlier_fixed = apply_outlier_bounds(new_data_imputed, outlier_bounds)

        # 3) Categorical encoding
        new_data_encoded, _ = handle_categorical_features(
            df=new_data_outlier_fixed,
            encoders=encoders,
            saved_column_names=saved_col_names,
            id_column=None
        )

        # 4) Feature Engineering
        feature_matrix = feature_engineering(
            df=new_data_encoded,
            training=False,
            feature_defs=feature_defs
        )
        feature_matrix = normalize_column_names(feature_matrix)

        # 5) Feature Selection
        final_matrix = feature_matrix[selected_feats]

        # 6) Prediction
        logger.info("Generating predictions.")
        y_pred = model.predict(final_matrix)

        # Combine with IDs
        predictions_with_ids = pd.DataFrame({"Predicted": y_pred})
        if ids is not None:
            predictions_with_ids[id_column] = ids

        logger.info(f"Prediction done. Sample:\n{predictions_with_ids.head()}")

        # 7) Save predictions to S3
        logger.info("Saving predictions to S3...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        predictions_csv_key = f"ml-artifacts/{chat_id}/predictions/predictions_{timestamp}.csv"

        csv_buffer = io.StringIO()
        predictions_with_ids.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        bytes_buffer = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
        upload_to_s3(bytes_buffer, bucket_name, predictions_csv_key)

        # Final metadata
        duration = time.time() - start_timer
        response = requests.patch(f"{metadata_api_url}{prediction_id}/", json={
            'status': 'success',
            'duration': duration,
            'predictions_csv_path': predictions_csv_key
        })
        if response.status_code != 200:
            logger.error(f"Failed to update metadata: {response.json()}")
            raise RuntimeError("Final metadata update failed.")

        return predictions_with_ids

    except Exception as e:
        logger.error(f"Error during predict_new_data: {e}")
        raise
