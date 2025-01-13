import io
from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.s3_operations import get_s3_client, upload_to_s3
from src.data_preprocessing import handle_categorical_features
from src.finalization import finalize_and_evaluate_model
from src.model_selection import train_test_model_selection
from src.hyperparameter_tuning import hyperparameter_tuning
import joblib
from src.utils import automatic_imputation
from src.feature_engineering import feature_engineering
from src.logging_config import get_logger
import os
import featuretools as ft
from src.feature_selection import feature_selection


from src.logging_config import get_logger
from src.helper import normalize_column_names
logger = get_logger(__name__)

from src.utils import automatic_imputation  # Import from utils




# automation/src/pipeline.py
import io
import os
import joblib
import numpy as np
import pandas as pd
from src.s3_operations import get_s3_client, upload_to_s3
from src.data_preprocessing import handle_categorical_features
from src.finalization import finalize_and_evaluate_model
from src.model_selection import train_test_model_selection
from src.hyperparameter_tuning import hyperparameter_tuning
from src.utils import automatic_imputation
from src.feature_engineering import feature_engineering
from src.logging_config import get_logger
from src.helper import normalize_column_names
from src.feature_selection import feature_selection

logger = get_logger(__name__)





# def train_pipeline(df, target_column, user_id, chat_id, column_id):
#     """
#     Complete machine learning pipeline with direct S3 upload for artifacts.
#     Includes user_id and chat_id to maintain uniqueness.
#     """
#     try:
#         logger.info("Dataset received successfully.")
#         logger.info(f"Dataset shape: {df.shape}")
#         logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
#         logger.info(f"Entity (ID) column '{column_id}' will be retained.")
#         # Save the ID column for later use
#         id_data = df[column_id].copy()

#         # Handle Missing Values, Encoding, Feature Engineering
#         logger.info("Starting missing value imputation...")
#         df, imputers = automatic_imputation(df, target_column=target_column)
#         logger.info("Handling categorical features...")
#         #df_encoded, encoder = handle_categorical_features(df, cardinality_threshold=3)
#         df_encoded, encoder = handle_categorical_features(
#             df, 
#             id_column=column_id, 
#             cardinality_threshold=3
#         )
#         saved_column_names = df_encoded.columns.tolist()

#         logger.info("Performing feature engineering...")
#         df_engineered, feature_defs = feature_engineering(df_encoded, target_column=target_column, training=True, id_column=column_id)
#         df_engineered = normalize_column_names(df_engineered)
#         logger.info("Feature engineering complete.")
#         # Add back the ID column (unaltered)
#         #df_engineered[column_id] = id_data

#         # Perform feature selection
#         logger.info("Performing feature selection...")
#         df_selected, selected_features = feature_selection(df_engineered, target_column=target_column, task="regression", id_column=column_id)
#         print(selected_features)
#         print("selected_features")

#         # Split data
#         logger.info("Splitting data into training and testing sets...")
#         best_model_name, X_train, y_train, X_test, y_test, test_ids = train_test_model_selection(
#             df_selected, target_column=target_column, task='regression', id_column=column_id
#         )

#         # Separate ID column for the test set
#         #test_ids = X_test[column_id].copy()
#         #X_train.drop(columns=[column_id], inplace=True)
#         #X_test.drop(columns=[column_id], inplace=True)

#         # Hyperparameter Tuning
#         logger.info(f"Performing hyperparameter tuning for the best model: {best_model_name}...")
#         best_model, best_params = hyperparameter_tuning(
#             best_model_name=best_model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, task='regression'
#         )

#         # Finalize and Save Model (now pass user_id and chat_id)
#         logger.info("Finalizing and evaluating the model...")
#         final_metrics = finalize_and_evaluate_model(
#             best_model.__class__, best_params, X_train, y_train, X_test, y_test,
#             # user_id=user_id, chat_id=chat_id  # Pass them here
#             user_id=user_id, chat_id=chat_id, test_ids=test_ids
#         )

#         # Upload artifacts to S3
#         logger.info("Uploading artifacts directly to S3...")
#         bucket_name = "artifacts1137"
#         prefix = "ml-artifacts/"

#         def save_to_s3(obj, filename):
#             with io.BytesIO() as f:
#                 joblib.dump(obj, f)
#                 f.seek(0)
#                 s3_key = f"{prefix}{filename}"
#                 upload_to_s3(f, bucket_name, s3_key)
#                 logger.info(f"Uploaded {filename} to S3 as {s3_key}")

#         save_to_s3(best_model, 'best_model.joblib')
#         save_to_s3(imputers, 'imputers.joblib')
#         save_to_s3(encoder, 'encoder.joblib')
#         save_to_s3(feature_defs, 'feature_defs.joblib')
#         save_to_s3(selected_features, 'selected_features.pkl')
#         if target_column in saved_column_names:
#             saved_column_names.remove(target_column)
#             save_to_s3(saved_column_names, 'saved_column_names.pkl')
#             logger.info("Categorical feature encoding complete.")

#         logger.info("Artifacts uploaded to S3 successfully.")
#         print(final_metrics)
#         return best_model, best_params

#     except Exception as e:
#         logger.error(f"Error during pipeline execution: {e}")
#         raise




# def load_from_s3(bucket_name, s3_key):
#     """
#     Loads a file from S3 into memory using a BytesIO buffer.
#     """
#     s3 = get_s3_client()
#     try:
#         # Check if the file exists
#         s3.head_object(Bucket=bucket_name, Key=s3_key)
#         logger.info(f"File {s3_key} exists in bucket {bucket_name}.")

#         # Download the file
#         buffer = io.BytesIO()
#         s3.download_fileobj(bucket_name, s3_key, buffer)
#         buffer.seek(0)  # Reset buffer to the beginning
#         return buffer
#     except s3.exceptions.ClientError as e:
#         if e.response['Error']['Code'] == "404":
#             logger.error(f"The key {s3_key} does not exist in bucket {bucket_name}.")
#             raise FileNotFoundError(f"S3 Key not found: {s3_key}")
#         else:
#             logger.error(f"Error accessing {s3_key} in bucket {bucket_name}: {e}")
#             raise








# def predict_new_data(new_data, bucket_name="artifacts1137",id_column=None):
#     """
#     Loads the trained model and saved preprocessing artifacts to predict on new data.
    
#     Parameters:
#     - new_csv_path: string, path to the new CSV file for prediction.
#     - feature_defs_path: string, path to the saved feature definitions.
    
#     Returns:
#     - predictions: array of predictions for the target column in the original scale.
#     """
#     try:
#         logger.info(f"Dataset loaded successfully. Shape: {new_data.shape}")
#         logger.info(f"Entity (ID) column '{id_column}' will be retained.")


#         # S3 keys for artifacts
#         prefix = "ml-artifacts/"
#         model_key = f"{prefix}best_model.joblib"
#         imputers_key = f"{prefix}imputers.joblib"
#         encoder_key = f"{prefix}encoder.joblib"
#         selected_features_key = f"{prefix}selected_features.pkl"
#         feature_defs_key = f"{prefix}feature_defs.joblib"
#         saved_column_names_key = f"{prefix}saved_column_names.pkl"

#         # Load artifacts directly from S3
#         logger.info("Loading trained model and artifacts from S3...")
#         model = joblib.load(load_from_s3(bucket_name, model_key))
#         imputers = joblib.load(load_from_s3(bucket_name, imputers_key))
#         encoder = joblib.load(load_from_s3(bucket_name, encoder_key))
#         selected_features = joblib.load(load_from_s3(bucket_name, selected_features_key))
#         feature_defs = joblib.load(load_from_s3(bucket_name, feature_defs_key))
#         saved_column_names = joblib.load(load_from_s3(bucket_name, saved_column_names_key))
#         logger.info("Model and artifacts loaded successfully.")
#         # import pdb;pdb.set_trace()
#         print("getting in to ifs")
        
#         if id_column:
#             ids = new_data[id_column].copy()
#             print(ids)
#             new_data = new_data.drop(columns=[id_column])  # Exclude ID column from processing
#             print(new_data.columns)
#             logger.info(f"Excluded ID column '{id_column}' from processing.")



#         # Load new data
#         # logger.info(f"Loading and preprocessing new data from {new_csv_path}...")
#         # new_data = pd.read_csv(new_csv_path)

#         # Apply imputation and encoding to new data
#         logger.info("Applying imputation and encoding to new data...")
#         new_data_processed, _ = automatic_imputation(new_data, target_column=None, imputers=imputers)
#         new_data_processed, _ = handle_categorical_features(new_data_processed, cardinality_threshold=10, encoders=encoder, saved_column_names=saved_column_names,id_column = id_column)
#         logger.info("Imputation and encoding complete.")

#         # Apply feature engineering using saved feature definitions
#         logger.info("Applying feature engineering using saved feature definitions...")
#         #feature_defs = joblib.load("featurengineering_feature_defs.pkl")
#         # feature_matrix, _ = feature_engineering(new_data_processed, training=False)
#         feature_matrix = feature_engineering(new_data_processed, training=False, feature_defs=feature_defs)
#         feature_matrix = normalize_column_names(feature_matrix)

#         logger.info("Feature engineering complete.")

#         logger.info("Applying feature selection...")
#         selected_feature_matrix = feature_matrix[selected_features]  # Align with selected features from training
#         logger.info(f"Feature matrix reduced to selected features: {selected_feature_matrix.shape}")


#         # Predict on the processed new data
#         logger.info("Making predictions...")
#         predictions_scaled = model.predict(selected_feature_matrix)
        
        
#         # Combine predictions with IDs
#         predictions_with_ids = pd.DataFrame({
#             id_column: ids,  # Include the original entity ID
#             "Predicted": predictions_scaled
#         })
#         print(predictions_with_ids)
#         logger.info("Predictions with IDs:")
#         logger.info(f"\n{predictions_with_ids.head()}")

#         # Check the shape before reshaping
#         logger.info(f"Predictions before reshape: {predictions_scaled.shape}")
#         #predictions_scaled = predictions_scaled.reshape(-1, 1)
#         logger.info(f"Predictions after reshape: {predictions_scaled.shape}")

#         # Inverse transform predictions to original scale
#         #predictions = target_scaler.inverse_transform(predictions_scaled).flatten()
#         #logger.info(f"Predictions after inverse transformation: {predictions[:10]}")

#         logger.info("Predictions complete.")
#         #return predictions
#         return predictions_scaled

#     except Exception as e:
#         logger.error(f"Error during prediction: {e}")
#         raise



# v2


# src/pipeline.py

import pandas as pd
import numpy as np
import joblib
import io
import shap  # ADDED for interpretability
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
from sklearn.model_selection import train_test_split
import datetime
import requests
import time
import uuid


logger = get_logger(__name__)

# NEW:
from src.outlier_handling import detect_and_handle_outliers_train, apply_outlier_bounds


# -----------------------------------------------------------------------------
# ADDED HELPER FUNCTION: Generate and log SHAP summary after model is trained
# -----------------------------------------------------------------------------
def generate_shap_summary(model, X_sample):
    """
    Generates a SHAP summary plot for interpretability.
    """
    try:
        explainer = shap.Explainer(model, X_sample)
        print(explainer)
        shap_values = explainer(X_sample)

        # Generate a SHAP summary plot (matplotlib figure)
        shap.summary_plot(shap_values, X_sample, show=False)
        shap_importance = pd.DataFrame({
            "Feature": X_sample.columns,
            "Mean_SHAP_Value": np.abs(shap_values.values).mean(axis=0)
        }).sort_values(by="Mean_SHAP_Value", ascending=False)
        logger.info("SHAP summary plot generated (not displayed in console).")
    except Exception as e:
        logger.error(f"Error generating SHAP explanations: {e}")


def train_pipeline(df, target_column, user_id, chat_id, column_id):
    """
    Complete machine learning pipeline with direct S3 upload for artifacts.
    """
    try:
        logger.info("Dataset received successfully.")
        logger.info(f"Dataset shape: {df.shape}")

        # 1. --------------------------
        #    DATA SCHEMA & NULL CHECKS
        #  ---------------------------
        if df.shape[0] == 0:
            raise ValueError("DataFrame is empty. Cannot proceed with training.")

        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

        if column_id not in df.columns:
            raise ValueError(f"ID column '{column_id}' not found in DataFrame.")

        if df[target_column].isnull().all():
            raise ValueError(f"Target column '{target_column}' is entirely null.")

        logger.info(f"Entity (ID) column '{column_id}' will be retained.")

        # 2. --------------------------
        #    Train-Test Split (EARLY)
        #    to avoid data leakage with
        #    target-based encoders
        #  ---------------------------
        # We do an early split here if we want to avoid target-based leakage
        # Then we apply transformations only on X_train / X_test separately
        # However, the existing code calls handle_categorical_features before splitting,
        # which can cause data leakage if using TargetEncoder. Let's fix that:

        # We'll do a minimal check here: we drop the ID column, do the split, then do
        # automatic imputation, categorical handling, feature engineering, etc. on each set.

        id_data = df[column_id].copy()
        

        # Keep a copy so we can do transformations on training set first
        # Temporarily drop ID so it won't be accidentally encoded
        df_no_id = df.drop(columns=[column_id])

        # Do an early split:
        X = df_no_id.drop(columns=[target_column])
        y = df_no_id[target_column]

        # Basic 80-20 split
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        test_ids = id_data.loc[X_test_raw.index]


        # 3. --------------------------------
        #    Imputation (Train & Test separately)
        # --------------------------------
        train_df = pd.concat([X_train_raw, y_train_raw], axis=1)
        test_df = pd.concat([X_test_raw, y_test_raw], axis=1)
        

        logger.info("Starting missing value imputation on TRAIN set...")
        train_df_imputed, imputers = automatic_imputation(train_df, target_column=target_column)
        logger.info("Starting missing value imputation on TEST set...")
        test_df_imputed, _ = automatic_imputation(test_df, target_column=target_column, imputers=imputers)
        
        # 4. --------------------------------
        #    OUTLIER DETECTION and handling (Train & Test separately)
        # --------------------------------
        
        # OUTLIER DETECTION on TRAIN
        logger.info("Computing outlier bounds and capping on TRAIN set (IQR-based).")
        train_df_outlier_fixed, outlier_bounds = detect_and_handle_outliers_train(
            train_df_imputed,
            factor=1.5
        )

        # Apply the same outlier bounds to TEST set
        logger.info("Applying outlier bounds to TEST set.")
        test_df_outlier_fixed = apply_outlier_bounds(
            test_df_imputed,
            outlier_bounds
        )

        # 5. ----------------------------------
        #    Handle Categorical (Train & Test)
        #    Using target_column for TEs only
        # -------------------------------------
        logger.info("Handling categorical features on TRAIN set...")
        train_df_encoded, encoders = handle_categorical_features(
            train_df_outlier_fixed,
            target_column=target_column,     # safe because we're doing training
            id_column=None,                  # we've already dropped ID here
            cardinality_threshold=3
        )

        logger.info("Handling categorical features on TEST set...")
        test_df_encoded, _ = handle_categorical_features(
            test_df_outlier_fixed,
            target_column=target_column,     # we want to transform using the same TEs
            encoders=encoders,
            id_column=None,
            cardinality_threshold=3,
            saved_column_names=train_df_encoded.columns.tolist()
        )

        # 6. --------------------------------
        #    Feature Engineering
        # ------------------------------------
        logger.info("Performing feature engineering on TRAIN set...")
        train_engineered, feature_defs = feature_engineering(
            train_df_encoded,
            target_column=target_column,
            training=True,
            id_column=None
        )
        train_engineered = normalize_column_names(train_engineered)

        logger.info("Performing feature engineering on TEST set...")
        test_engineered = feature_engineering(
            test_df_encoded,
            training=False,
            feature_defs=feature_defs,
            id_column=None
        )
        test_engineered = normalize_column_names(test_engineered)

        # 7. --------------------------------
        #    Feature Selection
        # ------------------------------------
        logger.info("Performing feature selection on TRAIN set...")
        train_selected, selected_features = feature_selection(
            train_engineered,
            target_column=target_column,
            task="regression",
            id_column=None
        )

        # Align test set to the same features
        test_selected = test_engineered[selected_features]

        # Separate final X/y for training and testing
        X_train = train_selected.drop(columns=[target_column])
        y_train = train_selected[target_column]
        X_test = test_selected
        #y_test = test_engineered[target_column]
        y_test = y_test_raw

        # 8. --------------------------------
        #    Model Selection (just to pick best model)
        # ------------------------------------
        logger.info("Selecting best model from baseline pool...")
        # We'll create a merged DF for model_selection
        # because model_selection logic expects a single DataFrame with target included
        # plus an id column if needed. We'll just pass None if we can't handle ID there.
        train_for_selection = train_selected.copy()
        train_for_selection['ID_dummy'] = np.arange(len(train_for_selection))
        best_model_name, _, _, _, _, _ = train_test_model_selection(
            train_for_selection,
            target_column=target_column,
            id_column='ID_dummy',  # dummy
            task='regression'
        )

        # 9. --------------------------------
        #    Hyperparameter Tuning
        # ------------------------------------
        logger.info(f"Performing hyperparameter tuning for the best model: {best_model_name}...")
        best_model, best_params = hyperparameter_tuning(
            best_model_name=best_model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            task='regression'
        )

        # 10. --------------------------------
        #    Finalize, Evaluate, & SHAP
        # ------------------------------------
        logger.info("Finalizing and evaluating the model...")
        final_metrics = finalize_and_evaluate_model(
            best_model.__class__,
            best_params,
            X_train, y_train,
            X_test, y_test,
            user_id=user_id,
            chat_id=chat_id,
            test_ids = test_ids
            #test_ids=None  # ID handling not shown here
        )

        # Generate SHAP explanations
        # We'll pick a small random sample of X_train for speed
        sample_size = min(100, X_train.shape[0])
        X_sample = X_train.sample(sample_size, random_state=42)
        generate_shap_summary(best_model, X_sample)
        #print(generate_shap_summary(best_model, X_sample))

        # 11. -------------------------------
        #     Save Artifacts to S3
        # ------------------------------------
        logger.info("Uploading artifacts directly to S3...")
        bucket_name = "artifacts1137"
        # prefix = "ml-artifacts/"
        prefix = f"ml-artifacts/{chat_id}/"

        def save_to_s3(obj, filename):
            with io.BytesIO() as f:
                joblib.dump(obj, f)
                f.seek(0)
                s3_key = f"{prefix}{filename}"
                upload_to_s3(f, bucket_name, s3_key)
                logger.info(f"Uploaded {filename} to S3 as {s3_key}")

        # Save all relevant artifacts
        save_to_s3(best_model, 'best_model.joblib')
        save_to_s3(imputers, 'imputers.joblib')
        save_to_s3(encoders, 'encoder.joblib')
        save_to_s3(feature_defs, 'feature_defs.joblib')
        save_to_s3(selected_features, 'selected_features.pkl')
        
        # NEW: Save the outlier_bounds for usage at prediction time
        save_to_s3(outlier_bounds, 'outlier_bounds.pkl')

        # We can store the train_df_encoded columns if needed for alignment
        saved_column_names = train_df_encoded.columns.tolist()
        if target_column in saved_column_names:
            saved_column_names.remove(target_column)
        save_to_s3(saved_column_names, 'saved_column_names.pkl')

        logger.info("Artifacts uploaded to S3 successfully.")
        logger.info(f"Final metrics: {final_metrics}")
        return best_model, best_params

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise



# def load_from_s3(bucket_name, s3_key):
#     """
#     Loads a file from S3 into memory using a BytesIO buffer.
#     """
#     s3 = get_s3_client()
#     try:
#         # Check if the file exists
#         s3.head_object(Bucket=bucket_name, Key=s3_key)
#         logger.info(f"File {s3_key} exists in bucket {bucket_name}.")

#         # Download the file
#         buffer = io.BytesIO()
#         s3.download_fileobj(bucket_name, s3_key, buffer)
#         buffer.seek(0)  # Reset buffer to the beginning
#         return buffer
#     except s3.exceptions.ClientError as e:
#         if e.response['Error']['Code'] == "404":
#             logger.error(f"The key {s3_key} does not exist in bucket {bucket_name}.")
#             raise FileNotFoundError(f"S3 Key not found: {s3_key}")
#         else:
#             logger.error(f"Error accessing {s3_key} in bucket {bucket_name}: {e}")
#             raise





def predict_new_data(new_data, bucket_name="artifacts1137", id_column=None,chat_id = None,user_id=None):
    """
    Loads trained model and artifacts to predict on new data.
    With data schema checks & improved handling.
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
            
        # Generate a unique prediction ID
        # prediction_id = str(uuid.uuid4())
        prediction_id = str(uuid.uuid4())[:8]
        
        # Step 1: Log the start time and initial status
        
        
        start_time = datetime.datetime.now()
        metadata_api_url = "http://127.0.0.1:8000/api/update_prediction_status/"
        entity_count = new_data.shape[0]
        
        # Create the initial metadata
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

            
       

            
        

# =============================================================================
#         requests.post(metadata_api_url, json={
#             'chat_id': chat_id,
#             'status': 'inprogress',
#             'entity_count': entity_count,
#             'user_id':user_id
#         })
# =============================================================================


        


        # Prediction process starts
        logger.info("Starting prediction process...")
        start_timer = time.time()

        prefix = f"ml-artifacts/{chat_id}/"
        model = joblib.load(load_from_s3(bucket_name, prefix + "best_model.joblib"))
        imputers = joblib.load(load_from_s3(bucket_name, prefix + "imputers.joblib"))
        encoders = joblib.load(load_from_s3(bucket_name, prefix + "encoder.joblib"))
        selected_features = joblib.load(load_from_s3(bucket_name, prefix + "selected_features.pkl"))
        feature_defs = joblib.load(load_from_s3(bucket_name, prefix + "feature_defs.joblib"))
        saved_column_names = joblib.load(load_from_s3(bucket_name, prefix + "saved_column_names.pkl"))
        # NEW: outlier_bounds
        outlier_bounds    = joblib.load(load_from_s3(bucket_name, prefix + "outlier_bounds.pkl"))

        # Imputation
        new_data_imputed, _ = automatic_imputation(new_data, target_column=None, imputers=imputers)

        
        # 2) Apply Outlier Bounds from Training
        logger.info("Applying stored outlier bounds to new data...")
        new_data_outlier_fixed = apply_outlier_bounds(new_data_imputed, outlier_bounds)

        # Encoding
        new_data_encoded, _ = handle_categorical_features(
            new_data_outlier_fixed,
            cardinality_threshold=10,
            encoders=encoders,
            saved_column_names=saved_column_names,
            id_column=None
        )

        # Feature Engineering
        feature_matrix = feature_engineering(
            new_data_encoded,
            training=False,
            feature_defs=feature_defs
        )
        feature_matrix = normalize_column_names(feature_matrix)

        # Feature Selection
        final_matrix = feature_matrix[selected_features]

        logger.info("Generating predictions.")
        y_pred = model.predict(final_matrix)

        predictions_with_ids = pd.DataFrame({
            "Predicted": y_pred
        })
        if ids is not None:
            predictions_with_ids[id_column] = ids

        logger.info(f"Prediction done. Sample:\n{predictions_with_ids.head()}")
        
        
        
        
        
        
        # Save predictions to S3
        logger.info("Saving predictions to S3...")
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")  # Unique timestamp
        predictions_csv_key = f"ml-artifacts/{chat_id}/predictions/predictions_{timestamp}.csv"

        # Convert DataFrame to CSV and save to S3
        csv_buffer = io.StringIO()
        predictions_with_ids.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)  # Reset buffer to the beginning

        # Convert to BytesIO for upload
        bytes_buffer = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
        upload_to_s3(bytes_buffer, bucket_name, predictions_csv_key)
        
        
        
        
        # Step 3: Calculate duration and log final status
        duration = time.time() - start_timer
        
        
        
        
        
        
        metadata_api_url = f"http://127.0.0.1:8000/api/update_prediction_status/{prediction_id}/"

        # Send PATCH request to update metadata
        response = requests.patch(metadata_api_url, json={
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