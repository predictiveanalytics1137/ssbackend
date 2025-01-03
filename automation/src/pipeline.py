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
#         df_encoded, encoder = handle_categorical_features(df, cardinality_threshold=3)
#         saved_column_names = df_encoded.columns.tolist()

#         logger.info("Performing feature engineering...")
#         df_engineered, feature_defs = feature_engineering(df_encoded, target_column=target_column, training=True)
#         df_engineered = normalize_column_names(df_engineered)
#         logger.info("Feature engineering complete.")
#         # Add back the ID column (unaltered)
#         df_engineered[column_id] = id_data

#         # Perform feature selection
#         logger.info("Performing feature selection...")
#         df_selected, selected_features = feature_selection(df_engineered, target_column=target_column, task="regression")
#         print(selected_features)
#         print("selected_features")

#         # Split data
#         logger.info("Splitting data into training and testing sets...")
#         best_model_name, X_train, y_train, X_test, y_test = train_test_model_selection(
#             df_selected, target_column=target_column, task='regression'
#         )

#         # Separate ID column for the test set
#         # test_ids = X_test[column_id].copy()
#         # X_train.drop(columns=[column_id], inplace=True)
#         # X_test.drop(columns=[column_id], inplace=True)

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








# def predict_new_data(new_data, bucket_name="artifacts1137",user_id=None, chat_id=None, column_id=None):
#     """
#     Loads the trained model and saved preprocessing artifacts to predict on new data.
    
#     Parameters:
#     - new_csv_path: string, path to the new CSV file for prediction.
#     - feature_defs_path: string, path to the saved feature definitions.
    
#     Returns:
#     - predictions: array of predictions for the target column in the original scale.
#     """
#     try:


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




#         # Load new data
#         # logger.info(f"Loading and preprocessing new data from {new_csv_path}...")
#         # new_data = pd.read_csv(new_csv_path)

#         # Apply imputation and encoding to new data
#         logger.info("Applying imputation and encoding to new data...")
#         new_data_processed, _ = automatic_imputation(new_data, target_column=None, imputers=imputers)
#         new_data_processed, _ = handle_categorical_features(new_data_processed, cardinality_threshold=10, encoders=encoder, saved_column_names=saved_column_names)
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




def train_pipeline(df, target_column, user_id, chat_id, column_id):
    """
    Complete machine learning pipeline with direct S3 upload for artifacts.
    Includes user_id and chat_id to maintain uniqueness.
    """
    try:
        logger.info("Dataset received successfully.")
        logger.info(f"Dataset shape: {df.shape}")
        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        logger.info(f"Entity (ID) column '{column_id}' will be retained.")
        # Save the ID column for later use
        id_data = df[column_id].copy()

        # Handle Missing Values, Encoding, Feature Engineering
        logger.info("Starting missing value imputation...")
        df, imputers = automatic_imputation(df, target_column=target_column)
        logger.info("Handling categorical features...")
        #df_encoded, encoder = handle_categorical_features(df, cardinality_threshold=3)
        df_encoded, encoder = handle_categorical_features(
            df, 
            id_column=column_id, 
            cardinality_threshold=3
        )
        saved_column_names = df_encoded.columns.tolist()

        logger.info("Performing feature engineering...")
        df_engineered, feature_defs = feature_engineering(df_encoded, target_column=target_column, training=True, id_column=column_id)
        df_engineered = normalize_column_names(df_engineered)
        logger.info("Feature engineering complete.")
        # Add back the ID column (unaltered)
        #df_engineered[column_id] = id_data

        # Perform feature selection
        logger.info("Performing feature selection...")
        df_selected, selected_features = feature_selection(df_engineered, target_column=target_column, task="regression", id_column=column_id)
        print(selected_features)
        print("selected_features")

        # Split data
        logger.info("Splitting data into training and testing sets...")
        best_model_name, X_train, y_train, X_test, y_test, test_ids = train_test_model_selection(
            df_selected, target_column=target_column, task='regression', id_column=column_id
        )

        # Separate ID column for the test set
        #test_ids = X_test[column_id].copy()
        #X_train.drop(columns=[column_id], inplace=True)
        #X_test.drop(columns=[column_id], inplace=True)

        # Hyperparameter Tuning
        logger.info(f"Performing hyperparameter tuning for the best model: {best_model_name}...")
        best_model, best_params = hyperparameter_tuning(
            best_model_name=best_model_name, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, task='regression'
        )

        # Finalize and Save Model (now pass user_id and chat_id)
        logger.info("Finalizing and evaluating the model...")
        final_metrics = finalize_and_evaluate_model(
            best_model.__class__, best_params, X_train, y_train, X_test, y_test,
            # user_id=user_id, chat_id=chat_id  # Pass them here
            user_id=user_id, chat_id=chat_id, test_ids=test_ids
        )

        # Upload artifacts to S3
        logger.info("Uploading artifacts directly to S3...")
        bucket_name = "artifacts1137"
        prefix = "ml-artifacts/"

        def save_to_s3(obj, filename):
            with io.BytesIO() as f:
                joblib.dump(obj, f)
                f.seek(0)
                s3_key = f"{prefix}{filename}"
                upload_to_s3(f, bucket_name, s3_key)
                logger.info(f"Uploaded {filename} to S3 as {s3_key}")

        save_to_s3(best_model, 'best_model.joblib')
        save_to_s3(imputers, 'imputers.joblib')
        save_to_s3(encoder, 'encoder.joblib')
        save_to_s3(feature_defs, 'feature_defs.joblib')
        save_to_s3(selected_features, 'selected_features.pkl')
        if target_column in saved_column_names:
            saved_column_names.remove(target_column)
            save_to_s3(saved_column_names, 'saved_column_names.pkl')
            logger.info("Categorical feature encoding complete.")

        logger.info("Artifacts uploaded to S3 successfully.")
        print(final_metrics)
        return best_model, best_params

    except Exception as e:
        logger.error(f"Error during pipeline execution: {e}")
        raise




def load_from_s3(bucket_name, s3_key):
    """
    Loads a file from S3 into memory using a BytesIO buffer.
    """
    s3 = get_s3_client()
    try:
        # Check if the file exists
        s3.head_object(Bucket=bucket_name, Key=s3_key)
        logger.info(f"File {s3_key} exists in bucket {bucket_name}.")

        # Download the file
        buffer = io.BytesIO()
        s3.download_fileobj(bucket_name, s3_key, buffer)
        buffer.seek(0)  # Reset buffer to the beginning
        return buffer
    except s3.exceptions.ClientError as e:
        if e.response['Error']['Code'] == "404":
            logger.error(f"The key {s3_key} does not exist in bucket {bucket_name}.")
            raise FileNotFoundError(f"S3 Key not found: {s3_key}")
        else:
            logger.error(f"Error accessing {s3_key} in bucket {bucket_name}: {e}")
            raise








def predict_new_data(new_data, bucket_name="artifacts1137",id_column=None):
    """
    Loads the trained model and saved preprocessing artifacts to predict on new data.
    
    Parameters:
    - new_csv_path: string, path to the new CSV file for prediction.
    - feature_defs_path: string, path to the saved feature definitions.
    
    Returns:
    - predictions: array of predictions for the target column in the original scale.
    """
    try:
        logger.info(f"Dataset loaded successfully. Shape: {new_data.shape}")
        logger.info(f"Entity (ID) column '{id_column}' will be retained.")


        # S3 keys for artifacts
        prefix = "ml-artifacts/"
        model_key = f"{prefix}best_model.joblib"
        imputers_key = f"{prefix}imputers.joblib"
        encoder_key = f"{prefix}encoder.joblib"
        selected_features_key = f"{prefix}selected_features.pkl"
        feature_defs_key = f"{prefix}feature_defs.joblib"
        saved_column_names_key = f"{prefix}saved_column_names.pkl"

        # Load artifacts directly from S3
        logger.info("Loading trained model and artifacts from S3...")
        model = joblib.load(load_from_s3(bucket_name, model_key))
        imputers = joblib.load(load_from_s3(bucket_name, imputers_key))
        encoder = joblib.load(load_from_s3(bucket_name, encoder_key))
        selected_features = joblib.load(load_from_s3(bucket_name, selected_features_key))
        feature_defs = joblib.load(load_from_s3(bucket_name, feature_defs_key))
        saved_column_names = joblib.load(load_from_s3(bucket_name, saved_column_names_key))
        logger.info("Model and artifacts loaded successfully.")
        # import pdb;pdb.set_trace()
        print("getting in to ifs")
        
        if id_column:
            ids = new_data[id_column].copy()
            print(ids)
            new_data = new_data.drop(columns=[id_column])  # Exclude ID column from processing
            print(new_data.columns)
            logger.info(f"Excluded ID column '{id_column}' from processing.")



        # Load new data
        # logger.info(f"Loading and preprocessing new data from {new_csv_path}...")
        # new_data = pd.read_csv(new_csv_path)

        # Apply imputation and encoding to new data
        logger.info("Applying imputation and encoding to new data...")
        new_data_processed, _ = automatic_imputation(new_data, target_column=None, imputers=imputers)
        new_data_processed, _ = handle_categorical_features(new_data_processed, cardinality_threshold=10, encoders=encoder, saved_column_names=saved_column_names,id_column = id_column)
        logger.info("Imputation and encoding complete.")

        # Apply feature engineering using saved feature definitions
        logger.info("Applying feature engineering using saved feature definitions...")
        #feature_defs = joblib.load("featurengineering_feature_defs.pkl")
        # feature_matrix, _ = feature_engineering(new_data_processed, training=False)
        feature_matrix = feature_engineering(new_data_processed, training=False, feature_defs=feature_defs)
        feature_matrix = normalize_column_names(feature_matrix)

        logger.info("Feature engineering complete.")

        logger.info("Applying feature selection...")
        selected_feature_matrix = feature_matrix[selected_features]  # Align with selected features from training
        logger.info(f"Feature matrix reduced to selected features: {selected_feature_matrix.shape}")


        # Predict on the processed new data
        logger.info("Making predictions...")
        predictions_scaled = model.predict(selected_feature_matrix)
        
        
        # Combine predictions with IDs
        predictions_with_ids = pd.DataFrame({
            id_column: ids,  # Include the original entity ID
            "Predicted": predictions_scaled
        })
        print(predictions_with_ids)
        logger.info("Predictions with IDs:")
        logger.info(f"\n{predictions_with_ids.head()}")

        # Check the shape before reshaping
        logger.info(f"Predictions before reshape: {predictions_scaled.shape}")
        #predictions_scaled = predictions_scaled.reshape(-1, 1)
        logger.info(f"Predictions after reshape: {predictions_scaled.shape}")

        # Inverse transform predictions to original scale
        #predictions = target_scaler.inverse_transform(predictions_scaled).flatten()
        #logger.info(f"Predictions after inverse transformation: {predictions[:10]}")

        logger.info("Predictions complete.")
        #return predictions
        return predictions_scaled

    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise
