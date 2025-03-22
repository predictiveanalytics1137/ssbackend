


# src/finalization.py
import requests
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.logging_config import get_logger
import pandas as pd

logger = get_logger(__name__)

# def finalize_and_evaluate_model(best_model_class, best_params, X_train, y_train, X_test, y_test, user_id, chat_id,test_ids):
#     """
#     Finalizes the model, evaluates it, and posts results along with user_id and chat_id.
#     """
#     try:
#         logger.info("Initializing the model with best hyperparameters...")
#         best_model = best_model_class(**best_params)

#         logger.info("Training the model with best hyperparameters...")
#         best_model.fit(X_train, y_train)

#         logger.info("Predicting on the test set...")
#         y_test_pred = best_model.predict(X_test)

#         # Combine predictions with test IDs
#         logger.info("Merging test predictions with IDs...")
#         predictions_with_ids = pd.DataFrame({
#             'ID': test_ids,
#             'Actual': y_test.tolist(),
#             'Predicted': y_test_pred.tolist()
#         })
#         logger.info("Predictions with IDs:")
#         logger.info(f"\n{predictions_with_ids.head()}")
#         print(predictions_with_ids)

#         logger.info("Predicting on the training set...")
#         y_train_pred = best_model.predict(X_train)

#         logger.info("Evaluating model performance on testing data...")
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)

#         logger.info("Evaluating model performance on training data...")
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)

#         logger.info(f"Testing - RMSE: {test_rmse}, R²: {test_r2}, MAE: {test_mae}")
#         logger.info(f"Training - RMSE: {train_rmse}, R²: {train_r2}, MAE: {train_mae}")

#         # Determine Model Performance
#         r2_difference = abs(train_r2 - test_r2)
#         if train_r2 > 0.9 and test_r2 > 0.9 and r2_difference < 0.05:
#             model_assessment = "Good Fit"
#         elif train_r2 > 0.9 and test_r2 < 0.7:
#             model_assessment = "Overfitting"
#         elif train_r2 < 0.7 and test_r2 < 0.7:
#             model_assessment = "Underfitting"
#         else:
#             model_assessment = "Potential Overfitting or Issues"

#         logger.info(f"Model assessment: {model_assessment}")

#         # final_metrics not used directly below, but we return it
#         final_metrics = {
#             'Testing': {'RMSE': test_rmse, 'R-squared': test_r2, 'MAE': test_mae},
#             'Training': {'RMSE': train_rmse, 'R-squared': train_r2, 'MAE': train_mae},
#             'Assessment': model_assessment
#         }

#         # Feature Importance
#         feature_importance = {}
#         if hasattr(best_model, 'feature_importances_'):
#             feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))

#         # Core statistics
#         core_statistics = X_train.describe().to_dict()

#         # Attribute table statistics
#         attribute_statistics = X_train.describe(include='all').to_dict()

#         payload = {
#             "model_metrics": {
#                 "training": {
#                     "rmse": float(train_rmse),
#                     "r2_score": float(train_r2),
#                     "mae": float(train_mae)
#                 },
#                 "testing": {
#                     "rmse": float(test_rmse),
#                     "r2_score": float(test_r2),
#                     "mae": float(test_mae)
#                 },
#                 "assessment": model_assessment
#             },
#             "attribute_columns": list(X_train.columns),
#             "feature_importance": {k: float(v) for k, v in feature_importance.items()},
#             "core_statistics": {
#                 k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val for stat, val in stats.items()}
#                 for k, stats in core_statistics.items()
#             },
#             "attribute_statistics": {
#                 k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val for stat, val in stats.items()}
#                 for k, stats in attribute_statistics.items()
#             },
#             "predictions": {
#                 "actual": y_test.tolist(),
#                 "predicted": y_test_pred.tolist()
#             },
#             # Include user_id and chat_id for uniqueness
#             "user_id": user_id,
#             "chat_id": chat_id
#         }

#         api_url = "http://127.0.0.1:8000/model/modelresults/"
#         headers = {"Content-Type": "application/json"}
#         response = requests.post(api_url, data=json.dumps(payload), headers=headers)

#         if response.status_code == 201:
#             logger.info("Data successfully posted to the Django API with {user_id} and chat_id.")
#         else:
#             logger.error(f"Failed to post data. Status code: {response.status_code}, Response: {response.text}")

#         logger.info("Final metrics computed successfully.")
#         return final_metrics

#     except Exception as e:
#         logger.error(f"Error during final model evaluation: {e}")
#         raise



# v2
# import requests
# import json
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from src.logging_config import get_logger
# import io
# from src.s3_operations import upload_to_s3

# logger = get_logger(__name__)

# def finalize_and_evaluate_model(best_model_class, best_params, X_train, y_train, 
#                                 X_test, y_test, user_id, chat_id, test_ids=None):
#     """
#     Finalizes the model by instantiating it with the best hyperparams, then fits & evaluates.
#     Returns final metrics and posts them to an API (stub).
#     """
#     try:
#         logger.info("Initializing the model with best hyperparameters...")
#         best_model = best_model_class(**best_params)

#         logger.info("Fitting the model on training data...")
#         best_model.fit(X_train, y_train)

#         logger.info("Predicting on test set...")
#         y_test_pred = best_model.predict(X_test)
        
#         # Prepare predictions DataFrame with IDs
#         predictions_with_ids = None

#         if test_ids is not None:
#             predictions_with_ids = pd.DataFrame({
#                 'ID': test_ids,
#                 'Actual': y_test.tolist(),
#                 'Predicted': y_test_pred.tolist()
#             })
#             logger.info(f"Sample predictions:\n{predictions_with_ids.head()}")

#         # Evaluate on test data
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)

#         # Evaluate on training data
#         y_train_pred = best_model.predict(X_train)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)

#         logger.info(f"Test RMSE: {test_rmse:.4f}, R^2: {test_r2:.4f}, MAE: {test_mae:.4f}")
#         logger.info(f"Train RMSE: {train_rmse:.4f}, R^2: {train_r2:.4f}, MAE: {train_mae:.4f}")

#         r2_diff = abs(train_r2 - test_r2)
#         if train_r2 > 0.9 and test_r2 > 0.9 and r2_diff < 0.05:
#             model_assessment = "Good Fit"
#         elif train_r2 > 0.9 and test_r2 < 0.7:
#             model_assessment = "Overfitting"
#         elif train_r2 < 0.7 and test_r2 < 0.7:
#             model_assessment = "Underfitting"
#         else:
#             model_assessment = "Average/Check further"

#         logger.info(f"Model assessment: {model_assessment}")

#         final_metrics = {
#             'Training': {'RMSE': train_rmse, 'R2': train_r2, 'MAE': train_mae},
#             'Testing': {'RMSE': test_rmse, 'R2': test_r2, 'MAE': test_mae},
#             'Assessment': model_assessment
#         }
        
        
        
        
#        # Save predictions to S3 as CSV
#         if predictions_with_ids is not None:
#             logger.info("Saving predictions with IDs to S3...")
#             bucket_name = "artifacts1137"
#             predictions_csv_key = f"ml-artifacts/{chat_id}/testpredictions_with_ids.csv"

#             # Convert StringIO to BytesIO
#             csv_buffer = io.StringIO()
#             predictions_with_ids.to_csv(csv_buffer, index=False)
#             csv_buffer.seek(0)  # Reset buffer to the beginning

#             # Convert to BytesIO for upload
#             bytes_buffer = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
#             upload_to_s3(bytes_buffer, bucket_name, predictions_csv_key)
#             logger.info(f"Predictions saved to s3://{bucket_name}/{predictions_csv_key}")
        
        
        
        
        
#         # Feature Importance
#         feature_importance = {}
#         if hasattr(best_model, 'feature_importances_'):
#             feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))

#         # Core statistics
#         core_statistics = X_train.describe().to_dict()

#         # Attribute table statistics
#         attribute_statistics = X_train.describe(include='all').to_dict()

#         payload = {
#             "model_metrics": {
#                 "training": {
#                     "rmse": float(train_rmse),
#                     "r2_score": float(train_r2),
#                     "mae": float(train_mae)
#                 },
#                 "testing": {
#                     "rmse": float(test_rmse),
#                     "r2_score": float(test_r2),
#                     "mae": float(test_mae)
#                 },
#                 "assessment": model_assessment
#             },
#             "attribute_columns": list(X_train.columns),
#             "feature_importance": {k: float(v) for k, v in feature_importance.items()},
#             "core_statistics": {
#                 k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val for stat, val in stats.items()}
#                 for k, stats in core_statistics.items()
#             },
#             "attribute_statistics": {
#                 k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val for stat, val in stats.items()}
#                 for k, stats in attribute_statistics.items()
#             },
#             "predictions": {
#                 "actual": y_test.tolist(),
#                 "predicted": y_test_pred.tolist()
#             },
#             # Include user_id and chat_id for uniqueness
#             "user_id": user_id,
#             "chat_id": chat_id
#         }
        
        
        
        
        

# # =============================================================================
# #         # Example of posting results to an API (placeholder)
# #         payload = {
# #             "model_metrics": {
# #                 "training": {
# #                     "rmse": float(train_rmse),
# #                     "r2_score": float(train_r2),
# #                     "mae": float(train_mae)
# #                 },
# #                 "testing": {
# #                     "rmse": float(test_rmse),
# #                     "r2_score": float(test_r2),
# #                     "mae": float(test_mae)
# #                 },
# #                 "assessment": model_assessment
# #             },
# #             "user_id": user_id,
# #             "chat_id": chat_id
# #         }
# # =============================================================================

#         api_url = "http://127.0.0.1:8000/model/modelresults/"
#         headers = {"Content-Type": "application/json"}
#         try:
#             response = requests.post(api_url, data=json.dumps(payload), headers=headers)
#             if response.status_code == 201:
#                 logger.info("Data successfully posted to the Django API.")
#             else:
#                 logger.warning(f"API post failed: {response.status_code} {response.text}")
#                 print(f"API post failed: {response.status_code} {response.text}")
#         except Exception as post_err:
#             logger.error(f"Error posting data to API: {post_err}")

#         return final_metrics

#     except Exception as e:
#         logger.error(f"Error during final model evaluation: {e}")
#         raise




# v3 incressed matrics


# import requests
# import json
# import numpy as np
# import pandas as pd
# from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
# from src.logging_config import get_logger
# import io
# from src.s3_operations import upload_to_s3
# import datetime
# import time

# logger = get_logger(__name__)

# def finalize_and_evaluate_model(best_model_class, best_params, X_train, y_train, 
#                                 X_test, y_test, user_id, chat_id, test_ids=None):
#     """
#     Finalizes the model by instantiating it with the best hyperparameters, then fits & evaluates.
#     Returns final metrics and posts them to an API (stub). The payload now includes extra metadata
#     for dashboard visualization.
#     """
#     try:
#         logger.info("Initializing the model with best hyperparameters...")
#         eval_start = time.time()  # Start evaluation timer
        
#         best_model = best_model_class(**best_params)

#         logger.info("Fitting the model on training data...")
#         best_model.fit(X_train, y_train)

#         logger.info("Predicting on test set...")
#         y_test_pred = best_model.predict(X_test)
        
#         # Prepare predictions DataFrame with IDs
#         predictions_with_ids = None

#         if test_ids is not None:
#             predictions_with_ids = pd.DataFrame({
#                 'ID': test_ids,
#                 'Actual': y_test.tolist(),
#                 'Predicted': y_test_pred.tolist()
#             })
#             logger.info(f"Sample predictions:\n{predictions_with_ids.head()}")

#         # Evaluate on test data
#         test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
#         test_r2 = r2_score(y_test, y_test_pred)
#         test_mae = mean_absolute_error(y_test, y_test_pred)

#         # Evaluate on training data
#         y_train_pred = best_model.predict(X_train)
#         train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
#         train_r2 = r2_score(y_train, y_train_pred)
#         train_mae = mean_absolute_error(y_train, y_train_pred)

#         logger.info(f"Test RMSE: {test_rmse:.4f}, R^2: {test_r2:.4f}, MAE: {test_mae:.4f}")
#         logger.info(f"Train RMSE: {train_rmse:.4f}, R^2: {train_r2:.4f}, MAE: {train_mae:.4f}")

#         r2_diff = abs(train_r2 - test_r2)
#         if train_r2 > 0.9 and test_r2 > 0.9 and r2_diff < 0.05:
#             model_assessment = "Good Fit"
#         elif train_r2 > 0.9 and test_r2 < 0.7:
#             model_assessment = "Overfitting"
#         elif train_r2 < 0.7 and test_r2 < 0.7:
#             model_assessment = "Underfitting"
#         else:
#             model_assessment = "Average/Check further"

#         logger.info(f"Model assessment: {model_assessment}")

#         final_metrics = {
#             'Training': {'RMSE': train_rmse, 'R2': train_r2, 'MAE': train_mae},
#             'Testing': {'RMSE': test_rmse, 'R2': test_r2, 'MAE': test_mae},
#             'Assessment': model_assessment
#         }
        
#         # Save predictions to S3 as CSV
#         if predictions_with_ids is not None:
#             logger.info("Saving predictions with IDs to S3...")
#             bucket_name = "artifacts1137"
#             predictions_csv_key = f"ml-artifacts/{chat_id}/testpredictions_with_ids.csv"

#             csv_buffer = io.StringIO()
#             predictions_with_ids.to_csv(csv_buffer, index=False)
#             csv_buffer.seek(0)
#             bytes_buffer = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
#             upload_to_s3(bytes_buffer, bucket_name, predictions_csv_key)
#             logger.info(f"Predictions saved to s3://{bucket_name}/{predictions_csv_key}")
        
#         # Feature Importance
#         feature_importance = {}
#         if hasattr(best_model, 'feature_importances_'):
#             feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))

#         # Core statistics
#         core_statistics = X_train.describe().to_dict()

#         # Attribute table statistics
#         attribute_statistics = X_train.describe(include='all').to_dict()

#         # Compute residuals and their statistics
#         residuals = np.array(y_test) - np.array(y_test_pred)
#         residuals_stats = {
#             "mean": float(np.mean(residuals)),
#             "median": float(np.median(residuals)),
#             "std": float(np.std(residuals)),
#             "min": float(np.min(residuals)),
#             "max": float(np.max(residuals))
#         }
        
#         # Compute evaluation duration
#         eval_duration = time.time() - eval_start

#         # Data overview metadata
#         data_overview = {
#             "train_samples": int(X_train.shape[0]),
#             "test_samples": int(len(y_test)),
#             "num_features": int(X_train.shape[1])
#         }

#         # Build the enhanced payload with additional metadata
#         payload = {
#             "model_metrics": {
#                 "training": {
#                     "rmse": float(train_rmse),
#                     "r2_score": float(train_r2),
#                     "mae": float(train_mae)
#                 },
#                 "testing": {
#                     "rmse": float(test_rmse),
#                     "r2_score": float(test_r2),
#                     "mae": float(test_mae)
#                 },
#                 "assessment": model_assessment
#             },
#             "attribute_columns": list(X_train.columns),
#             "feature_importance": {k: float(v) for k, v in feature_importance.items()},
#             "core_statistics": {
#                 k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val for stat, val in stats.items()}
#                 for k, stats in core_statistics.items()
#             },
#             "attribute_statistics": {
#                 k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val for stat, val in stats.items()}
#                 for k, stats in attribute_statistics.items()
#             },
#             "predictions": {
#                 "actual": y_test.tolist(),
#                 "predicted": y_test_pred.tolist()
#             },
#             "best_model_info": {
#                 "name": best_model.__class__.__name__,
#                 "parameters": best_params
#             },
#             "residuals_statistics": residuals_stats,
#             "data_overview": data_overview,
#             "evaluation_duration": float(eval_duration),
#             "timestamp": datetime.datetime.now().isoformat(),
#             "user_id": user_id,
#             "chat_id": chat_id
#         }

#         print("this is the payload")
#         print(payload)
#         print("---------------------------------------------")
        
        
#         api_url = "http://127.0.0.1:8000/model/modelresults/"
#         headers = {"Content-Type": "application/json"}
#         try:
#             response = requests.post(api_url, data=json.dumps(payload), headers=headers)
#             if response.status_code == 201:
#                 logger.info("Data successfully posted to the Django API.")
#             else:
#                 logger.warning(f"API post failed: {response.status_code} {response.text}")
#                 print(f"API post failed: {response.status_code} {response.text}")
#         except Exception as post_err:
#             logger.error(f"Error posting data to API: {post_err}")

#         return final_metrics

#     except Exception as e:
#         logger.error(f"Error during final model evaluation: {e}")
#         raise




# testing

import requests
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.logging_config import get_logger
import io
from src.s3_operations import upload_to_s3
import datetime
import time
import shap

logger = get_logger(__name__)

def finalize_and_evaluate_model(best_model_class, best_params, X_train, y_train, 
                              X_test, y_test, user_id, chat_id, test_ids=None):
    """
    Finalizes the model by instantiating it with the best hyperparameters, then fits & evaluates.
    Returns final metrics and posts them to an API. The payload includes comprehensive metadata
    for dashboard visualization and model analysis.
    """
    try:
        logger.info("Initializing the model with best hyperparameters...")
        eval_start = time.time()  # Start evaluation timer
        
        best_model = best_model_class(**best_params)

        logger.info("Fitting the model on training data...")
        best_model.fit(X_train, y_train)

        logger.info("Predicting on test set...")
        y_test_pred = best_model.predict(X_test)
        
        # Prepare predictions DataFrame with IDs
        predictions_with_ids = None
        if test_ids is not None:
            predictions_with_ids = pd.DataFrame({
                'ID': test_ids,
                'Actual': y_test.tolist(),
                'Predicted': y_test_pred.tolist()
            })
            logger.info(f"Sample predictions:\n{predictions_with_ids.head()}")

        # Evaluate on test data
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        # Evaluate on training data
        y_train_pred = best_model.predict(X_train)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)

        logger.info(f"Test RMSE: {test_rmse:.4f}, R^2: {test_r2:.4f}, MAE: {test_mae:.4f}")
        logger.info(f"Train RMSE: {train_rmse:.4f}, R^2: {train_r2:.4f}, MAE: {train_mae:.4f}")

        # Model assessment
        r2_diff = abs(train_r2 - test_r2)
        if train_r2 > 0.9 and test_r2 > 0.9 and r2_diff < 0.05:
            model_assessment = "Good Fit"
        elif train_r2 > 0.9 and test_r2 < 0.7:
            model_assessment = "Overfitting"
        elif train_r2 < 0.7 and test_r2 < 0.7:
            model_assessment = "Underfitting"
        else:
            model_assessment = "Average/Check further"

        logger.info(f"Model assessment: {model_assessment}")

        # Save predictions to S3 as CSV
        if predictions_with_ids is not None:
            logger.info("Saving predictions with IDs to S3...")
            bucket_name = "artifacts1137"
            predictions_csv_key = f"ml-artifacts/{chat_id}/testpredictions_with_ids.csv"

            csv_buffer = io.StringIO()
            predictions_with_ids.to_csv(csv_buffer, index=False)
            csv_buffer.seek(0)
            bytes_buffer = io.BytesIO(csv_buffer.getvalue().encode('utf-8'))
            upload_to_s3(bytes_buffer, bucket_name, predictions_csv_key)
            logger.info(f"Predictions saved to s3://{bucket_name}/{predictions_csv_key}")

        # ============== NEW METRICS CALCULATIONS ==============
        # Residual analysis
        residuals = y_test - y_test_pred
        residual_stats = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            '25%': float(np.percentile(residuals, 25)),
            'median': float(np.median(residuals)),
            '75%': float(np.percentile(residuals, 75)),
            'max': float(np.max(residuals))
        }

        # Model metadata
        model_type = best_model.__class__.__name__
        training_samples = X_train.shape[0]
        testing_samples = X_test.shape[0]
        num_features = X_train.shape[1]

        # Prediction distributions
        actual_stats = {
            'mean': float(np.mean(y_test)),
            'std': float(np.std(y_test)),
            'min': float(np.min(y_test)),
            'max': float(np.max(y_test))
        }
        
        predicted_stats = {
            'mean': float(np.mean(y_test_pred)),
            'std': float(np.std(y_test_pred)),
            'min': float(np.min(y_test_pred)),
            'max': float(np.max(y_test_pred))
        }

        # Feature correlations with target
        feature_correlation = X_train.corrwith(y_train).to_dict()
        feature_correlation = {k: float(v) for k, v in feature_correlation.items()}

        # Enhanced feature importance
        feature_importance = {}
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
        elif hasattr(best_model, 'coef_'):
            if len(best_model.coef_.shape) > 1:
                coefs = best_model.coef_[0]  # Handle multi-class
            else:
                coefs = best_model.coef_
            feature_importance = dict(zip(X_train.columns, coefs))

        # Top 10 features
        top_n = 10
        sorted_features = sorted(feature_importance.items(), 
                               key=lambda x: abs(x[1]), reverse=True)
        top_features = {k: float(v) for k, v in sorted_features[:top_n]}

        # SHAP importance calculation
        shap_importance = {}
        try:
            sample_size = min(100, X_train.shape[0])
            X_sample = X_train.sample(sample_size, random_state=42)
            explainer = shap.Explainer(best_model, X_sample)
            shap_values = explainer(X_sample)
            
            shap_importance = pd.DataFrame({
                'feature': X_train.columns,
                'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
            
            shap_importance = {
                k: float(v) 
                for k, v in shap_importance.set_index('feature')['mean_abs_shap'].items()
            }
        except Exception as e:
            logger.error(f"SHAP calculation failed: {e}")

        # Core and attribute statistics
        core_statistics = X_train.describe().to_dict()
        attribute_statistics = X_train.describe(include='all').to_dict()
        
        # Compute evaluation duration
        eval_duration = time.time() - eval_start

        # Build comprehensive payload
        payload = {
            "model_metrics": {
                "training": {
                    "rmse": float(train_rmse),
                    "r2_score": float(train_r2),
                    "mae": float(train_mae)
                },
                "testing": {
                    "rmse": float(test_rmse),
                    "r2_score": float(test_r2),
                    "mae": float(test_mae)
                },
                "assessment": model_assessment,
                "residuals": residual_stats,
            },
            "model_metadata": {
                "model_type": model_type,
                "hyperparameters": best_params,
                "training_samples": training_samples,
                "testing_samples": testing_samples,
                "num_features": num_features,
                "evaluation_duration": float(eval_duration),
                "timestamp": datetime.datetime.now().isoformat()
            },
            "data_characteristics": {
                "actual_distribution": actual_stats,
                "predicted_distribution": predicted_stats,
                "feature_correlations": feature_correlation,
            },
            "feature_analysis": {
                "attribute_columns": list(X_train.columns),
                "feature_importance": {k: float(v) for k, v in feature_importance.items()},
                "top_features": top_features,
                "shap_importance": shap_importance,
            },
            "core_statistics": {
                k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val 
                   for stat, val in stats.items()}
                for k, stats in core_statistics.items()
            },
            "attribute_statistics": {
                k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val 
                   for stat, val in stats.items()}
                for k, stats in attribute_statistics.items()
            },
            "predictions": {
                "actual": y_test.tolist(),
                "predicted": y_test_pred.tolist()
            },
            "user_id": user_id,
            "chat_id": chat_id
        }

        # Post results to API
        api_url = "http://127.0.0.1:8000/model/modelresults/"
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(api_url, data=json.dumps(payload), headers=headers)
            if response.status_code == 201:
                logger.info("Data successfully posted to the Django API.")
            else:
                logger.warning(f"API post failed: {response.status_code} {response.text}")
                print(f"API post failed: {response.status_code} {response.text}")
        except Exception as post_err:
            logger.error(f"Error posting data to API: {post_err}")

        # Return final metrics for backward compatibility
        final_metrics = {
            'Training': {'RMSE': train_rmse, 'R2': train_r2, 'MAE': train_mae},
            'Testing': {'RMSE': test_rmse, 'R2': test_r2, 'MAE': test_mae},
            'Assessment': model_assessment
        }
        return final_metrics

    except Exception as e:
        logger.error(f"Error during final model evaluation: {e}")
        raise









import time
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
import io
import joblib
from src.logging_config import get_logger
import shap  # For feature importance (optional)

logger = get_logger(__name__)

# def finalize_and_evaluate_model_timeseries(final_model, X_train, predictions_df, user_id, chat_id, best_params=None):
#     """
#     Finalizes the time-series model by saving it, and generates metadata using validation predictions
#     from the pipeline. Posts results to an API.

#     Parameters:
#     - final_model: The trained model instance from the pipeline.
#     - X_train (pd.DataFrame): Training features (full dataset up to cutoff, for metadata).
#     - predictions_df (pd.DataFrame): Validation predictions from the pipeline (with 'actual' and 'predicted' columns).
#     - user_id (str): User identifier for logging and API.
#     - chat_id (str): Chat identifier for tracking and storing artifacts.

#     Returns:
#     - final_model: The finalized model instance (unchanged).
#     - final_metrics (dict): Aggregated validation metrics (RMSE, MAE, R2).
#     """
#     try:
#         logger.info("Finalizing the time-series model and generating metadata from validation predictions...")
#         eval_start = time.time()  # Start evaluation timer

#         # Use validation predictions from the pipeline for metrics
#         if predictions_df.empty:
#             logger.warning("No validation predictions provided; using training data as fallback.")
#             y_val_actual = X_train[target_column]  # Assuming target_column is available or adjust logic
#             y_val_pred = final_model.predict(X_train)
#         else:
#             y_val_actual = predictions_df['actual'].dropna()
#             y_val_pred = predictions_df['predicted'].dropna()

#         # Calculate aggregated validation metrics
#         val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
#         val_mae = mean_absolute_error(y_val_actual, y_val_pred)
#         val_r2 = r2_score(y_val_actual, y_val_pred) if len(y_val_actual) > 1 else 0.0

#         logger.info(f"Aggregated Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R^2: {val_r2:.4f}")

#         # Model assessment based on validation metrics
#         model_assessment = "Good Fit" if val_r2 > 0.7 else "Check Further"
#         logger.info(f"Model assessment: {model_assessment}")

#         # Model metadata
#         model_type = final_model.__class__.__name__
#         training_samples = X_train.shape[0]
#         num_features = X_train.shape[1]
#         eval_duration = time.time() - eval_start

#         # Feature importance (if available)
#         feature_importance = {}
#         if hasattr(final_model, 'feature_importances_'):
#             feature_importance = dict(zip(X_train.columns, final_model.feature_importances_))
#         elif hasattr(final_model, 'coef_'):
#             if len(final_model.coef_.shape) > 1:
#                 coefs = final_model.coef_[0]  # Handle multi-class
#             else:
#                 coefs = final_model.coef_
#             feature_importance = dict(zip(X_train.columns, coefs))

#         # Top 10 features
#         top_n = 10
#         sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
#         top_features = {k: float(v) for k, v in sorted_features[:top_n]}

#         # SHAP importance (optional, for advanced analysis)
#         shap_importance = {}
#         try:
#             sample_size = min(100, X_train.shape[0])
#             X_sample = X_train.sample(sample_size, random_state=42)
#             explainer = shap.Explainer(final_model, X_sample)
#             shap_values = explainer(X_sample)
            
#             shap_importance = pd.DataFrame({
#                 'feature': X_train.columns,
#                 'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
#             }).sort_values('mean_abs_shap', ascending=False)
            
#             shap_importance = {
#                 k: float(v) 
#                 for k, v in shap_importance.set_index('feature')['mean_abs_shap'].items()
#             }
#         except Exception as e:
#             logger.warning(f"SHAP calculation failed: {e}")

#         # Data characteristics from validation data
#         actual_stats = {
#             'mean': float(np.mean(y_val_actual)),
#             'std': float(np.std(y_val_actual)) if len(y_val_actual) > 1 else 0.0,
#             'min': float(np.min(y_val_actual)),
#             'max': float(np.max(y_val_actual))
#         }
        
#         predicted_stats = {
#             'mean': float(np.mean(y_val_pred)),
#             'std': float(np.std(y_val_pred)) if len(y_val_pred) > 1 else 0.0,
#             'min': float(np.min(y_val_pred)),
#             'max': float(np.max(y_val_pred))
#         }

#         # Feature correlations with target (using training data for stability)
#         feature_correlation = X_train.corrwith(pd.Series(y_val_actual, index=X_train.index)).to_dict()
#         feature_correlation = {k: float(v) for k, v in feature_correlation.items()}

#         # Core and attribute statistics (from training data)
#         core_statistics = X_train.describe().to_dict()
#         attribute_statistics = X_train.describe(include='all').to_dict()

#         # Build comprehensive payload for API using only validation predictions
#         payload = {
#             "model_metrics": {
#                 "validation": {
#                     "rmse": float(val_rmse),
#                     "mae": float(val_mae),
#                     "r2_score": float(val_r2)
#                 },
#                 "assessment": model_assessment
#             },
#             "model_metadata": {
#                 "model_type": model_type,
#                 "hyperparameters": best_params if best_params else {},  # Ensure best_params exists
#                 "training_samples": training_samples,
#                 "num_features": num_features,
#                 "evaluation_duration": float(eval_duration),
#                 "timestamp": pd.Timestamp.now().isoformat()
#             },
#             "data_characteristics": {
#                 "actual_distribution": actual_stats,
#                 "predicted_distribution": predicted_stats,
#                 "feature_correlations": feature_correlation
#             },
#             "feature_analysis": {
#                 "attribute_columns": list(X_train.columns),
#                 "feature_importance": {k: float(v) for k, v in feature_importance.items()},
#                 "top_features": top_features,
#                 "shap_importance": shap_importance
#             },
#             "core_statistics": {
#                 k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val 
#                    for stat, val in stats.items()}
#                 for k, stats in core_statistics.items()
#             },
#             "attribute_statistics": {
#                 k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val 
#                    for stat, val in stats.items()}
#                 for k, stats in attribute_statistics.items()
#             },
#             "user_id": user_id,
#             "chat_id": chat_id
#         }
#         import pdb; pdb.set_trace()

#         # Post results to API
#         api_url = "http://127.0.0.1:8000/model/modelresults/"
#         headers = {"Content-Type": "application/json"}
#         try:
#             response = requests.post(api_url, data=json.dumps(payload), headers=headers)
#             if response.status_code == 201:
#                 logger.info("Data successfully posted to the Django API.")
#             else:
#                 logger.warning(f"API post failed: {response.status_code} {response.text}")
#                 print(f"API post failed: {response.status_code} {response.text}")
#         except Exception as post_err:
#             logger.error(f"Error posting data to API: {post_err}")

#         # Save the model to S3
#         logger.info("Saving final model to S3...")
#         bucket_name = "artifacts1137"
#         prefix = f"ml-artifacts/{chat_id}/"
#         model_key = f"final_model_{chat_id}.joblib"
#         with io.BytesIO() as f:
#             joblib.dump(final_model, f)
#             f.seek(0)
#             upload_to_s3(f, bucket_name, f"{prefix}{model_key}")
#         logger.info(f"Final model saved to s3://{bucket_name}/{prefix}{model_key}")

#         # Return final metrics for logging
#         final_metrics = {
#             'Validation': {'RMSE': val_rmse, 'MAE': val_mae, 'R2': val_r2},
#             'Assessment': model_assessment
#         }
#         return final_model, final_metrics

#     except Exception as e:
#         logger.error(f"Error during final time-series model evaluation: {e}")
#         raise


# import time
# import json
# import pandas as pd
# import numpy as np
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import requests
# import io
# import joblib
# from src.logging_config import get_logger
# import shap  # For feature importance (optional)

# logger = get_logger(__name__)
# import shap  # For feature importance (optional)

# def clean_nan_values(data):
#     """
#     Recursively clean NaN values from a dictionary or list, replacing them with None or 0.0.
#     Handles nested structures and NumPy/pandas NaN values.
#     """
#     if isinstance(data, (dict, list)):
#         return [
#             clean_nan_values(item) if isinstance(item, (dict, list))
#             else None if pd.isna(item) or (isinstance(item, float) and np.isnan(item))
#             else 0.0 if isinstance(item, float) and np.isinf(item) else item
#             for item in (data if isinstance(data, list) else data.items())
#         ] if isinstance(data, list) else {
#             key: clean_nan_values(value)
#             for key, value in data.items()
#         }
#     return None if pd.isna(data) or (isinstance(data, float) and np.isnan(data)) else 0.0 if isinstance(data, float) and np.isinf(data) else data

# def finalize_and_evaluate_model_timeseries(final_model, X_train, predictions_df, user_id, chat_id, best_params=None):
#     """
#     Finalizes the time-series model by saving it, and generates metadata using validation predictions
#     from the pipeline. Posts results to an API, including predictions.

#     Parameters:
#     - final_model: The trained model instance from the pipeline.
#     - X_train (pd.DataFrame): Training features (full dataset up to cutoff, for metadata).
#     - predictions_df (pd.DataFrame): Validation predictions from the pipeline (with 'actual' and 'predicted' columns).
#     - user_id (str): User identifier for logging and API.
#     - chat_id (str): Chat identifier for tracking and storing artifacts.

#     Returns:
#     - final_model: The finalized model instance (unchanged).
#     - final_metrics (dict): Aggregated validation metrics (RMSE, MAE, R2).
#     """
#     try:
#         logger.info("Finalizing the time-series model and generating metadata from validation predictions...")
#         eval_start = time.time()  # Start evaluation timer

#         # Use validation predictions from the pipeline for metrics and predictions
#         if predictions_df.empty:
#             logger.warning("No validation predictions provided; using training data as fallback.")
#             y_val_actual = X_train[target_column]  # Assuming target_column is available or adjust logic
#             y_val_pred = final_model.predict(X_train)
#         else:
#             y_val_actual = predictions_df['actual'].dropna()
#             y_val_pred = predictions_df['predicted'].dropna()

#         # Calculate aggregated validation metrics
#         val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
#         val_mae = mean_absolute_error(y_val_actual, y_val_pred)
#         val_r2 = r2_score(y_val_actual, y_val_pred) if len(y_val_actual) > 1 else 0.0

#         logger.info(f"Aggregated Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R^2: {val_r2:.4f}")

#         # Model assessment based on validation metrics
#         model_assessment = "Good Fit" if val_r2 > 0.7 else "Check Further"
#         logger.info(f"Model assessment: {model_assessment}")

#         # Model metadata
#         model_type = final_model.__class__.__name__
#         training_samples = X_train.shape[0]
#         num_features = X_train.shape[1]
#         eval_duration = time.time() - eval_start

#         # Feature importance (if available)
#         feature_importance = {}
#         if hasattr(final_model, 'feature_importances_'):
#             feature_importance = dict(zip(X_train.columns, final_model.feature_importances_))
#         elif hasattr(final_model, 'coef_'):
#             if len(final_model.coef_.shape) > 1:
#                 coefs = final_model.coef_[0]  # Handle multi-class
#             else:
#                 coefs = final_model.coef_
#             feature_importance = dict(zip(X_train.columns, coefs))

#         # Top 10 features
#         top_n = 10
#         sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
#         top_features = {k: float(v) for k, v in sorted_features[:top_n]}

#         # SHAP importance (optional, for advanced analysis)
#         shap_importance = {}
#         try:
#             sample_size = min(100, X_train.shape[0])
#             X_sample = X_train.sample(sample_size, random_state=42)
#             explainer = shap.Explainer(final_model, X_sample)
#             shap_values = explainer(X_sample)
            
#             shap_importance = pd.DataFrame({
#                 'feature': X_train.columns,
#                 'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
#             }).sort_values('mean_abs_shap', ascending=False)
            
#             shap_importance = {
#                 k: float(v) 
#                 for k, v in shap_importance.set_index('feature')['mean_abs_shap'].items()
#             }
#         except Exception as e:
#             logger.warning(f"SHAP calculation failed: {e}")

#         # Data characteristics from validation data
#         actual_stats = {
#             'mean': float(np.nan_to_num(np.mean(y_val_actual), nan=0.0)),
#             'std': float(np.nan_to_num(np.std(y_val_actual), nan=0.0)) if len(y_val_actual) > 1 else 0.0,
#             'min': float(np.nan_to_num(np.min(y_val_actual), nan=0.0)),
#             'max': float(np.nan_to_num(np.max(y_val_actual), nan=0.0))
#         }
        
#         predicted_stats = {
#             'mean': float(np.nan_to_num(np.mean(y_val_pred), nan=0.0)),
#             'std': float(np.nan_to_num(np.std(y_val_pred), nan=0.0)) if len(y_val_pred) > 1 else 0.0,
#             'min': float(np.nan_to_num(np.min(y_val_pred), nan=0.0)),
#             'max': float(np.nan_to_num(np.max(y_val_pred), nan=0.0))
#         }

#         # Feature correlations with target (using training data for stability, cleaning NaN)
#         feature_correlation = X_train.corrwith(pd.Series(y_val_actual, index=X_train.index)).to_dict()
#         feature_correlation = {k: float(np.nan_to_num(v, nan=0.0)) for k, v in feature_correlation.items()}

#         # Core and attribute statistics (from training data, cleaning NaN)
#         core_statistics = X_train.describe().to_dict()
#         core_statistics = {
#             k: {stat: float(np.nan_to_num(val, nan=0.0)) if isinstance(val, (np.integer, np.floating)) else val 
#                 for stat, val in stats.items()}
#             for k, stats in core_statistics.items()
#         }
        
#         attribute_statistics = X_train.describe(include='all').to_dict()
#         attribute_statistics = {
#             k: {stat: float(np.nan_to_num(val, nan=0.0)) if isinstance(val, (np.integer, np.floating)) else val 
#                 for stat, val in stats.items()}
#             for k, stats in attribute_statistics.items()
#         }

#         # Prepare predictions for the payload (validation actual and predicted values)
#         predictions_data = {
#             'actual': y_val_actual.tolist(),
#             'predicted': y_val_pred.tolist()
#         }

#         # Build comprehensive payload, cleaning NaN values
#         payload = {
#             "model_metrics": {
#                 "validation": {
#                     "rmse": float(np.nan_to_num(val_rmse, nan=0.0)),
#                     "mae": float(np.nan_to_num(val_mae, nan=0.0)),
#                     "r2_score": float(np.nan_to_num(val_r2, nan=0.0))
#                 },
#                 "assessment": model_assessment
#             },
#             "model_metadata": {
#                 "model_type": model_type,
#                 "hyperparameters": best_params if best_params else {},
#                 "training_samples": training_samples,
#                 "num_features": num_features,
#                 "evaluation_duration": float(np.nan_to_num(eval_duration, nan=0.0)),
#                 "timestamp": pd.Timestamp.now().isoformat()
#             },
#             "data_characteristics": {
#                 "actual_distribution": actual_stats,
#                 "predicted_distribution": predicted_stats,
#                 "feature_correlations": feature_correlation
#             },
#             "feature_analysis": {
#                 "attribute_columns": list(X_train.columns),
#                 "feature_importance": {k: float(np.nan_to_num(v, nan=0.0)) for k, v in feature_importance.items()},
#                 "top_features": top_features,
#                 "shap_importance": {k: float(np.nan_to_num(v, nan=0.0)) for k, v in (shap_importance or {}).items()}
#             },
#             "core_statistics": core_statistics,
#             "attribute_statistics": attribute_statistics,
#             "predictions": predictions_data,  # Add predictions field with actual and predicted values
#             "user_id": user_id,
#             "chat_id": chat_id
#         }

#         # Clean NaN values in the payload
#         cleaned_payload = clean_nan_values(payload)
#         #import pdb; pdb.set_trace()

#         # Post results to API
#         api_url = "http://127.0.0.1:8000/model/modelresults/"
#         headers = {"Content-Type": "application/json"}
#         try:
#             response = requests.post(api_url, data=json.dumps(cleaned_payload), headers=headers)
#             if response.status_code == 201:
#                 logger.info("Data successfully posted to the Django API.")
#             else:
#                 logger.warning(f"API post failed: {response.status_code} {response.text}")
#                 print(f"API post failed: {response.status_code} {response.text}")
#         except Exception as post_err:
#             logger.error(f"Error posting data to API: {post_err}")

#         # Save the model to S3
#         logger.info("Saving final model to S3...")
#         bucket_name = "artifacts1137"
#         prefix = f"ml-artifacts/{chat_id}/"
#         model_key = f"final_model.joblib"
#         with io.BytesIO() as f:
#             joblib.dump(final_model, f)
#             f.seek(0)
#             upload_to_s3(f, bucket_name, f"{prefix}{model_key}")
#         logger.info(f"Final model saved to s3://{bucket_name}/{prefix}{model_key}")

#         # Return final metrics for logging
#         final_metrics = {
#             'Validation': {
#                 'RMSE': float(np.nan_to_num(val_rmse, nan=0.0)),
#                 'MAE': float(np.nan_to_num(val_mae, nan=0.0)),
#                 'R2': float(np.nan_to_num(val_r2, nan=0.0))
#             },
#             'Assessment': model_assessment
#         }
#         return final_model, final_metrics

#     except Exception as e:
#         logger.error(f"Error during final time-series model evaluation: {e}")
#         raise

import time
import json
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import requests
import io
import joblib
from src.logging_config import get_logger
import shap  # For feature importance (optional)

logger = get_logger(__name__)

def clean_nan_values(data):
    """
    Recursively clean NaN, inf, and non-JSON-serializable values from a dictionary or list.
    Converts Timestamp objects to ISO strings and handles nested structures.
    """
    if isinstance(data, (dict, list)):
        return [
            clean_nan_values(item) if isinstance(item, (dict, list))
            else item.isoformat() if isinstance(item, pd.Timestamp)
            else None if pd.isna(item) or (isinstance(item, float) and np.isnan(item))
            else 0.0 if isinstance(item, float) and np.isinf(item) else item
            for item in (data if isinstance(data, list) else data.items())
        ] if isinstance(data, list) else {
            key: clean_nan_values(value)
            for key, value in data.items()
        }
    return data.isoformat() if isinstance(data, pd.Timestamp) else \
           None if pd.isna(data) or (isinstance(data, float) and np.isnan(data)) else \
           0.0 if isinstance(data, float) and np.isinf(data) else data

def finalize_and_evaluate_model_timeseries(final_model, X_train, predictions_df, user_id, chat_id, best_params=None, entity_column=None):
    """
    Finalizes the time-series model by saving it, and generates metadata using validation predictions
    from the pipeline. Posts results to an API, including predictions with product_id and date.

    Parameters:
    - final_model: The trained model instance from the pipeline.
    - X_train (pd.DataFrame): Training features (full dataset up to cutoff, for metadata).
    - predictions_df (pd.DataFrame): Validation predictions from the pipeline (with 'actual', 'predicted', 'product_id', and 'analysis_time' columns).
    - user_id (str): User identifier for logging and API.
    - chat_id (str): Chat identifier for tracking and storing artifacts.

    Returns:
    - final_model: The finalized model instance (unchanged).
    - final_metrics (dict): Aggregated validation metrics (RMSE, MAE, R2) and assessment.
    """
    try:
        logger.info("Finalizing the time-series model and generating metadata from validation predictions...")
        eval_start = time.time()  # Start evaluation timer

        # Use validation predictions from the pipeline for metrics and predictions
        if predictions_df.empty:
            logger.warning("No validation predictions provided; using training data as fallback.")
            y_val_actual = X_train[target_column] if 'target_column' in globals() else X_train.iloc[:, -1]  # Fallback logic
            y_val_pred = final_model.predict(X_train)
            val_product_ids = X_train.index if entity_column in X_train else [f"prod_{i}" for i in range(X_train.shape[0])]
            val_dates = pd.Series([pd.Timestamp.now()] * X_train.shape[0], index=X_train.index).apply(lambda x: x.isoformat())
        else:
            y_val_actual = predictions_df['actual'].dropna()
            y_val_pred = predictions_df['predicted'].dropna()
            val_product_ids = predictions_df[entity_column].dropna()
            val_dates = pd.to_datetime(predictions_df['analysis_time']).dropna().apply(lambda x: x.isoformat())

        # Ensure aligned lengths
        min_length = min(len(y_val_actual), len(y_val_pred), len(val_product_ids), len(val_dates))
        y_val_actual = y_val_actual.iloc[:min_length].reset_index(drop=True)
        y_val_pred = y_val_pred.iloc[:min_length].reset_index(drop=True)
        val_product_ids = val_product_ids.iloc[:min_length].reset_index(drop=True)
        val_dates = val_dates.iloc[:min_length].reset_index(drop=True)

        # Calculate aggregated validation metrics
        val_rmse = np.sqrt(mean_squared_error(y_val_actual, y_val_pred))
        val_mae = mean_absolute_error(y_val_actual, y_val_pred)
        val_r2 = r2_score(y_val_actual, y_val_pred) if len(y_val_actual) > 1 else 0.0

        logger.info(f"Aggregated Validation RMSE: {val_rmse:.4f}, MAE: {val_mae:.4f}, R^2: {val_r2:.4f}")

        # Enhanced model assessment
        target_mean = np.nan_to_num(np.mean(y_val_actual), nan=0.0)
        rmse_threshold = target_mean * 0.2  # 20% of mean as threshold
        r2_threshold_low = 0.5
        r2_threshold_high = 0.7
        model_assessment = "Good Fit" if val_r2 > r2_threshold_high and val_rmse <= rmse_threshold else \
                          "Overfitting" if val_r2 > 0.9 and (val_rmse > rmse_threshold or len(y_val_actual) < X_train.shape[0] * 0.1) else \
                          "Underfitting" if val_r2 < r2_threshold_low or val_rmse > target_mean else \
                          "Check Further"
        logger.info(f"Model assessment: {model_assessment} (R²: {val_r2:.4f}, RMSE: {val_rmse:.4f}, Target Mean: {target_mean:.4f})")

        # Model metadata including testing samples
        model_type = final_model.__class__.__name__
        training_samples = X_train.shape[0]
        testing_samples = len(y_val_actual)  # Number of validation samples
        num_features = X_train.shape[1]
        eval_duration = time.time() - eval_start

        # Feature importance (if available)
        feature_importance = {}
        if hasattr(final_model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, final_model.feature_importances_))
        elif hasattr(final_model, 'coef_'):
            if len(final_model.coef_.shape) > 1:
                coefs = final_model.coef_[0]  # Handle multi-class
            else:
                coefs = final_model.coef_
            feature_importance = dict(zip(X_train.columns, coefs))

        # Top 10 features
        top_n = 10
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        top_features = {k: float(v) for k, v in sorted_features[:top_n]}

        # SHAP importance (optional, for advanced analysis)
        shap_importance = {}
        try:
            sample_size = min(100, X_train.shape[0])
            X_sample = X_train.sample(sample_size, random_state=42)
            explainer = shap.Explainer(final_model, X_sample)
            shap_values = explainer(X_sample)
            
            shap_importance = pd.DataFrame({
                'feature': X_train.columns,
                'mean_abs_shap': np.abs(shap_values.values).mean(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
            
            shap_importance = {
                k: float(v) 
                for k, v in shap_importance.set_index('feature')['mean_abs_shap'].items()
            }
        except Exception as e:
            logger.warning(f"SHAP calculation failed: {e}")

        # Data characteristics from validation data
        actual_stats = {
            'mean': float(np.nan_to_num(np.mean(y_val_actual), nan=0.0)),
            'std': float(np.nan_to_num(np.std(y_val_actual), nan=0.0)) if len(y_val_actual) > 1 else 0.0,
            'min': float(np.nan_to_num(np.min(y_val_actual), nan=0.0)),
            'max': float(np.nan_to_num(np.max(y_val_actual), nan=0.0))
        }
        
        predicted_stats = {
            'mean': float(np.nan_to_num(np.mean(y_val_pred), nan=0.0)),
            'std': float(np.nan_to_num(np.std(y_val_pred), nan=0.0)) if len(y_val_pred) > 1 else 0.0,
            'min': float(np.nan_to_num(np.min(y_val_pred), nan=0.0)),
            'max': float(np.nan_to_num(np.max(y_val_pred), nan=0.0))
        }

        # Feature correlations with target (using training data for stability, cleaning NaN)
        feature_correlation = X_train.corrwith(pd.Series(y_val_actual, index=X_train.index)).to_dict()
        feature_correlation = {k: float(np.nan_to_num(v, nan=0.0)) for k, v in feature_correlation.items()}

        # Core and attribute statistics (from training data, cleaning NaN)
        core_statistics = X_train.describe().to_dict()
        core_statistics = {
            k: {stat: float(np.nan_to_num(val, nan=0.0)) if isinstance(val, (np.integer, np.floating)) else val 
                for stat, val in stats.items()}
            for k, stats in core_statistics.items()
        }
        
        attribute_statistics = X_train.describe(include='all').to_dict()
        attribute_statistics = {
            k: {stat: float(np.nan_to_num(val, nan=0.0)) if isinstance(val, (np.integer, np.floating)) else val 
                for stat, val in stats.items()}
            for k, stats in attribute_statistics.items()
        }

        # Prepare predictions for the payload with product_id and analysis_time
        predictions_data = {
            "entity_colum": val_product_ids.tolist(),
            'analysis_time': val_dates.tolist(),
            'actual': y_val_actual.tolist(),
            'predicted': y_val_pred.tolist()
        }

        # Build comprehensive payload, cleaning NaN values
        payload = {
            "model_metrics": {
                "validation": {
                    "rmse": float(np.nan_to_num(val_rmse, nan=0.0)),
                    "mae": float(np.nan_to_num(val_mae, nan=0.0)),
                    "r2_score": float(np.nan_to_num(val_r2, nan=0.0))
                },
                "assessment": model_assessment
            },
            "model_metadata": {
                "model_type": model_type,
                "hyperparameters": best_params if best_params else {},
                "training_samples": training_samples,
                "testing_samples": testing_samples,  # Added testing samples
                "num_features": num_features,
                "evaluation_duration": float(np.nan_to_num(eval_duration, nan=0.0)),
                "timestamp": pd.Timestamp.now().isoformat()  # Ensure string format
            },
            "data_characteristics": {
                "actual_distribution": actual_stats,
                "predicted_distribution": predicted_stats,
                "feature_correlations": feature_correlation
            },
            "feature_analysis": {
                "attribute_columns": list(X_train.columns),
                "feature_importance": {k: float(np.nan_to_num(v, nan=0.0)) for k, v in feature_importance.items()},
                "top_features": top_features,
                "shap_importance": {k: float(np.nan_to_num(v, nan=0.0)) for k, v in (shap_importance or {}).items()}
            },
            "core_statistics": core_statistics,
            "attribute_statistics": attribute_statistics,
            "predictions": predictions_data,
            "user_id": user_id,
            "chat_id": chat_id
        }

        # Clean NaN values and ensure JSON serialization
        cleaned_payload = clean_nan_values(payload)
        logger.debug(f"Cleaned payload: {json.dumps(cleaned_payload)}")  # Debug log

        # Post results to API
        api_url = "http://127.0.0.1:8000/model/modelresults/"
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(api_url, data=json.dumps(cleaned_payload), headers=headers)
            if response.status_code == 201:
                logger.info("Data successfully posted to the Django API.")
            else:
                logger.warning(f"API post failed: {response.status_code} {response.text}")
                print(f"API post failed: {response.status_code} {response.text}")
        except Exception as post_err:
            logger.error(f"Error posting data to API: {post_err}")
            raise  # Re-raise to stop pipeline on failure

        # Save the model to S3
        logger.info("Saving final model to S3...")
        bucket_name = "artifacts1137"
        prefix = f"ml-artifacts/{chat_id}/"
        model_key = f"final_model.joblib"
        with io.BytesIO() as f:
            joblib.dump(final_model, f)
            f.seek(0)
            upload_to_s3(f, bucket_name, f"{prefix}{model_key}")
        logger.info(f"Final model saved to s3://{bucket_name}/{prefix}{model_key}")

        # Return final metrics for logging
        final_metrics = {
            'Validation': {
                'RMSE': float(np.nan_to_num(val_rmse, nan=0.0)),
                'MAE': float(np.nan_to_num(val_mae, nan=0.0)),
                'R2': float(np.nan_to_num(val_r2, nan=0.0))
            },
            'Assessment': model_assessment
        }
        return final_model, final_metrics

    except Exception as e:
        logger.error(f"Error during final time-series model evaluation: {e}")
        raise