


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