


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
import requests
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.logging_config import get_logger

logger = get_logger(__name__)

def finalize_and_evaluate_model(best_model_class, best_params, X_train, y_train, 
                                X_test, y_test, user_id, chat_id, test_ids=None):
    """
    Finalizes the model by instantiating it with the best hyperparams, then fits & evaluates.
    Returns final metrics and posts them to an API (stub).
    """
    try:
        logger.info("Initializing the model with best hyperparameters...")
        best_model = best_model_class(**best_params)

        logger.info("Fitting the model on training data...")
        best_model.fit(X_train, y_train)

        logger.info("Predicting on test set...")
        y_test_pred = best_model.predict(X_test)

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

        final_metrics = {
            'Training': {'RMSE': train_rmse, 'R2': train_r2, 'MAE': train_mae},
            'Testing': {'RMSE': test_rmse, 'R2': test_r2, 'MAE': test_mae},
            'Assessment': model_assessment
        }
        
        
        
        
        
        # Feature Importance
        feature_importance = {}
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))

        # Core statistics
        core_statistics = X_train.describe().to_dict()

        # Attribute table statistics
        attribute_statistics = X_train.describe(include='all').to_dict()

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
                "assessment": model_assessment
            },
            "attribute_columns": list(X_train.columns),
            "feature_importance": {k: float(v) for k, v in feature_importance.items()},
            "core_statistics": {
                k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val for stat, val in stats.items()}
                for k, stats in core_statistics.items()
            },
            "attribute_statistics": {
                k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val for stat, val in stats.items()}
                for k, stats in attribute_statistics.items()
            },
            "predictions": {
                "actual": y_test.tolist(),
                "predicted": y_test_pred.tolist()
            },
            # Include user_id and chat_id for uniqueness
            "user_id": user_id,
            "chat_id": chat_id
        }
        
        
        
        
        

# =============================================================================
#         # Example of posting results to an API (placeholder)
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
#             "user_id": user_id,
#             "chat_id": chat_id
#         }
# =============================================================================

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

        return final_metrics

    except Exception as e:
        logger.error(f"Error during final model evaluation: {e}")
        raise
