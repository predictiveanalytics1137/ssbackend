# # final evaluation

# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import mean_squared_error, r2_score
# import numpy as np
# from src.logging_config import get_logger
# from sklearn.metrics import mean_absolute_error
# import requests
# import json
# # import matplotlib
# # matplotlib.use('Agg')


# logger = get_logger(__name__)
# logger.info("src.finalization module loaded")


# # def finalize_and_evaluate_model(best_model_class, best_params, X_train, y_train, X_test, y_test):
# #     """
# #     Finalizes the model by training on the best hyperparameters, evaluates it, 
# #     extracts feature importance, and saves the trained model.
    
# #     Parameters:
# #     - best_model_class: class of the best model (e.g., XGBRegressor or RandomForestRegressor).
# #     - best_params: dictionary of the best hyperparameters.
# #     - X_train, y_train: training data and labels.
# #     - X_test, y_test: testing data and labels.
    
# #     Returns:
# #     - final_metrics: dictionary containing RMSE and R-squared metrics.
# #     """
# #     try:
# #         logger.info("Initializing the model with best hyperparameters...")
# #         best_model = best_model_class(**best_params)

# #         # Train the final model
# #         logger.info("Training the model with best hyperparameters...")
# #         best_model.fit(X_train, y_train)

# #         # Predict on the test set
# #         logger.info("Predicting on the test set...")
# #         y_pred = best_model.predict(X_test)

# #         # Evaluate the model performance
# #         logger.info("Evaluating model performance...")
# #         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# #         r2 = r2_score(y_test, y_pred)
        
# #         logger.info(f"RMSE of the final model: {rmse}")
# #         logger.info(f"R-squared of the final model: {r2}")
        
# #         final_metrics = {'RMSE': rmse, 'R-squared': r2}

# #         # Feature Importance (for tree-based models)
# #         if hasattr(best_model, 'feature_importances_'):
# #             logger.info("Extracting feature importances...")
# #             feature_importance = best_model.feature_importances_
# #             sorted_idx = np.argsort(feature_importance)[::-1]

# #             # Plotting Feature Importance
# #             logger.info("Plotting feature importances...")
# #             plt.figure(figsize=(12, 8))
# #             sns.barplot(x=feature_importance[sorted_idx], y=np.array(X_train.columns)[sorted_idx], palette='viridis')
# #             plt.title("Feature Importances")
# #             plt.xlabel("Importance Score")
# #             plt.ylabel("Features")
# #             plt.show()

# #         # Save the Model
# #         # model_filename = 'best_model.joblib'
# #         # joblib.dump(best_model, model_filename)
# #         # logger.info(f"Model saved as {model_filename}")

# #         return final_metrics

# #     except Exception as e:
# #         logger.error(f"Error during final model evaluation: {e}")
# #         raise


# # def finalize_and_evaluate_model(best_model_class, best_params, X_train, y_train, X_test, y_test):
# #     """
# #     Finalizes the model by training on the best hyperparameters, evaluates it, 
# #     extracts feature importance, and saves the trained model.
    
# #     Parameters:
# #     - best_model_class: class of the best model (e.g., XGBRegressor or RandomForestRegressor).
# #     - best_params: dictionary of the best hyperparameters.
# #     - X_train, y_train: training data and labels.
# #     - X_test, y_test: testing data and labels.
    
# #     Returns:
# #     - final_metrics: dictionary containing RMSE and R-squared metrics.
# #     """
# #     try:
# #         logger.info("Initializing the model with best hyperparameters...")
# #         best_model = best_model_class(**best_params)

# #         # Train the final model
# #         logger.info("Training the model with best hyperparameters...")
# #         best_model.fit(X_train, y_train)

# #         # Predict on the test set
# #         logger.info("Predicting on the test set...")
# #         y_pred = best_model.predict(X_test)

# #         # Evaluate the model performance
# #         logger.info("Evaluating model performance...")
# #         rmse = np.sqrt(mean_squared_error(y_test, y_pred))
# #         r2 = r2_score(y_test, y_pred)
# #         mae = mean_absolute_error(y_test, y_pred)
        
# #         logger.info(f"RMSE of the final model: {rmse}")
# #         logger.info(f"R-squared of the final model: {r2}")
        
# #         final_metrics = {'RMSE': rmse, 'R-squared': r2}

# #         # Feature Importance (for tree-based models)
# #         feature_importance = {}
# #         # Core statistics
# #         core_statistics = X_train.describe().to_dict()

# #         # Attribute table statistics
# #         attribute_statistics = X_train.describe(include='all').to_dict()
        
        
        
        
# #         if hasattr(best_model, 'feature_importances_'):
# #             feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))
# #             logger.info("Extracting feature importances...")
# #             feature_importance = best_model.feature_importances_
# #             sorted_idx = np.argsort(feature_importance)[::-1]

# #             # Plotting Feature Importance
# #             logger.info("Plotting feature importances...")
# #             plt.figure(figsize=(12, 8))
# #             sns.barplot(x=feature_importance[sorted_idx], y=np.array(X_train.columns)[sorted_idx], palette='viridis')
# #             plt.title("Feature Importances")
# #             plt.xlabel("Importance Score")
# #             plt.ylabel("Features")
# #             plt.show()

# #         # # Save the Model
# #         # model_filename = 'best_model.joblib'
# #         # joblib.dump(best_model, model_filename)
# #         # logger.info(f"Model saved as {model_filename}")
        
# #         # Prepare the payload
# #         payload = {
# #             "model_metrics": {
# #                 "rmse": rmse,
# #                 "r2_score": r2,
# #                 "mae": mae
# #             },
# #             "attribute_columns": list(X_train.columns),
# #             "feature_importance": feature_importance,
# #             "core_statistics": core_statistics,
# #             "attribute_statistics": attribute_statistics,
# #             "predictions": {
# #                 "actual": y_test.tolist(),
# #                 "predicted": y_pred.tolist()
# #             }
# #         }
        
# #         # POST the data to Django API
# #         api_url = "http://127.0.0.1:8000/model/modelresults/"
# #         headers = {"Content-Type": "application/json"}
# #         response = requests.post(api_url, data=json.dumps(payload), headers=headers)

# #         #if response.status_code == 201:
# #         #    logger.info("Data successfully posted to the Django API.")
# #         #else:
# #         #    logger.error(f"Failed to post data. Status code: {response.status_code}, Response: {response.text}")


# #         #return final_metrics
# #         print({"RMSE": rmse, "R2": r2, "MAE": mae})
# #         return final_metrics

# #     except Exception as e:
# #         logger.error(f"Error during final model evaluation: {e}")
# #         raise




# def finalize_and_evaluate_model(best_model_class, best_params, X_train, y_train, X_test, y_test):
#     """
#     Finalizes the model by training on the best hyperparameters, evaluates it, 
#     extracts feature importance, and saves the trained model.
    
#     Parameters:
#     - best_model_class: class of the best model (e.g., XGBRegressor or RandomForestRegressor).
#     - best_params: dictionary of the best hyperparameters.
#     - X_train, y_train: training data and labels.
#     - X_test, y_test: testing data and labels.
    
#     Returns:
#     - final_metrics: dictionary containing RMSE, R-squared metrics, and model assessment.
#     """
#     try:
#         logger.info("Initializing the model with best hyperparameters...")
#         best_model = best_model_class(**best_params)

#         # Train the final model
#         logger.info("Training the model with best hyperparameters...")
#         best_model.fit(X_train, y_train)

#         # Predict on the test set
#         logger.info("Predicting on the test set...")
#         y_test_pred = best_model.predict(X_test)

#         # Predict on the training set
#         logger.info("Predicting on the training set...")
#         y_train_pred = best_model.predict(X_train)

#         # Evaluate the model performance
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

#         logger.info(f"Model assessment based on R² scores: {model_assessment}")

#         final_metrics = {
#             'Testing': {'RMSE': test_rmse, 'R-squared': test_r2, 'MAE': test_mae},
#             'Training': {'RMSE': train_rmse, 'R-squared': train_r2, 'MAE': train_mae},
#             'Assessment': model_assessment
#         }

#         final_metrics = {'RMSE': train_rmse, 'R-squared': train_r2}

#         # Feature Importance (for tree-based models)
#         feature_importance = {}
#         if hasattr(best_model, 'feature_importances_'):
#             feature_importance = dict(zip(X_train.columns, best_model.feature_importances_))

#         # Core statistics
#         core_statistics = X_train.describe().to_dict()

#         # Attribute table statistics
#         attribute_statistics = X_train.describe(include='all').to_dict()

#         # Prepare the payload
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
#             "attribute_columns": list(X_train.columns),  # Ensure it’s a Python list
#             "feature_importance": {k: float(v) for k, v in feature_importance.items()},  # Convert importance scores to float
#             "core_statistics": {
#                 k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val for stat, val in stats.items()}
#                 for k, stats in core_statistics.items()
#             },
#             "attribute_statistics": {
#                 k: {stat: float(val) if isinstance(val, (np.integer, np.floating)) else val for stat, val in stats.items()}
#                 for k, stats in attribute_statistics.items()
#             },
#             "predictions": {
#                 "actual": y_test.tolist(),  # Convert ndarray to list
#                 "predicted": y_test_pred.tolist()  # Convert ndarray to list
#             }
#         }

#         # POST the data to Django API
#         api_url = "http://127.0.0.1:8000/model/modelresults/"
#         headers = {"Content-Type": "application/json"}
#         response = requests.post(api_url, data=json.dumps(payload), headers=headers)

#         if response.status_code == 201:
#             logger.info("Data successfully posted to the Django API.")
#         else:
#             logger.error(f"Failed to post data. Status code: {response.status_code}, Response: {response.text}")

#         # Return final metrics
#         logger.info("Final metrics computed successfully.")
#         return final_metrics

#     except Exception as e:
#         logger.error(f"Error during final model evaluation: {e}")
#         raise






# src/finalization.py
import requests
import json
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from src.logging_config import get_logger

logger = get_logger(__name__)

def finalize_and_evaluate_model(best_model_class, best_params, X_train, y_train, X_test, y_test, user_id, chat_id):
    """
    Finalizes the model, evaluates it, and posts results along with user_id and chat_id.
    """
    try:
        logger.info("Initializing the model with best hyperparameters...")
        best_model = best_model_class(**best_params)

        logger.info("Training the model with best hyperparameters...")
        best_model.fit(X_train, y_train)

        logger.info("Predicting on the test set...")
        y_test_pred = best_model.predict(X_test)

        logger.info("Predicting on the training set...")
        y_train_pred = best_model.predict(X_train)

        logger.info("Evaluating model performance on testing data...")
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        test_mae = mean_absolute_error(y_test, y_test_pred)

        logger.info("Evaluating model performance on training data...")
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)

        logger.info(f"Testing - RMSE: {test_rmse}, R²: {test_r2}, MAE: {test_mae}")
        logger.info(f"Training - RMSE: {train_rmse}, R²: {train_r2}, MAE: {train_mae}")

        # Determine Model Performance
        r2_difference = abs(train_r2 - test_r2)
        if train_r2 > 0.9 and test_r2 > 0.9 and r2_difference < 0.05:
            model_assessment = "Good Fit"
        elif train_r2 > 0.9 and test_r2 < 0.7:
            model_assessment = "Overfitting"
        elif train_r2 < 0.7 and test_r2 < 0.7:
            model_assessment = "Underfitting"
        else:
            model_assessment = "Potential Overfitting or Issues"

        logger.info(f"Model assessment: {model_assessment}")

        # final_metrics not used directly below, but we return it
        final_metrics = {
            'Testing': {'RMSE': test_rmse, 'R-squared': test_r2, 'MAE': test_mae},
            'Training': {'RMSE': train_rmse, 'R-squared': train_r2, 'MAE': train_mae},
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

        api_url = "http://127.0.0.1:8000/model/modelresults/"
        headers = {"Content-Type": "application/json"}
        response = requests.post(api_url, data=json.dumps(payload), headers=headers)

        if response.status_code == 201:
            logger.info("Data successfully posted to the Django API with {user_id} and chat_id.")
        else:
            logger.error(f"Failed to post data. Status code: {response.status_code}, Response: {response.text}")

        logger.info("Final metrics computed successfully.")
        return final_metrics

    except Exception as e:
        logger.error(f"Error during final model evaluation: {e}")
        raise
