# final evaluation

import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
from src.logging_config import get_logger

# import matplotlib
# matplotlib.use('Agg')


logger = get_logger(__name__)
logger.info("src.finalization module loaded")


def finalize_and_evaluate_model(best_model_class, best_params, X_train, y_train, X_test, y_test):
    """
    Finalizes the model by training on the best hyperparameters, evaluates it, 
    extracts feature importance, and saves the trained model.
    
    Parameters:
    - best_model_class: class of the best model (e.g., XGBRegressor or RandomForestRegressor).
    - best_params: dictionary of the best hyperparameters.
    - X_train, y_train: training data and labels.
    - X_test, y_test: testing data and labels.
    
    Returns:
    - final_metrics: dictionary containing RMSE and R-squared metrics.
    """
    try:
        logger.info("Initializing the model with best hyperparameters...")
        best_model = best_model_class(**best_params)

        # Train the final model
        logger.info("Training the model with best hyperparameters...")
        best_model.fit(X_train, y_train)

        # Predict on the test set
        logger.info("Predicting on the test set...")
        y_pred = best_model.predict(X_test)

        # Evaluate the model performance
        logger.info("Evaluating model performance...")
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"RMSE of the final model: {rmse}")
        logger.info(f"R-squared of the final model: {r2}")
        
        final_metrics = {'RMSE': rmse, 'R-squared': r2}

        # Feature Importance (for tree-based models)
        if hasattr(best_model, 'feature_importances_'):
            logger.info("Extracting feature importances...")
            feature_importance = best_model.feature_importances_
            sorted_idx = np.argsort(feature_importance)[::-1]

            # Plotting Feature Importance
            logger.info("Plotting feature importances...")
            plt.figure(figsize=(12, 8))
            sns.barplot(x=feature_importance[sorted_idx], y=np.array(X_train.columns)[sorted_idx], palette='viridis')
            plt.title("Feature Importances")
            plt.xlabel("Importance Score")
            plt.ylabel("Features")
            plt.show()

        # Save the Model
        # model_filename = 'best_model.joblib'
        # joblib.dump(best_model, model_filename)
        # logger.info(f"Model saved as {model_filename}")

        return final_metrics

    except Exception as e:
        logger.error(f"Error during final model evaluation: {e}")
        raise
