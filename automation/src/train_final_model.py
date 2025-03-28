import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, f1_score
import logging
import time
from catboost import CatBoostRegressor, CatBoostClassifier
from xgboost import XGBRegressor, XGBClassifier
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

logger = logging.getLogger(__name__)

def train_final_model(best_model_name, best_params, X_train, y_train, task='regression', early_stopping_rounds=10):
    """
    Trains the final model using the best model name and parameters on the full training data.
    Supports early stopping for boosting models to prevent overfitting.

    Parameters:
    - best_model_name (str): Name of the best model (e.g., 'XGBoost', 'Random Forest').
    - best_params (dict): Best hyperparameters from tuning.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - task (str): 'regression' or 'classification' (default: 'regression').
    - early_stopping_rounds (int): Number of rounds for early stopping (default: 10).

    Returns:
    - final_model: Trained model instance.
    """
    try:
        logger.info(f"Training final model: {best_model_name} with parameters: {best_params}")

        # Validate inputs
        if X_train is None or y_train is None:
            raise ValueError("X_train and y_train must not be None.")
        X_train = pd.DataFrame(X_train) if not isinstance(X_train, pd.DataFrame) else X_train
        y_train = pd.Series(y_train) if not isinstance(y_train, pd.Series) else y_train

        # Define model classes based on task
        model_classes = {
            'XGBoost': XGBRegressor if task == 'regression' else XGBClassifier,
            'Random Forest': RandomForestRegressor if task == 'regression' else RandomForestClassifier,
            'LightGBM': LGBMRegressor if task == 'regression' else LGBMClassifier,
            'CatBoost': CatBoostRegressor if task == 'regression' else CatBoostClassifier
        }

        if best_model_name not in model_classes:
            raise ValueError(f"Model '{best_model_name}' not supported.")

        # Initialize the model with best parameters
        model_class = model_classes[best_model_name]
        final_model = model_class(**best_params, random_state=42)

        # Handle early stopping for boosting models
        if best_model_name in ['XGBoost', 'LightGBM', 'CatBoost']:
            # Use a portion of training data as validation for early stopping
            val_size = int(0.1 * len(X_train))
            X_val = X_train.iloc[-val_size:]
            y_val = y_train.iloc[-val_size:]
            X_train_subset = X_train.iloc[:-val_size]
            y_train_subset = y_train.iloc[:-val_size]

            # Fit with early stopping
            start_time = time.time()
            if best_model_name == 'XGBoost':
                final_model.fit(
                    X_train_subset, y_train_subset,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=0
                )
            elif best_model_name == 'LightGBM':
                final_model.fit(
                    X_train_subset, y_train_subset,
                    eval_set=[(X_val, y_val)],
                    early_stopping_rounds=early_stopping_rounds,
                    verbose=-1
                )
            elif best_model_name == 'CatBoost':
                cat_features = [col for col in X_train.columns if X_train[col].dtype.name in ['category', 'object']]
                final_model.fit(
                    X_train_subset, y_train_subset,
                    eval_set=(X_val, y_val),
                    early_stopping_rounds=early_stopping_rounds,
                    cat_features=cat_features,
                    verbose=0
                )
            logger.info(f"Training with early stopping completed in {time.time() - start_time:.2f} seconds")
        else:
            # For non-boosting models, fit on full training data
            start_time = time.time()
            final_model.fit(X_train, y_train)
            logger.info(f"Training completed in {time.time() - start_time:.2f} seconds")

        # Evaluate on training data
        y_pred_train = final_model.predict(X_train)
        if task == 'regression':
            rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            logger.info(f"Final model training RMSE: {rmse:.4f}")
        else:
            f1 = f1_score(y_train, y_pred_train, average='weighted')
            logger.info(f"Final model training F1 Score: {f1:.4f}")

        return final_model

    except Exception as e:
        logger.error(f"Error during final model training: {e}")
        raise