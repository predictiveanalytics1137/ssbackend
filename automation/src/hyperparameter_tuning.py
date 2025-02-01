# hyper parameter tuning
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor, AdaBoostClassifier, AdaBoostRegressor, ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

from sklearn.model_selection import GridSearchCV

from src.logging_config import get_logger
logger = get_logger(__name__)


# hyper parameter tuning

logger = get_logger(__name__)


# def hyperparameter_tuning(best_model_name, X_train, y_train, X_test, y_test, task='regression'):
#     """
#     Automatically performs hyperparameter tuning for the best-performing model.
    
#     Parameters:
#     - best_model_name: string, the name of the best-performing model (from model selection).
#     - X_train, y_train: training features and target.
#     - X_test, y_test: testing features and target.
#     - task: string, 'classification' or 'regression', depending on the task type.
    
#     Returns:
#     - best_model: the best-tuned model.
#     - best_params: the best hyperparameters.
#     """
#     logger.info(f"Starting hyperparameter tuning for {best_model_name}...")
    
#     try:
#         # Define hyperparameter grids for different models
#         param_grids = {
#             'XGBoost': {
#                 'n_estimators': [100, 500],  
#                 'max_depth': [3, 7],        
#                 'learning_rate': [0.01, 0.1],
#                 'subsample': [0.8, 1.0]
#             },
#             'Random Forest': {
#                 'n_estimators': [100, 200, 500],
#                 'max_depth': [None, 10, 20, 30],
#                 'min_samples_split': [2, 5, 10],
#                 'min_samples_leaf': [1, 2, 4],
#                 'bootstrap': [True, False]
#             },
#             'LightGBM': {
#                 'n_estimators': [100, 200, 500],
#                 'max_depth': [-1, 10, 20, 30],
#                 'learning_rate': [0.01, 0.1, 0.2],
#                 'num_leaves': [31, 50, 100],
#                 'subsample': [0.6, 0.8, 1.0]
#             },
#             'Gradient Boosting': {
#                 'n_estimators': [100, 200, 500],
#                 'max_depth': [3, 5, 7, 10],
#                 'learning_rate': [0.01, 0.1, 0.2],
#                 'subsample': [0.6, 0.8, 1.0]
#             },
#             'KNN': {
#                 'n_neighbors': [3, 5, 10, 20],
#                 'weights': ['uniform', 'distance'],
#                 'metric': ['euclidean', 'manhattan']
#             },
#             'SVR': {
#                 'kernel': ['linear', 'poly'],
#                 'C': [0.1, 1],
#                 'epsilon': [0.1, 0.2]
#             },
#             'Linear Regression': {}
#         }
        
#         # Define models mapping to the best model names
#         models = {
#             'XGBoost': XGBRegressor(random_state=42) if task == 'regression' else XGBClassifier(random_state=42),
#             'Random Forest': RandomForestRegressor(random_state=42) if task == 'regression' else RandomForestClassifier(random_state=42),
#             'LightGBM': LGBMRegressor(random_state=42) if task == 'regression' else LGBMClassifier(random_state=42),
#             'Gradient Boosting': GradientBoostingRegressor(random_state=42) if task == 'regression' else GradientBoostingClassifier(random_state=42),
#             'KNN': KNeighborsRegressor() if task == 'regression' else KNeighborsClassifier(),
#             'SVR': SVR(),
#             'Linear Regression': LinearRegression()
#         }
        
#         # Select the best model based on the name
#         best_model = models[best_model_name]
#         param_grid = param_grids[best_model_name]

#         # If Linear Regression, skip hyperparameter tuning
#         if best_model_name == 'Linear Regression':
#             best_model.fit(X_train, y_train)
#             logger.info("Linear Regression does not have hyperparameters to tune.")
#             y_pred = best_model.predict(X_test)
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             logger.info(f"RMSE of the Linear Regression model: {rmse}")
#             return best_model, {}
        
#         # Initialize GridSearchCV
#         grid_search = GridSearchCV(
#             estimator=best_model, 
#             param_grid=param_grid, 
#             cv=5, 
#             scoring='neg_mean_squared_error' if task == 'regression' else 'accuracy', 
#             verbose=1, 
#             n_jobs=-1
#         )

#         # Fit GridSearchCV to the data
#         logger.info(f"Performing hyperparameter tuning for {best_model_name}...")
#         grid_search.fit(X_train, y_train)
        
#         # Get the best parameters and model
#         best_params = grid_search.best_params_
#         best_model = grid_search.best_estimator_
#         logger.info(f"Best hyperparameters for {best_model_name}: {best_params}")

#         # Predict and evaluate the tuned model on the test set
#         y_pred = best_model.predict(X_test)
#         if task == 'regression':
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             logger.info(f"RMSE of the tuned {best_model_name} model: {rmse}")
#         else:
#             accuracy = accuracy_score(y_test, y_pred)
#             logger.info(f"Accuracy of the tuned {best_model_name} model: {accuracy}")

#         return best_model, best_params

#     except Exception as e:
#         logger.error(f"Error during hyperparameter tuning: {e}")
#         raise


# v2
# =============================================================================
# 
# import numpy as np
# import pandas as pd
# from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import mean_squared_error, accuracy_score
# from xgboost import XGBRegressor, XGBClassifier
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
# from lightgbm import LGBMRegressor, LGBMClassifier
# from sklearn.linear_model import LinearRegression
# from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
# from sklearn.svm import SVR
# from src.logging_config import get_logger
# 
# logger = get_logger(__name__)
# 
# def hyperparameter_tuning(best_model_name, X_train, y_train, X_test, y_test, task='regression'):
#     """
#     Automatically performs hyperparameter tuning for the best-performing model.
#     """
#     logger.info(f"Starting hyperparameter tuning for {best_model_name}...")
# 
#     try:
#         param_grids = {
#             'XGBoost': {
#                 'n_estimators': [100, 500],
#                 'max_depth': [3, 7],
#                 'learning_rate': [0.01, 0.1],
#                 'subsample': [0.8, 1.0]
#             },
#             'Random Forest': {
#                 'n_estimators': [100, 200, 500],
#                 'max_depth': [None, 10, 20],
#                 'min_samples_split': [2, 5],
#                 'min_samples_leaf': [1, 2, 4],
#                 'bootstrap': [True, False]
#             },
#             'LightGBM': {
#                 'n_estimators': [100, 200],
#                 'max_depth': [-1, 10, 20],
#                 'learning_rate': [0.01, 0.1],
#                 'num_leaves': [31, 50],
#                 'subsample': [0.6, 1.0]
#             },
# # =============================================================================
# #             'Gradient Boosting': {
# #                 'n_estimators': [100, 200],
# #                 'max_depth': [3, 5, 7],
# #                 'learning_rate': [0.01, 0.1],
# #                 'subsample': [0.6, 1.0]
# #             },
# # =============================================================================
#             'Gradient Boosting': {
#                              'n_estimators': [100],
#                              'max_depth': [3],
#                              'learning_rate': [0.01]
#                          },
#             'KNN': {
#                 'n_neighbors': [3, 5, 10],
#                 'weights': ['uniform', 'distance'],
#                 'metric': ['euclidean', 'manhattan']
#             },
#             'SVR': {
#                 'kernel': ['linear', 'poly'],
#                 'C': [0.1, 1],
#                 'epsilon': [0.1, 0.2]
#             },
#             'Linear Regression': {}
#         }
# 
#         models = {
#             'XGBoost': XGBRegressor(random_state=42) if task == 'regression' else XGBClassifier(random_state=42),
#             'Random Forest': RandomForestRegressor(random_state=42) if task == 'regression' else RandomForestClassifier(random_state=42),
#             'LightGBM': LGBMRegressor(random_state=42) if task == 'regression' else LGBMClassifier(random_state=42),
#             'Gradient Boosting': GradientBoostingRegressor(random_state=42) if task == 'regression' else GradientBoostingClassifier(random_state=42),
#             'KNN': KNeighborsRegressor() if task == 'regression' else KNeighborsClassifier(),
#             'SVR': SVR(),
#             'Linear Regression': LinearRegression()
#         }
# 
#         best_model = models[best_model_name]
#         param_grid = param_grids[best_model_name]
# 
#         # If linear regression, skip hyperparameter tuning
#         if best_model_name == 'Linear Regression':
#             best_model.fit(X_train, y_train)
#             logger.info("Linear Regression has no parameters to tune.")
#             return best_model, {}
# 
#         scoring = 'neg_mean_squared_error' if task == 'regression' else 'accuracy'
#         grid_search = GridSearchCV(
#             estimator=best_model,
#             param_grid=param_grid,
#             cv=5,
#             scoring=scoring,
#             verbose=1,
#             n_jobs=-1
#         )
#         grid_search.fit(X_train, y_train)
# 
#         best_params = grid_search.best_params_
#         best_model = grid_search.best_estimator_
#         logger.info(f"Best hyperparameters for {best_model_name}: {best_params}")
# 
#         # Evaluate
#         y_pred = best_model.predict(X_test)
#         if task == 'regression':
#             rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#             logger.info(f"RMSE of tuned {best_model_name}: {rmse:.4f}")
#         else:
#             acc = accuracy_score(y_test, y_pred)
#             logger.info(f"Accuracy of tuned {best_model_name}: {acc:.4f}")
# 
#         return best_model, best_params
# 
#     except Exception as e:
#         logger.error(f"Error during hyperparameter tuning: {e}")
#         raise
# 
# =============================================================================







#v3




import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, accuracy_score
from xgboost import XGBRegressor, XGBClassifier
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
    ExtraTreesRegressor, ExtraTreesClassifier
)
from lightgbm import LGBMRegressor, LGBMClassifier
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.svm import SVR
from src.logging_config import get_logger

logger = get_logger(__name__)

def hyperparameter_tuning(best_model_name, X_train, y_train, X_test, y_test, task='regression'):
    """
    Automatically performs hyperparameter tuning for the best-performing model.
    
    Parameters:
    - best_model_name (str): Name of the model to tune.
    - X_train (pd.DataFrame): Training features.
    - y_train (pd.Series): Training target.
    - X_test (pd.DataFrame): Testing features.
    - y_test (pd.Series): Testing target.
    - task (str): 'regression' or 'classification'.
    
    Returns:
    - best_model: The model with the best found parameters.
    - best_params (dict): Best hyperparameters found.
    """
    logger.info(f"Starting hyperparameter tuning for {best_model_name}...")

    try:
        # Define hyperparameter grids for each model
        param_grids = {
            'XGBoost': {
                'n_estimators': [100, 500],
                'max_depth': [3, 7],
                'learning_rate': [0.01, 0.1],
                'subsample': [0.8, 1.0]
            },
            'Random Forest': {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False]
            },
            'Extra Trees': {  # Added Extra Trees hyperparameters
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'criterion': ['gini', 'entropy'] if task == 'classification' else ['squared_error']
            },
            'LightGBM': {
                'n_estimators': [100, 200],
                'max_depth': [-1, 10, 20],
                'learning_rate': [0.01, 0.1],
                'num_leaves': [31, 50],
                'subsample': [0.6, 1.0]
            },
            'Gradient Boosting': {
                'n_estimators': [100],
                'max_depth': [3],
                'learning_rate': [0.01]
            },
            'KNN': {
                'n_neighbors': [3, 5, 10],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'SVR': {
                'kernel': ['linear', 'poly'],
                'C': [0.1, 1],
                'epsilon': [0.1, 0.2]
            },
            'Linear Regression': {}
        }

        # Define models based on the task
        models = {
            'XGBoost': XGBRegressor(random_state=42) if task == 'regression' else XGBClassifier(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42) if task == 'regression' else RandomForestClassifier(random_state=42),
            'Extra Trees': ExtraTreesRegressor(random_state=42) if task == 'regression' else ExtraTreesClassifier(random_state=42),  # Added Extra Trees
            'LightGBM': LGBMRegressor(random_state=42) if task == 'regression' else LGBMClassifier(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42) if task == 'regression' else GradientBoostingClassifier(random_state=42),
            'KNN': KNeighborsRegressor() if task == 'regression' else KNeighborsClassifier(),
            'SVR': SVR(),
            'Linear Regression': LinearRegression()
        }

        # Ensure the model to be tuned exists
        if best_model_name not in models:
            raise ValueError(f"Model '{best_model_name}' is not supported for hyperparameter tuning.")

        best_model = models[best_model_name]
        param_grid = param_grids.get(best_model_name, {})

        # If linear regression, skip hyperparameter tuning
        if best_model_name == 'Linear Regression':
            best_model.fit(X_train, y_train)
            logger.info("Linear Regression has no parameters to tune.")
            return best_model, {}

        # Define scoring metric
        if task == 'regression':
            scoring = 'neg_mean_squared_error'
        elif task == 'classification':
            scoring = 'accuracy'
        else:
            raise ValueError("Task must be either 'regression' or 'classification'.")

        # Initialize GridSearchCV
        grid_search = GridSearchCV(
            estimator=best_model,
            param_grid=param_grid,
            cv=5,
            scoring=scoring,
            verbose=1,
            n_jobs=-1
        )

        # Fit GridSearchCV
        grid_search.fit(X_train, y_train)

        # Retrieve best parameters and estimator
        best_params = grid_search.best_params_
        best_model = grid_search.best_estimator_
        logger.info(f"Best hyperparameters for {best_model_name}: {best_params}")

        # Evaluate the best model on the test set
        y_pred = best_model.predict(X_test)
        if task == 'regression':
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            logger.info(f"RMSE of tuned {best_model_name}: {rmse:.4f}")
        else:
            acc = accuracy_score(y_test, y_pred)
            logger.info(f"Accuracy of tuned {best_model_name}: {acc:.4f}")

        return best_model, best_params

    except Exception as e:
        logger.error(f"Error during hyperparameter tuning: {e}")
        raise
