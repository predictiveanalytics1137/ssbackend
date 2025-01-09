# train test split and model selection

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
from src.logging_config import get_logger

logger = get_logger(__name__)


# def train_test_model_selection(df, target_column, id_column=None, task='regression'):
#     """
#     Splits the dataset into training and testing sets, and evaluates multiple machine learning algorithms.

#     Parameters:
#     - df: pandas DataFrame containing the dataset
#     - target_column: string, name of the target column
#     - id_column: string, name of the ID column (excluded from training features)
#     - task: string, either 'classification' or 'regression' depending on the problem type

#     Returns:
#     - best_model_name: string, name of the best-performing model
#     - X_train, X_test, y_train, y_test, test_ids: train-test split data with test IDs for prediction mapping
#     """
#     try:
#         logger.info("Starting train-test split and model selection...")

#         # 1. Clean up column names
#         df.columns = df.columns.str.replace('<', '', regex=True).str.replace('>', '', regex=True).str.replace(' ', '_', regex=True)
#         logger.info(f"Cleaned column names: {list(df.columns)}")

#         # 2. Separate ID column and target column
#         if id_column:
#             # Save IDs for filtering after the split
#             ids = df[id_column].copy()
#             df = df.drop(columns=[id_column])
#             logger.info(f"Excluded ID column '{id_column}' from training features.")

#         # 3. Split data into features (X) and target (y)
#         X = df.drop(columns=[target_column])
#         y = df[target_column]
#         logger.info(f"Splitting dataset into training and testing sets with target column: {target_column}")

#         # Stratified splitting for classification or regular splitting for regression
#         if task == 'classification':
#             stratify = y
#         else:
#             stratify = None

#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
#         logger.info(f"Train-test split completed. Training size: {X_train.shape[0]}, Testing size: {X_test.shape[0]}")

#         # Filter IDs to include only the ones corresponding to the test set
#         test_ids = ids.loc[X_test.index]  # Select IDs matching the test set indices

#         # 4. Define models for classification and regression
#         if task == 'classification':
#             models = {
#                 'Logistic Regression': LogisticRegression(),
#                 'KNN': KNeighborsClassifier(),
#                 'Decision Tree': DecisionTreeClassifier(),
#                 'Random Forest': RandomForestClassifier(),
#                 'Gradient Boosting': GradientBoostingClassifier(),
#                 'SVM': SVC(),
#                 'Naive Bayes': GaussianNB(),
#                 'XGBoost': XGBClassifier(),
#                 'AdaBoost': AdaBoostClassifier(),
#                 'Extra Trees': ExtraTreesClassifier(),
#                 'LightGBM': LGBMClassifier()
#             }
#             metric = accuracy_score
#         elif task == 'regression':
#             models = {
#                 'KNN': KNeighborsRegressor(),
#                 'Decision Tree': DecisionTreeRegressor(),
#                 'Random Forest': RandomForestRegressor(),
#                 'Gradient Boosting': GradientBoostingRegressor(),
#                 'SVR': SVR(),
#                 'XGBoost': XGBRegressor(),
#                 'AdaBoost': AdaBoostRegressor(),
#                 'Extra Trees': ExtraTreesRegressor(),
#                 'LightGBM': LGBMRegressor()
#             }
#             metric = mean_squared_error
#         else:
#             raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")

#         # 5. Train and evaluate each model
#         results = []
#         for name, model in models.items():
#             logger.info(f"Training {name}...")
#             model.fit(X_train, y_train)

#             # Exclude the ID column from X_test before predicting
#             y_pred = model.predict(X_test)
#             logger.info(f"Predicted with {name}.")

#             # Evaluate the model
#             if task == 'classification':
#                 score = accuracy_score(y_test, y_pred)
#             elif task == 'regression':
#                 score = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE

#             logger.info(f"Model: {name}, Score: {score}")
#             results.append({'Model': name, 'Score': score})

#         # 6. Select the best-performing model
#         results_df = pd.DataFrame(results).sort_values(by='Score', ascending=(task == 'regression'))  # Higher is better for classification
#         logger.info("\nModel Selection Complete. Results:")
#         logger.info(f"\n{results_df}")

#         best_model_name = results_df.iloc[0]['Model']
#         logger.info(f"Best Model: {best_model_name}")

#         # Return test IDs along with train-test split data
#         return best_model_name, X_train, y_train, X_test, y_test, test_ids

#     except Exception as e:
#         logger.error(f"Error during model selection: {e}")
#         raise


# v2

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVC, SVR
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from src.logging_config import get_logger

logger = get_logger(__name__)

def train_test_model_selection(df, target_column, id_column=None, task='regression'):
    """
    Basic model selection: Splits data, trains a variety of models, picks the best performer.
    """
    try:
        logger.info("Starting train-test split and model selection...")

        # Null/Empty checks
        if df.shape[0] == 0:
            raise ValueError("DataFrame is empty. Cannot proceed with model selection.")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
        if df[target_column].isnull().all():
            raise ValueError(f"Target column '{target_column}' is entirely null.")

        # If ID column is present, remove for modeling
        if id_column and id_column in df.columns:
            ids = df[id_column].copy()
            df = df.drop(columns=[id_column])
        else:
            ids = None

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # Choose stratify for classification
        stratify = y if task == 'classification' else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify
        )
        logger.info(f"Split done. Train size: {X_train.shape}, Test size: {X_test.shape}")

        if ids is not None:
            test_ids = ids.iloc[X_test.index]
        else:
            test_ids = None

        # Define models
        if task == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'SVM': SVC(),
                'XGBoost': XGBClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'Extra Trees': ExtraTreesClassifier(),
                'LightGBM': LGBMClassifier()
            }
            scorer = accuracy_score
        elif task == 'regression':
            models = {
                'KNN': KNeighborsRegressor(),
                'Decision Tree': DecisionTreeRegressor(),
                'Random Forest': RandomForestRegressor(),
                'Gradient Boosting': GradientBoostingRegressor(),
                'SVR': SVR(),
                'XGBoost': XGBRegressor(),
                'AdaBoost': AdaBoostRegressor(),
                'Extra Trees': ExtraTreesRegressor(),
                'LightGBM': LGBMRegressor()
            }
            scorer = mean_squared_error  # We'll take RMSE
        else:
            raise ValueError("task must be 'classification' or 'regression'.")

        results = []
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            if task == 'classification':
                score = accuracy_score(y_test, y_pred)
            else:
                # Use RMSE
                score = np.sqrt(mean_squared_error(y_test, y_pred))

            results.append({'Model': model_name, 'Score': score})
            logger.info(f"{model_name} => {score:.4f}")

        results_df = pd.DataFrame(results)
        ascending = True if task == 'regression' else False
        results_df.sort_values(by="Score", ascending=ascending, inplace=True)
        best_model_name = results_df.iloc[0]['Model']
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Model selection results:\n{results_df}")

        return best_model_name, X_train, y_train, X_test, y_test, test_ids

    except Exception as e:
        logger.error(f"Error in model_selection: {e}")
        raise
