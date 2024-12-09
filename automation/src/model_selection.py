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

def train_test_model_selection(df, target_column, task='regression'):
    """
    Splits the dataset into training and testing sets, and evaluates multiple machine learning algorithms.
    
    Parameters:
    - df: pandas DataFrame containing the dataset
    - target_column: string, name of the target column
    - task: string, either 'classification' or 'regression' depending on the problem type
    
    Returns:
    - best_model_name: string, name of the best-performing model
    - X_train, X_test, y_train, y_test: train-test split data
    """
    try:
        logger.info("Starting train-test split and model selection...")

        # 1. Clean up column names
        df.columns = df.columns.str.replace('<', '', regex=True).str.replace('>', '', regex=True).str.replace(' ', '_', regex=True)
        logger.info(f"Cleaned column names: {list(df.columns)}")

        # 2. Split data into features (X) and target (y)
        X = df.drop(columns=[target_column])
        y = df[target_column]
        logger.info(f"Splitting dataset into training and testing sets with target column: {target_column}")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Train-test split completed. Training size: {X_train.shape[0]}, Testing size: {X_test.shape[0]}")

        # 3. Define models for classification and regression
        if task == 'classification':
            models = {
                'Logistic Regression': LogisticRegression(),
                'KNN': KNeighborsClassifier(),
                'Decision Tree': DecisionTreeClassifier(),
                'Random Forest': RandomForestClassifier(),
                'Gradient Boosting': GradientBoostingClassifier(),
                'SVM': SVC(),
                'Naive Bayes': GaussianNB(),
                'XGBoost': XGBClassifier(),
                'AdaBoost': AdaBoostClassifier(),
                'Extra Trees': ExtraTreesClassifier(),
                'LightGBM': LGBMClassifier()
            }
            metric = accuracy_score
        elif task == 'regression':
            models = {
                #'Linear Regression': LinearRegression(),
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
            metric = mean_squared_error
        else:
            raise ValueError("Invalid task type. Choose 'classification' or 'regression'.")

        # 4. Train and evaluate each model
        results = []
        for name, model in models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Predict on the test set
            y_pred = model.predict(X_test)
            
            # Evaluate the model
            if task == 'classification':
                score = accuracy_score(y_test, y_pred)
            elif task == 'regression':
                score = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE
            
            logger.info(f"Model: {name}, Score: {score}")
            results.append({'Model': name, 'Score': score})

        # 5. Select the best-performing model
        results_df = pd.DataFrame(results).sort_values(by='Score', ascending=(task == 'regression'))  # Higher is better for classification
        logger.info("\nModel Selection Complete. Results:")
        logger.info(f"\n{results_df}")

        best_model_name = results_df.iloc[0]['Model']
        logger.info(f"Best Model: {best_model_name}")

        return best_model_name, X_train, y_train, X_test, y_test

    except Exception as e:
        logger.error(f"Error during model selection: {e}")
        raise
