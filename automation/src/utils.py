
'''

1. Why are we removing the target column in automatic_imputation?
Answer: Yes, this is a necessary step because the target column should not be imputed. 
Imputation is performed on features (independent variables), not on the target (dependent variable), 
as the target is what the model is trying to predict. Imputing the target could introduce bias or compromise the validity of the model.



4. Why are we skipping the target column in the imputation step?
Answer: Skipping the target column during imputation is necessary because:

Imputation alters the data by filling missing values with estimations (mean, median, KNN, etc.).
If the target column is imputed, the model's learning process can become biased or invalid as the target values should reflect the true labels without manipulation.
Thus, skipping the target column is correct and necessary.


'''



# def automatic_imputation(df, target_column, threshold_knn=0.05, threshold_iterative=0.15, imputers=None):
#     """
#     Automatically imputes missing values for numerical and categorical features.
#     Ensures that imputed data is assigned as a 1D array, preventing ValueError: 2.
#     """
#     try:
#         logger.info("Starting automatic imputation...")

#         # Separate numerical and categorical columns
#         numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#         categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

#         # Remove the target column from the lists (if it's in there)
#         if target_column in numerical_columns:
#             numerical_columns.remove(target_column)
#         if target_column in categorical_columns:
#             categorical_columns.remove(target_column)

#         logger.info(f"Numerical columns: {numerical_columns}")
#         logger.info(f"Categorical columns: {categorical_columns}")

#         # If imputers are not provided (training case), initialize an empty dictionary for them
#         if imputers is None:
#             imputers = {}

#         # Impute missing values for each column based on threshold
#         for column in df.columns:
#             if column == target_column:
#                 continue  # Skip the target column

#             missing_percentage = df[column].isnull().mean()
#             logger.info(f"Processing column '{column}' with {missing_percentage:.2%} missing values.")

#             if missing_percentage == 0:
#                 logger.info(f"Column '{column}' has no missing values. Skipping imputation.")
#                 continue

#             # For prediction (when imputers are loaded), apply the saved imputer directly
#             if column in imputers:
#                 imputer = imputers[column]
#                 transformed = imputer.transform(df[[column]])
#                 # Flatten the transformed array before assigning
#                 df[column] = transformed.ravel()
#             else:
#                 # Choose imputer based on the threshold and column type
#                 if column in categorical_columns:
#                     logger.info(f"Applying SimpleImputer (most frequent) to column '{column}'.")
#                     imputer = SimpleImputer(strategy='most_frequent')
#                 elif column in numerical_columns:
#                     if missing_percentage < threshold_knn:
#                         logger.info(f"Applying SimpleImputer (median) to column '{column}'.")
#                         imputer = SimpleImputer(strategy='median')
#                     elif missing_percentage < threshold_iterative:
#                         logger.info("Applying KNNImputer to numerical columns...")
#                         imputer = KNNImputer(n_neighbors=5)
#                     else:
#                         logger.info("Applying IterativeImputer to numerical columns...")
#                         imputer = IterativeImputer(max_iter=10, random_state=0)

#                 transformed = imputer.fit_transform(df[[column]])
#                 # Flatten the transformed array to avoid ValueError: 2 issues
#                 df[column] = transformed.ravel()
#                 imputers[column] = imputer

#         logger.info("Imputation complete.")
#         return df, imputers

#     except Exception as e:
#         logger.error(f"Error during automatic imputation: {e}")
#         raise



# v2



# src/utils.py


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.impute import IterativeImputer
from src.logging_config import get_logger

logger = get_logger(__name__)

# def automatic_imputation(df, target_column, threshold_knn=0.05, threshold_iterative=0.15, imputers=None):
#     """
#     Automatically imputes missing values for numerical & categorical columns 
#     to avoid data leakage. If 'imputers' is given, apply for inference.
#     """
#     try:
#         logger.info("Starting automatic imputation...")

#         if df.shape[0] == 0:
#             raise ValueError("DataFrame is empty, cannot impute.")

#         numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
#         categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

#         if target_column and target_column in numerical_columns:
#             numerical_columns.remove(target_column)
#         if target_column and target_column in categorical_columns:
#             categorical_columns.remove(target_column)

#         if imputers is None:
#             imputers = {}

#         for column in df.columns:
#             if column == target_column:
#                 continue
#             missing_pct = df[column].isnull().mean()
#             if missing_pct == 0:
#                 continue

#             logger.info(f"Imputing column '{column}' with {missing_pct:.2%} missing.")

#             if column in imputers:
#                 imputer = imputers[column]
#                 df[column] = imputer.transform(df[[column]]).ravel()
#             else:
#                 if column in categorical_columns:
#                     imputer = SimpleImputer(strategy='most_frequent')
#                 elif column in numerical_columns:
#                     if missing_pct < threshold_knn:
#                         imputer = SimpleImputer(strategy='median')
#                     elif missing_pct < threshold_iterative:
#                         imputer = KNNImputer(n_neighbors=5)
#                     else:
#                         imputer = IterativeImputer(random_state=42, max_iter=10)

#                 df[column] = imputer.fit_transform(df[[column]]).ravel()
#                 imputers[column] = imputer

#         logger.info("Imputation complete.")
#         return df, imputers

#     except Exception as e:
#         logger.error(f"Error in automatic_imputation: {e}")
#         raise


def automatic_imputation(df, target_column=None, threshold_knn=0.05, threshold_iterative=0.15, 
                        imputers=None, mode='train', drop_target_na=False, lightweight_mode=False):
    """
    Automatically imputes missing values for numerical and categorical columns in an AutoML pipeline.
    
    Parameters:
    - df: pd.DataFrame, input data
    - target_column: str, column to exclude from imputation (optional)
    - threshold_knn: float, threshold for KNNImputer (default: 0.05)
    - threshold_iterative: float, threshold for IterativeImputer (default: 0.15)
    - imputers: dict, pre-fitted imputers for inference (default: None)
    - mode: str, 'train' to fit imputers, 'infer' to apply them (default: 'train')
    - drop_target_na: bool, drop rows with missing target values (default: False)
    - lightweight_mode: bool, use only SimpleImputer for speed (default: False)
    
    Returns:
    - df: pd.DataFrame, imputed data
    - imputers: dict, fitted imputers (in train mode)
    """
    logger = get_logger(__name__)
    logger.info(f"Starting automatic imputation in {mode} mode...")

    if df.shape[0] == 0:
        raise ValueError("DataFrame is empty, cannot impute.")
    if target_column and target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")
    if mode == 'infer' and imputers is None:
        raise ValueError("Imputers dictionary required in inference mode.")

    if drop_target_na and target_column and df[target_column].isnull().any():
        df = df.dropna(subset=[target_column])
        logger.info(f"Dropped rows with missing '{target_column}' values.")

    numerical_columns = df.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if target_column:
        numerical_columns = [col for col in numerical_columns if col != target_column]
        categorical_columns = [col for col in categorical_columns if col != target_column]

    if mode == 'train' and imputers is None:
        imputers = {}

    try:
        for column in df.columns:
            if column == target_column or not df[column].isnull().any():
                continue
            missing_pct = df[column].isnull().mean()
            logger.info(f"Imputing '{column}' with {missing_pct:.2%} missing.")

            if mode == 'infer':
                if column not in imputers:
                    raise ValueError(f"No imputer for '{column}' in inference mode.")
                imputer = imputers[column]
                df[column] = imputer.transform(df[[column]]).ravel()
            else:  # mode == 'train'
                if column in categorical_columns:
                    imputer = SimpleImputer(strategy='most_frequent')
                elif column in numerical_columns:
                    if lightweight_mode or missing_pct < threshold_knn:
                        imputer = SimpleImputer(strategy='median')
                    elif missing_pct < threshold_iterative:
                        imputer = KNNImputer(n_neighbors=5)
                    else:
                        imputer = IterativeImputer(random_state=42, max_iter=10)
                else:
                    logger.warning(f"Skipping '{column}' with unsupported type: {df[column].dtype}")
                    continue

                df[column] = imputer.fit_transform(df[[column]]).ravel()
                imputers[column] = imputer
                logger.info(f"Fitted {imputer.__class__.__name__} for '{column}'.")

        logger.info("Imputation complete.")
        return df, imputers if mode == 'train' else None

    except Exception as e:
        logger.error(f"Error imputing column '{column}': {e}")
        raise