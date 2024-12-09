import os
import joblib
import pandas as pd
import numpy as np
from src.helper import normalize_column_names
from src.logging_config import get_logger

# Initialize the logger
logger = get_logger(__name__)
import featuretools as ft



# def feature_engineering(df, target_column=None, dataframe_name="main", training = None):
#     """
#     Performs automated feature engineering using FeatureTools for both training and prediction.
    
#     Parameters:
#     - df: pandas DataFrame containing the dataset.
#     - target_column: The name of the target column (optional, only for training).
#     - dataframe_name: Name for the main dataframe in FeatureTools.
#     - feature_defs_path: Path to save/load feature definitions for reproducibility.
    
#     Returns:
#     - feature_matrix: DataFrame with engineered features.
#     - feature_defs: List of feature definitions (only during training).
#     """
#     try:
#         # Separate the target column if provided
#         if target_column:
#             target = df[target_column]
#             df = df.drop(columns=[target_column])
#         else:
#             target = None

#         # Identify binary columns
#         binary_columns = df.columns[(df.nunique() == 2) & (df.dtypes == "int64") | (df.dtypes == "float64")].tolist()

#         # Exclude binary columns temporarily for feature engineering
#         non_binary_df = df.drop(columns=binary_columns)

#         # Create an EntitySet
#         entity_set = ft.EntitySet()

#         # Add the non-binary dataframe to the EntitySet
#         entity_set = entity_set.add_dataframe(
#             dataframe_name=dataframe_name,
#             dataframe=non_binary_df,
#             index="index",  # Add a unique index if none exists
#             make_index=True,
#         )
#         feature_defs_path = "featurengineering_feature_defs.pkl"
#         if not training:
#             # Prediction phase: Load precomputed feature definitions
#             if not os.path.exists(feature_defs_path):
#                 raise RuntimeError("Feature definitions file not found for prediction.")
#             feature_defs = joblib.load(feature_defs_path)
#             feature_matrix = ft.calculate_feature_matrix(
#                 features=feature_defs,
#                 entityset=entity_set,
#                 verbose=True,
#             )
#         else:
#             # Generate feature definitions using DFS during training
#             feature_matrix, feature_defs = ft.dfs(
#                 entityset=entity_set,
#                 target_dataframe_name=dataframe_name,
#                 agg_primitives=["mean", "sum", "min", "max", "std"],
#                 trans_primitives=["add_numeric", "subtract_numeric", "divide_numeric"],
#                 max_depth=1,
#                 verbose=True,
#             )
#             # Save feature definitions
#             joblib.dump(feature_defs, feature_defs_path)
        

#         # Handle NaN and infinite values in the generated features
#         feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
#         feature_matrix.fillna(0, inplace=True)
        
#         # Normalize feature names for consistency
#         feature_matrix = normalize_column_names(feature_matrix)

#         # Add back the binary columns and the target column if applicable
#         feature_matrix = pd.concat([feature_matrix, df[binary_columns]], axis=1)
#         if target is not None:
#             feature_matrix[target_column] = target

#         # Return the feature matrix and feature definitions (only for training)
#         return feature_matrix, feature_defs if not feature_defs_path else None

#     except Exception as e:
#         raise RuntimeError(f"Error during feature engineering: {e}")


def feature_engineering(df, target_column=None, dataframe_name="main", training=True, feature_defs=None):
    """
    Performs automated feature engineering using FeatureTools for both training and prediction.

    Parameters:
    - df: pandas DataFrame containing the dataset.
    - target_column: The name of the target column (optional, only for training).
    - dataframe_name: Name for the main dataframe in FeatureTools.
    - training: Boolean indicating whether it is training or prediction phase.
    - feature_defs: List of feature definitions (required during prediction).

    Returns:
    - feature_matrix: DataFrame with engineered features.
    - feature_defs: List of feature definitions (only during training).
    """
    try:
        # Separate the target column if provided
        if target_column:
            target = df[target_column]
            df = df.drop(columns=[target_column])
        else:
            target = None

        # Identify binary columns
        binary_columns = df.columns[(df.nunique() == 2) & ((df.dtypes == "int64") | (df.dtypes == "float64"))].tolist()

        # Exclude binary columns temporarily for feature engineering
        non_binary_df = df.drop(columns=binary_columns)

        # Create an EntitySet
        entity_set = ft.EntitySet()

        # Add the non-binary dataframe to the EntitySet
        entity_set = entity_set.add_dataframe(
            dataframe_name=dataframe_name,
            dataframe=non_binary_df,
            index="index",  # Add a unique index if none exists
            make_index=True,
        )

        if training:
            # Training phase: Generate feature definitions using DFS
            feature_matrix, feature_defs = ft.dfs(
                entityset=entity_set,
                target_dataframe_name=dataframe_name,
                agg_primitives=["mean", "sum", "min", "max", "std"],
                trans_primitives=["add_numeric", "subtract_numeric", "divide_numeric"],
                max_depth=1,
                verbose=True,
            )
        else:
            # Prediction phase: Use provided feature definitions
            if feature_defs is None:
                raise RuntimeError("Feature definitions must be provided for prediction.")
            feature_matrix = ft.calculate_feature_matrix(
                features=feature_defs,
                entityset=entity_set,
                verbose=True,
            )

        # Handle NaN and infinite values in the generated features
        feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_matrix.fillna(0, inplace=True)

        # Normalize feature names for consistency
        feature_matrix = normalize_column_names(feature_matrix)

        # Add back the binary columns and the target column if applicable
        feature_matrix = pd.concat([feature_matrix, df[binary_columns]], axis=1)
        if target is not None:
            feature_matrix[target_column] = target

        # Return the feature matrix and feature definitions (only for training)
        if training:
            return feature_matrix, feature_defs
        else:
            return feature_matrix

    except Exception as e:
        raise RuntimeError(f"Error during feature engineering: {e}")
