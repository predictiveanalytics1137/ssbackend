import os
import joblib
import pandas as pd
import numpy as np
from automation.src import feature_selection
from automation.src.data_preprocessing import handle_categorical_features
from src.helper import normalize_column_names
from src.logging_config import get_logger

# Initialize the logger
logger = get_logger(__name__)
import featuretools as ft



# def feature_engineering(df, target_column=None, dataframe_name="main", training=True, feature_defs=None, id_column=None):
#     """
#     Performs automated feature engineering using FeatureTools for both training and prediction.

#     Parameters:
#     - df: pandas DataFrame containing the dataset.
#     - target_column: The name of the target column (optional, only for training).
#     - dataframe_name: Name for the main dataframe in FeatureTools.
#     - training: Boolean indicating whether it is training or prediction phase.
#     - feature_defs: List of feature definitions (required during prediction).
#     - id_column: The name of the entity column (e.g., ID column) to exclude from feature engineering.

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
            
#         # Exclude the ID column if specified
#         if id_column and id_column in df.columns:
#             id_data = df[id_column].copy()  # Save the ID column for later use
#             df = df.drop(columns=[id_column])
#             logger.info(f"Excluded entity column '{id_column}' from feature engineering.")

#         # Identify binary columns
#         binary_columns = df.columns[(df.nunique() == 2) & ((df.dtypes == "int64") | (df.dtypes == "float64"))].tolist()

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

#         if training:
#             # Training phase: Generate feature definitions using DFS
#             feature_matrix, feature_defs = ft.dfs(
#                 entityset=entity_set,
#                 target_dataframe_name=dataframe_name,
#                 agg_primitives=["mean", "sum", "min", "max", "std"],
#                 trans_primitives=["add_numeric", "subtract_numeric", "divide_numeric"],
#                 max_depth=1,
#                 verbose=True,
#             )
#         else:
#             # Prediction phase: Use provided feature definitions
#             if feature_defs is None:
#                 raise RuntimeError("Feature definitions must be provided for prediction.")
#             feature_matrix = ft.calculate_feature_matrix(
#                 features=feature_defs,
#                 entityset=entity_set,
#                 verbose=True,
#             )

#         # Handle NaN and infinite values in the generated features
#         feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
#         feature_matrix.fillna(0, inplace=True)

#         # Normalize feature names for consistency
#         feature_matrix = normalize_column_names(feature_matrix)

#         # Add back the binary columns and the target column if applicable
#         feature_matrix = pd.concat([feature_matrix, df[binary_columns]], axis=1)
#         if target is not None:
#             feature_matrix[target_column] = target
            
#         # Add back the ID column (unaltered) if applicable
#         if id_column:
#             feature_matrix[id_column] = id_data

#         # Return the feature matrix and feature definitions (only for training)
#         if training:
#             return feature_matrix, feature_defs
#         else:
#             return feature_matrix

#     except Exception as e:
#         raise RuntimeError(f"Error during feature engineering: {e}")



# v2


# =============================================================================
# import featuretools as ft
# import numpy as np
# import pandas as pd
# from src.helper import normalize_column_names
# from src.logging_config import get_logger
# 
# logger = get_logger(__name__)
# 
# def feature_engineering(df, target_column=None, dataframe_name="main", training=True,
#                         feature_defs=None, id_column=None):
#     """
#     Performs feature engineering using FeatureTools.
#     If training=True, generates new feature defs; if training=False, uses pre-fitted defs.
#     """
#     try:
#         if df.shape[0] == 0:
#             raise ValueError("DataFrame is empty; cannot perform feature engineering.")
# 
#         if target_column and (target_column in df.columns):
#             target = df[target_column].copy()
#             df = df.drop(columns=[target_column])
#         else:
#             target = None
# 
#         if id_column and (id_column in df.columns):
#             id_data = df[id_column].copy()
#             df = df.drop(columns=[id_column])
#         else:
#             id_data = None
# 
#         # Identify binary columns
#         binary_columns = df.columns[
#             (df.nunique() == 2) & 
#             ((df.dtypes == "int64") | (df.dtypes == "float64"))
#         ].tolist()
# 
#         non_binary_df = df.drop(columns=binary_columns)
# 
#         entity_set = ft.EntitySet()
#         entity_set = entity_set.add_dataframe(
#             dataframe_name=dataframe_name,
#             dataframe=non_binary_df,
#             index="index",
#             make_index=True,
#         )
# 
#         if training:
#             feature_matrix, new_feature_defs = ft.dfs(
#                 entityset=entity_set,
#                 target_dataframe_name=dataframe_name,
#                 agg_primitives=["mean", "sum", "min", "max", "std"],
#                 trans_primitives=["add_numeric", "subtract_numeric", "divide_numeric"],
#                 max_depth=1,
#                 verbose=True,
#             )
#         else:
#             if feature_defs is None:
#                 raise RuntimeError("Feature definitions must be provided for prediction.")
#             feature_matrix = ft.calculate_feature_matrix(
#                 features=feature_defs,
#                 entityset=entity_set,
#                 verbose=True,
#             )
#             new_feature_defs = feature_defs
# 
#         feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
#         feature_matrix.fillna(0, inplace=True)
# 
#         # Normalize names
#         feature_matrix = normalize_column_names(feature_matrix)
# 
#         # Add back binary columns
#         feature_matrix = pd.concat([feature_matrix, df[binary_columns]], axis=1)
# 
#         # Add back the target if it was removed
#         if target is not None:
#             feature_matrix[target_column] = target
# 
#         # Add back the ID column
#         if id_column and id_data is not None:
#             feature_matrix[id_column] = id_data
# 
#         if training:
#             return feature_matrix, new_feature_defs
#         else:
#             return feature_matrix
# 
#     except Exception as e:
#         logger.error(f"Error during feature engineering: {e}")
#         raise
# =============================================================================






# =============================================================================
# 
# import featuretools as ft
# import numpy as np
# import pandas as pd
# from src.helper import normalize_column_names
# from src.logging_config import get_logger
# 
# logger = get_logger(__name__)
# 
# def feature_engineering(
#     df, 
#     target_column=None, 
#     dataframe_name="main", 
#     training=True,
#     feature_defs=None, 
#     id_column=None
# ):
#     """
#     Performs feature engineering using FeatureTools.
#     If training=True, it generates new feature definitions;
#     if training=False, it uses the pre-fitted feature_defs.
#     """
# 
#     try:
#         # --------------------------------------------------------------------
#         # 1) Basic Checks
#         # --------------------------------------------------------------------
#         if df.shape[0] == 0:
#             raise ValueError("DataFrame is empty; cannot perform feature engineering.")
#         
#         # --------------------------------------------------------------------
#         # 2) Reset Index to Avoid Mismatch
#         #    Ensures a 0..(n-1) index for consistent alignment in FeatureTools.
#         # --------------------------------------------------------------------
#         df = df.reset_index(drop=True)
#         df.index.name = "original_index"  # A meaningful index name
# 
#         # --------------------------------------------------------------------
#         # 3) Separate Out Target Column, ID Column
#         # --------------------------------------------------------------------
#         target = None
#         if target_column and target_column in df.columns:
#             target = df[target_column].copy()
#             df = df.drop(columns=[target_column])
#         
#         id_data = None
#         if id_column and id_column in df.columns:
#             id_data = df[id_column].copy()
#             df = df.drop(columns=[id_column])
#         
#         # --------------------------------------------------------------------
#         # 4) Identify "Binary" Columns
#         #    - If your prior encoding turned bool -> {0,1}, you can catch them here.
#         #    - BUT in many pipelines, these columns might become float dtypes (e.g. 0.0, 1.0).
#         #      If so, you can tweak the check below.
#         # --------------------------------------------------------------------
#         potential_binary_cols = []
#         for col in df.columns:
#             unique_vals = df[col].dropna().unique()
#             if len(unique_vals) == 2:
#                 # For example, {0, 1} or {1.0, 0.0} or {True, False}
#                 # We'll store them as 'potential' binary columns
#                 potential_binary_cols.append(col)
#         
#         # If you truly want to exclude these from FeatureTools, we drop them now.
#         # Otherwise, if you want FeatureTools to generate features from them, skip this drop.
#         non_binary_df = df.drop(columns=potential_binary_cols) if potential_binary_cols else df.copy()
# 
#         # --------------------------------------------------------------------
#         # 5) Create an EntitySet WITHOUT auto indexing
#         #    make_index=False ensures we keep the DataFrame’s existing index.
#         # --------------------------------------------------------------------
#         entity_set = ft.EntitySet(id="entity_set")
#         entity_set.add_dataframe(
#             dataframe_name=dataframe_name,
#             dataframe=non_binary_df,
#             index=non_binary_df.index.name,  # "original_index"
#             make_index=False
#         )
# 
#         # --------------------------------------------------------------------
#         # 6) Run Deep Feature Synthesis or Calculate Feature Matrix
#         # --------------------------------------------------------------------
#         if training:
#             feature_matrix, new_feature_defs = ft.dfs(
#                 entityset=entity_set,
#                 target_dataframe_name=dataframe_name,
#                 agg_primitives=["mean", "sum", "min", "max", "std"],
#                 trans_primitives=["add_numeric", "subtract_numeric", "divide_numeric"],
#                 max_depth=1,
#                 verbose=True
#             )
#         else:
#             if feature_defs is None:
#                 raise RuntimeError("Feature definitions must be provided for prediction.")
#             feature_matrix = ft.calculate_feature_matrix(
#                 features=feature_defs,
#                 entityset=entity_set,
#                 verbose=True
#             )
#             new_feature_defs = feature_defs
# 
#         # Replace inf with NaN, then fill
#         feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
#         feature_matrix.fillna(0, inplace=True)
# 
#         # Normalize column names
#         feature_matrix = normalize_column_names(feature_matrix)
# 
#         # --------------------------------------------------------------------
#         # 7) Re-Add Potential Binary Columns
#         #    Use reindex to match the new FeatureTools index
#         # --------------------------------------------------------------------
#         if potential_binary_cols:
#             # Reindex to align with feature_matrix's index
#             binary_df = df[potential_binary_cols].reindex(feature_matrix.index)
#             feature_matrix = pd.concat([feature_matrix, binary_df], axis=1)
#             logger.info(f"Binary columns re-added: {potential_binary_cols}")
#         else:
#             logger.info("No binary columns were detected to add back.")
# 
#         # --------------------------------------------------------------------
#         # 8) Add back Target Column & ID Column
#         #    Also reindex them to match the feature_matrix index
#         # --------------------------------------------------------------------
#         if target is not None:
#             # Reindex the target to avoid mismatch
#             target_aligned = target.reindex(feature_matrix.index)
#             feature_matrix[target_column] = target_aligned
#         
#         if id_data is not None:
#             id_aligned = id_data.reindex(feature_matrix.index)
#             feature_matrix[id_column] = id_aligned
# 
#         # --------------------------------------------------------------------
#         # 9) Return
#         # --------------------------------------------------------------------
#         if training:
#             return feature_matrix, new_feature_defs
#         else:
#             return feature_matrix
# 
#     except Exception as e:
#         logger.error(f"Error in feature_engineering: {e}")
#         raise
# 
# =============================================================================





#v3


# before time series

import featuretools as ft
import numpy as np
import pandas as pd
from src.helper import normalize_column_names
from src.logging_config import get_logger

logger = get_logger(__name__)

def feature_engineering(
    df, 
    target_column=None, 
    dataframe_name="main", 
    training=True,
    feature_defs=None, 
    id_column=None,
    fixed_binary_cols=None  # New parameter to accept binary columns from training
):
    """
    Performs feature engineering using FeatureTools.
    If training=True, it generates new feature definitions and identifies binary columns;
    if training=False, it uses the pre-fitted feature_defs and fixed_binary_cols.
    """

    try:
        # --------------------------------------------------------------------
        # 1) Basic Checks
        # --------------------------------------------------------------------
        if df.shape[0] == 0:
            raise ValueError("DataFrame is empty; cannot perform feature engineering.")
        
        # --------------------------------------------------------------------
        # 2) Reset Index to Avoid Mismatch
        #    Ensures a 0..(n-1) index for consistent alignment in FeatureTools.
        # --------------------------------------------------------------------
        df = df.reset_index(drop=True)
        df.index.name = "original_index"  # A meaningful index name

        # --------------------------------------------------------------------
        # 3) Separate Out Target Column, ID Column
        # --------------------------------------------------------------------
        target = None
        if target_column and target_column in df.columns:
            target = df[target_column].copy()
            df = df.drop(columns=[target_column])
        
        id_data = None
        if id_column and id_column in df.columns:
            id_data = df[id_column].copy()
            df = df.drop(columns=[id_column])
        
        # --------------------------------------------------------------------
        # 4) Identify "Binary" Columns
        #    - If your prior encoding turned bool -> {0,1}, you can catch them here.
        #    - BUT in many pipelines, these columns might become float dtypes (e.g. 0.0, 1.0).
        #      If so, you can tweak the check below.
        # --------------------------------------------------------------------
        if training:
            potential_binary_cols = []
            for col in df.columns:
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) == 2:
                    # For example, {0, 1} or {1.0, 0.0} or {True, False}
                    # We'll store them as 'potential' binary columns
                    potential_binary_cols.append(col)
            logger.info(f"Identified potential binary columns: {potential_binary_cols}")
        else:
            if fixed_binary_cols is not None:
                potential_binary_cols = fixed_binary_cols
                logger.info(f"Using fixed binary columns from training: {potential_binary_cols}")
            else:
                # If not provided, fallback to identifying from test data
                potential_binary_cols = []
                for col in df.columns:
                    unique_vals = df[col].dropna().unique()
                    if len(unique_vals) == 2:
                        potential_binary_cols.append(col)
                logger.warning("Fixed binary columns not provided. Identifying binary columns from test data.")
        
        # If training, return the list of binary columns
        if training:
            # Save the list to pass to test
            pass  # Will handle later
        # If you truly want to exclude these from FeatureTools, we drop them now.
        # Otherwise, if you want FeatureTools to generate features from them, skip this drop.
        non_binary_df = df.drop(columns=potential_binary_cols) if potential_binary_cols else df.copy()

        # --------------------------------------------------------------------
        # 5) Create an EntitySet WITHOUT auto indexing
        #    make_index=False ensures we keep the DataFrame’s existing index.
        # --------------------------------------------------------------------
        entity_set = ft.EntitySet(id="entity_set")
        entity_set.add_dataframe(
            dataframe_name=dataframe_name,
            dataframe=non_binary_df,
            index=non_binary_df.index.name,  # "original_index"
            make_index=False
        )

        # --------------------------------------------------------------------
        # 6) Run Deep Feature Synthesis or Calculate Feature Matrix
        # --------------------------------------------------------------------
        if training:
            feature_matrix, new_feature_defs = ft.dfs(
                entityset=entity_set,
                target_dataframe_name=dataframe_name,
                agg_primitives=["mean", "sum", "min", "max", "std"],
                trans_primitives=["add_numeric", "subtract_numeric", "divide_numeric"],
                max_depth=1,
                verbose=True
            )
        else:
            if feature_defs is None:
                raise RuntimeError("Feature definitions must be provided for prediction.")
            feature_matrix = ft.calculate_feature_matrix(
                features=feature_defs,
                entityset=entity_set,
                verbose=True
            )
            new_feature_defs = feature_defs

        # Replace inf with NaN, then fill
        feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
        feature_matrix.fillna(0, inplace=True)

        # Normalize column names
        feature_matrix = normalize_column_names(feature_matrix)

        # --------------------------------------------------------------------
        # 7) Re-Add Potential Binary Columns
        #    Use reindex to match the new FeatureTools index
        # --------------------------------------------------------------------
        if potential_binary_cols:
            # Reindex to align with feature_matrix's index
            binary_df = df[potential_binary_cols].reindex(feature_matrix.index)
            feature_matrix = pd.concat([feature_matrix, binary_df], axis=1)
            logger.info(f"Binary columns re-added: {potential_binary_cols}")
        else:
            logger.info("No binary columns were detected to add back.")
        
        # --------------------------------------------------------------------
        # 8) Add back Target Column & ID Column
        #    Also reindex them to match the feature_matrix index
        # --------------------------------------------------------------------
        if target is not None:
            # Reindex the target to avoid mismatch
            target_aligned = target.reindex(feature_matrix.index)
            feature_matrix[target_column] = target_aligned
        
        if id_data is not None:
            id_aligned = id_data.reindex(feature_matrix.index)
            feature_matrix[id_column] = id_aligned

        # --------------------------------------------------------------------
        # 9) Return
        # --------------------------------------------------------------------
        if training:
            return feature_matrix, new_feature_defs, potential_binary_cols
        else:
            return feature_matrix

    except Exception as e:
        logger.error(f"Error in feature_engineering: {e}")
        raise


# =============================================================================
# 
# def feature_engineering_timeseries(
#     df,
#     target_column=None,
#     id_column=None,
#     time_column=None,
#     training=True,
#     feature_defs=None,
#     dataframe_name="demand"
# ):
#     try:
#         if df.empty:
#             raise ValueError("DataFrame is empty; cannot perform feature engineering.")
# 
#         if time_column and time_column in df.columns:
#             df[time_column] = pd.to_datetime(df[time_column], errors="coerce")
# 
#         if "index" not in df.columns:
#             df["index"] = range(len(df))
#         df = df.reset_index(drop=True)
# 
#         es = ft.EntitySet(id="entity_set")
#         es.add_dataframe(
#             dataframe_name=dataframe_name,
#             dataframe=df,
#             index="index",
#             time_index=time_column if time_column in df.columns else None
#         )
# 
#         if id_column and id_column in df.columns:
#             entity_df = df[[id_column]].drop_duplicates().reset_index(drop=True)
#             es.add_dataframe(
#                 dataframe_name="entities",
#                 dataframe=entity_df,
#                 index=id_column
#             )
#             es.add_relationship("entities", id_column, dataframe_name, id_column)
# 
#         agg_primitives = ["mean", "sum", "count"]
#         trans_primitives = ["month"] if time_column else []
# 
#         if training:
#             cutoff_times = df[["index", time_column]].rename(columns={time_column: "time"}) if time_column in df.columns else None
#             feature_matrix, new_feature_defs = ft.dfs(
#                 entityset=es,
#                 target_dataframe_name=dataframe_name,
#                 cutoff_time=cutoff_times,
#                 agg_primitives=agg_primitives,
#                 trans_primitives=trans_primitives,
#                 max_depth=2,
#                 verbose=False
#             )
#             if target_column and target_column in df.columns:
#                 feature_matrix[target_column] = df[target_column].reindex(feature_matrix.index)
#             # Reattach analysis_time if it exists
#             if time_column and time_column in df.columns:
#                 feature_matrix[time_column] = df[time_column].reindex(feature_matrix.index)
#         else:
#             if feature_defs is None:
#                 raise RuntimeError("Feature definitions must be provided for prediction.")
#             available_cols = set(df.columns)
#             computable_defs = [
#                 f for f in feature_defs if all(
#                     isinstance(dep, str) and dep in available_cols
#                     for dep in f.get_dependencies(deep=True) or []
#                 )
#             ]
#             if not computable_defs:
#                 logger.info("No computable Featuretools features; returning input DataFrame.")
#                 return df
#             feature_matrix = ft.calculate_feature_matrix(
#                 features=computable_defs,
#                 entityset=es,
#                 verbose=False
#             )
#             new_feature_defs = feature_defs
# 
#         feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
#         numeric_cols = feature_matrix.select_dtypes(include=['float64', 'int64']).columns
#         if not numeric_cols.empty:
#             feature_matrix[numeric_cols] = feature_matrix[numeric_cols].fillna(0)
#         if id_column and id_column in df.columns:
#             feature_matrix[id_column] = df[id_column].reindex(feature_matrix.index)
#         if time_column and time_column in df.columns:
#             feature_matrix[time_column] = df[time_column].reindex(feature_matrix.index)  # Ensure time_column is retained
# 
#         logger.info(f"Generated features: {feature_matrix.columns.tolist()}")
#         return (feature_matrix, new_feature_defs) if training else feature_matrix
# 
#     except Exception as e:
#         logger.error(f"Error in feature_engineering: {e}")
#         raise
# 
# 
# 
# =============================================================================





import featuretools as ft
import pandas as pd
import numpy as np
import logging
from src.logging_config import get_logger

logger = get_logger(__name__)

# def feature_engineering_timeseries(
#     df,
#     target_column=None,
#     id_column=None,
#     time_column=None,
#     training=True,
#     feature_defs=None,
#     dataframe_name="demand"
# ):
#     """
#     Performs feature engineering using Featuretools, excluding the target_column from feature generation
#     to prevent data leakage. Retains the target_column as a label for training.

#     Parameters:
#     - df (pd.DataFrame): Input DataFrame.
#     - target_column (str, optional): Target column name to exclude from feature generation.
#     - id_column (str, optional): ID column for entity relationships.
#     - time_column (str, optional): Time column for time-series features.
#     - training (bool): Whether in training mode (default: True).
#     - feature_defs (list, optional): Precomputed feature definitions for prediction.
#     - dataframe_name (str): Name of the dataframe in the EntitySet (default: "demand").

#     Returns:
#     - tuple: (feature_matrix, feature_defs) if training, feature_matrix if prediction.
#     """
#     try:
#         if df.empty:
#             raise ValueError("DataFrame is empty; cannot perform feature engineering.")

#         if time_column and time_column in df.columns:
#             df[time_column] = pd.to_datetime(df[time_column], errors="coerce")

#         if "index" not in df.columns:
#             df["index"] = range(len(df))
#         df = df.reset_index(drop=True)

#         logger.info(f"DataFrame columns: {df.columns.tolist()}")


#         # Create a copy of df excluding target_column for feature generation
#         feature_df = df.drop(columns=[target_column], errors='ignore') if target_column and target_column in df.columns else df.copy()

#         es = ft.EntitySet(id="entity_set")
#         es.add_dataframe(
#             dataframe_name=dataframe_name,
#             dataframe=feature_df,
#             index="index",
#             time_index=time_column if time_column in feature_df.columns else None
#         )

#         if id_column and id_column in feature_df.columns:
#             entity_df = feature_df[[id_column]].drop_duplicates().reset_index(drop=True)
#             es.add_dataframe(
#                 dataframe_name="entities",
#                 dataframe=entity_df,
#                 index=id_column
#             )
#             es.add_relationship("entities", id_column, dataframe_name, id_column)

#         agg_primitives = ["mean", "sum", "count"]
#         trans_primitives = ["month"] if time_column else []

#         if training:
#             cutoff_times = feature_df[["index", time_column]].rename(columns={time_column: "time"}) if time_column in feature_df.columns else None
#             feature_matrix, new_feature_defs = ft.dfs(
#                 entityset=es,
#                 target_dataframe_name=dataframe_name,
#                 cutoff_time=cutoff_times,
#                 agg_primitives=agg_primitives,
#                 trans_primitives=trans_primitives,
#                 max_depth=2,
#                 verbose=False
#             )
#             # Reattach target_column as a label
#             if target_column and target_column in df.columns:
#                 feature_matrix[target_column] = df[target_column].iloc[:len(feature_matrix)].reset_index(drop=True)
#             # Reattach time_column if it exists
#             if time_column and time_column in df.columns:
#                 feature_matrix[time_column] = df[time_column].iloc[:len(feature_matrix)].reset_index(drop=True)
#             logger.info(f"Generated feature_defs: {[f.get_name() for f in new_feature_defs]}")
#         else:
#             if feature_defs is None:
#                 raise RuntimeError("Feature definitions must be provided for prediction.")
#             available_cols = set(feature_df.columns)
#             computable_defs = [
#                 f for f in feature_defs if all(
#                     isinstance(dep, str) and dep in available_cols
#                     for dep in f.get_dependencies(deep=True) or []
#                 )
#             ]
#             if not computable_defs:
#                 logger.warning("No computable Featuretools features; falling back to input data.")
#                 return df
#             feature_matrix = ft.calculate_feature_matrix(
#                 features=computable_defs,
#                 entityset=es,
#                 verbose=False
#             )
#             new_feature_defs = feature_defs
#             # Reset index to ensure a standard pandas Index
#             feature_matrix = feature_matrix.reset_index(drop=True)

#         logger.info(f"Generated features: {feature_matrix.columns.tolist()}")
#         feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
#         numeric_cols = feature_matrix.select_dtypes(include=['float64', 'int64']).columns
#         if not numeric_cols.empty:
#             feature_matrix[numeric_cols] = feature_matrix[numeric_cols].fillna(0)
#         if id_column and id_column in df.columns:
#             feature_matrix[id_column] = df[id_column].iloc[:len(feature_matrix)].reset_index(drop=True)
#         if time_column and time_column in df.columns:
#             feature_matrix[time_column] = df[time_column].iloc[:len(feature_matrix)].reset_index(drop=True)

#         logger.info(f"Generated features: {feature_matrix.columns.tolist()}")
#         return (feature_matrix, new_feature_defs) if training else feature_matrix

#     except Exception as e:
#         logger.error(f"Error in feature_engineering: {e}")
#         raise




# V4

import featuretools as ft
import pandas as pd
import numpy as np
import logging
from src.logging_config import get_logger

logger = get_logger(__name__)

def feature_engineering_timeseries(
    df,
    target_column=None,
    id_column=None,
    time_column=None,
    training=True,
    feature_defs=None,
    dataframe_name="demand",
    # agg_primitives=["mean", "std", "sum", "count", "min", "max"],
    agg_primitives=["mean"],
    # trans_primitives=["month", "weekday", "week", "lag", "rolling_mean", "rolling_std"],
    trans_primitives=["month","rolling_mean"],
    max_depth=2,
    chunk_size=None
):
    """
    Optimized feature engineering for time-series using Featuretools, with integrated time-based feature
    extraction (month, day_of_week, week_of_year). Generates features efficiently while preventing data
    leakage and integrating custom preprocessing and selection.

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str, optional): Target column to exclude from feature generation.
    - id_column (str, optional): ID column for entity relationships.
    - time_column (str, optional): Time column for time-series features.
    - training (bool): Training mode (True) or prediction mode (False).
    - feature_defs (list, optional): Precomputed feature definitions for prediction.
    - dataframe_name (str): Name of the EntitySet dataframe (default: "demand").
    - agg_primitives (list): Aggregation primitives for feature generation.
    - trans_primitives (list): Transformation primitives for feature generation.
    - max_depth (int): Maximum depth of feature synthesis (default: 2).
    - chunk_size (int, optional): Size of chunks for large datasets to manage memory.

    Returns:
    - tuple: (feature_matrix, feature_defs) if training, feature_matrix if prediction.
    """
    try:
        if df.empty:
            raise ValueError("Input DataFrame is empty.")

        # Convert time_column to datetime efficiently
        if time_column and time_column in df.columns:
            df[time_column] = pd.to_datetime(df[time_column], errors="coerce", infer_datetime_format=True)
        else:
            logger.warning("Time column not provided or not found; time-based features will be limited.")

        # Add index if missing
        if "index" not in df.columns:
            df["index"] = np.arange(len(df))

        logger.info(f"Input DataFrame shape: {df.shape}, columns: {df.columns.tolist()}")

        # Integrate time-based features manually (from create_time_based_features)
        if time_column and time_column in df.columns:
            df["month"] = df[time_column].dt.month
            df["day_of_week"] = df[time_column].dt.dayofweek
            df["week_of_year"] = df[time_column].dt.isocalendar().week.astype(int)
            logger.info("Added manual time-based features: month, day_of_week, week_of_year")

        # Apply custom categorical preprocessing (assumed function)
        # df = handle_categorical_features(df)

        # Isolate target to prevent leakage
        feature_df = (
            df.drop(columns=[target_column], errors="ignore")
            if target_column and target_column in df.columns
            else df.copy()
        )

        # Initialize EntitySet
        es = ft.EntitySet(id="entity_set")
        es.add_dataframe(
            dataframe_name=dataframe_name,
            dataframe=feature_df,
            index="index",
            time_index=time_column if time_column in feature_df.columns else None,
            logical_types={col: "Categorical" for col in feature_df.select_dtypes(include=["object"]).columns}
        )

        # Add entity relationship if id_column exists
        if id_column and id_column in feature_df.columns:
            entity_df = feature_df[[id_column]].drop_duplicates()
            es.add_dataframe(dataframe_name="entities", dataframe=entity_df, index=id_column)
            es.add_relationship("entities", id_column, dataframe_name, id_column)

        # Feature generation
        if training:
            cutoff_times = (
                feature_df[["index", time_column]].rename(columns={time_column: "time"})
                if time_column in feature_df.columns
                else None
            )
            feature_matrix, new_feature_defs = ft.dfs(
                entityset=es,
                target_dataframe_name=dataframe_name,
                cutoff_time=cutoff_times,
                agg_primitives=agg_primitives,
                trans_primitives=trans_primitives,
                max_depth=max_depth,
                verbose=False,
                chunk_size=chunk_size
            )
        else:
            if not feature_defs:
                raise ValueError("Feature definitions required for prediction mode.")
            # Filter computable features
            available_cols = set(feature_df.columns)
            computable_defs = [
                f for f in feature_defs
                if all(dep in available_cols for dep in f.get_dependencies(deep=True) if isinstance(dep, str))
            ]
            if not computable_defs:
                logger.warning("No computable features; returning preprocessed input.")
                return df
            feature_matrix = ft.calculate_feature_matrix(
                features=computable_defs,
                entityset=es,
                verbose=False,
                chunk_size=chunk_size
            )
            new_feature_defs = feature_defs

        # Reattach target_column (if present) in both training and prediction modes
        if target_column and target_column in df.columns:
            feature_matrix[target_column] = df[target_column].iloc[:len(feature_matrix)].values
            logger.info(f"Reattached target column '{target_column}' to feature matrix.")

        if time_column and time_column in df.columns:
            feature_matrix[time_column] = df[time_column].iloc[:len(feature_matrix)].values

        # Ensure time-based features from Featuretools are numeric
        for col in feature_matrix.columns:
            if col.startswith(("MONTH", "WEEKDAY", "WEEK")) and feature_matrix[col].dtype.name == "category":
                feature_matrix[col] = feature_matrix[col].astype(int)
                logger.info(f"Converted '{col}' from category to int.")

        # Reset index for consistency
        feature_matrix = feature_matrix.reset_index(drop=True)

        # Clean up infinite values and handle NaNs
        feature_matrix.replace([np.inf, -np.inf], np.nan, inplace=True)
        numeric_cols = feature_matrix.select_dtypes(include=["float64", "int64"]).columns
        if not numeric_cols.empty:
            feature_matrix[numeric_cols] = (
                feature_matrix[numeric_cols]
                .fillna(method="ffill")
                .fillna(method="bfill")
                .fillna(0)
            )

        # Apply custom feature selection (assumed function)
        # feature_matrix = feature_selection(feature_matrix)

        logger.info(f"Final feature matrix shape: {feature_matrix.shape}, columns: {feature_matrix.columns.tolist()}")

        return (feature_matrix, new_feature_defs) if training else feature_matrix

    except Exception as e:
        logger.error(f"Feature engineering failed: {str(e)}", exc_info=True)
        raise

# Example usage (commented out):
# df = pd.DataFrame({...})
# feature_matrix, feature_defs = feature_engineering_timeseries(df, target_column="sales", time_column="date", id_column="store_id")