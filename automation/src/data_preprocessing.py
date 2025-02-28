from sklearn.experimental import enable_iterative_imputer  # Needed to enable IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import BinaryEncoder, TargetEncoder
import pandas as pd
from src.logging_config import get_logger

logger = get_logger(__name__)

'''

Ordinary Encoding:
Replaces each category in a feature with a unique integer (e.g., Red -> 0, Blue -> 1). 
It's simple but may mistakenly imply an order between categories.

Target Encoding:
Replaces each category with the mean of the target variable for that category (e.g., Red -> 0.7 if 70% of "Red" rows have the target as 1). 
It captures the relationship between the category and the target but requires careful handling to avoid overfitting.


'''


# def handle_categorical_features(df, target_column=None, cardinality_threshold=3, encoders=None, saved_column_names=None, id_column=None):
#     logger.info("Starting to handle categorical features...")

#     try:
#         # Identify categorical columns
#         categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
#         logger.info(f"Categorical columns identified: {categorical_columns}")
#         if id_column in categorical_columns:
#             categorical_columns.remove(id_column)
#             logger.info(f"Excluded ID column '{id_column}' from categorical processing.")

#         df_encoded = df.copy()
#         if encoders is None:
#             encoders = {}

#         all_encoded_columns = []

#         for column in categorical_columns:
#             unique_values = df[column].nunique()
#             logger.info(f"Processing column '{column}' with {unique_values} unique values.")

#             # Ensure consistent column name (replace spaces with underscores and convert to lowercase)
#             consistent_column_name = column.replace(' ', '_').replace("'", "").lower()
#             df_encoded.rename(columns={column: consistent_column_name}, inplace=True)
#             column = consistent_column_name  # Update the column name in the loop

#             # Apply encoding
#             if column in encoders:
#                 encoder = encoders[column]
#                 # Transform the column using the preloaded encoder
#                 if isinstance(encoder, OrdinalEncoder):
#                     df_encoded[column] = encoder.transform(df_encoded[[column]])
#                 elif isinstance(encoder, TargetEncoder):
#                     df_encoded[column] = encoder.transform(df_encoded[column])
#                 elif isinstance(encoder, BinaryEncoder):
#                     binary_transformed = encoder.transform(df_encoded[[column]])
#                     df_encoded = pd.concat([df_encoded.drop(column, axis=1), binary_transformed], axis=1)
#                 logger.info(f"Applied preloaded encoder to column '{column}'.")
#             else:
#                 # Fit and transform using a new encoder during training
#                 if unique_values <= cardinality_threshold:
#                     one_hot_encoded = pd.get_dummies(df[column], prefix=column)
#                     all_encoded_columns.extend(one_hot_encoded.columns)
#                     df_encoded = pd.concat([df_encoded.drop(column, axis=1), one_hot_encoded], axis=1)
#                     encoder = None
#                 elif target_column and target_column in df.columns:
#                     encoder = TargetEncoder().fit(df[column], df[target_column])
#                     df_encoded[column] = encoder.transform(df[column])
#                 elif unique_values > 100:
#                     encoder = BinaryEncoder().fit(df[[column]])
#                     binary_transformed = encoder.transform(df[[column]])
#                     all_encoded_columns.extend(binary_transformed.columns)
#                     df_encoded = pd.concat([df_encoded.drop(column, axis=1), binary_transformed], axis=1)
#                 else:
#                     encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(df[[column]])
#                     df_encoded[column] = encoder.transform(df[[column]])

#                 if encoder is not None:
#                     encoders[column] = encoder

#         # Convert boolean columns to 0 and 1
#         # bool_columns = df_encoded.select_dtypes(include=['bool']).columns.tolist()
#         # if bool_columns:
#         #     df_encoded[bool_columns] = df_encoded[bool_columns].replace({True: 1, False: 0})

#         # Convert boolean columns to 0 and 1
#         # bool_columns = df_encoded.select_dtypes(include=['bool']).columns.tolist()
#         # if bool_columns:
#         #     # Perform the replacement and ensure the dtype is explicitly set to int
#         #     df_encoded[bool_columns] = (
#         #         df_encoded[bool_columns]
#         #         .replace({True: 1, False: 0})
#         #         .astype(int)
#         #     )

#         # Convert boolean columns to integers explicitly, avoiding replace
#         bool_columns = df_encoded.select_dtypes(include=['bool']).columns.tolist()
#         if bool_columns:
#             for col in bool_columns:
#                 logger.info(f"Converting boolean column '{col}' to integers.")
#                 # Directly convert boolean to integers using astype
#                 df_encoded[col] = df_encoded[col].astype('int64')



#         logger.info("Completed handling of categorical features.")

#         # Align columns with the training data, ensuring that any missing columns are added
#         if saved_column_names:
#             logger.info(f"Aligning columns with training data: {saved_column_names}")
#             # Exclude the ID column from alignment
#             saved_column_names = [col for col in saved_column_names if col != id_column]
#             missing_cols = [col for col in saved_column_names if col not in df_encoded.columns]
#             for col in missing_cols:
#                 df_encoded[col] = 0
#             df_encoded = df_encoded[saved_column_names]

#         return df_encoded, encoders

#     except Exception as e:
#         logger.error(f"Error while handling categorical features: {e}")
#         raise


# v2
# =============================================================================
# 
# from sklearn.experimental import enable_iterative_imputer  # Required to enable IterativeImputer
# from sklearn.preprocessing import OrdinalEncoder
# from category_encoders import BinaryEncoder, TargetEncoder
# import pandas as pd
# from src.logging_config import get_logger
# 
# logger = get_logger(__name__)
# 
# """
# This module handles categorical features with either one-hot, ordinal, binary, or target encoding.
# Improvements:
#  - Data schema checks are handled in higher-level modules to prevent partial transformations.
#  - Potential data leakage is reduced by ensuring target-based encoders are only fit on training data.
# """
# 
# def handle_categorical_features(df, target_column=None, cardinality_threshold=3, encoders=None,
#                                 saved_column_names=None, id_column=None):
#     """
#     Encodes categorical features. If 'encoders' is provided, it applies pre-fitted encoders 
#     (for prediction). Otherwise, it fits new encoders (during training).
#     """
#     try:
#         logger.info("Starting to handle categorical features...")
#         if df.shape[0] == 0:
#             raise ValueError("DataFrame is empty; cannot encode categorical features.")
# 
#         # Ensure the ID column is not treated as categorical
#         categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
#         if id_column in categorical_columns:
#             categorical_columns.remove(id_column)
#             logger.info(f"Excluded ID column '{id_column}' from categorical processing.")
# 
#         df_encoded = df.copy()
#         if encoders is None:
#             encoders = {}
# 
#         for column in categorical_columns:
#             unique_values = df_encoded[column].nunique()
#             logger.info(f"Processing col '{column}' with {unique_values} unique values.")
# 
#             # Consistent column name
#             consistent_col_name = column.replace(' ', '_').replace("'", "").lower()
#             df_encoded.rename(columns={column: consistent_col_name}, inplace=True)
#             column = consistent_col_name  # update in loop
# 
#             # If we have a pre-fitted encoder, apply it
#             if column in encoders:
#                 encoder = encoders[column]
#                 if isinstance(encoder, OrdinalEncoder):
#                     df_encoded[column] = encoder.transform(df_encoded[[column]])
#                 elif isinstance(encoder, TargetEncoder):
#                     df_encoded[column] = encoder.transform(df_encoded[column])
#                 elif isinstance(encoder, BinaryEncoder):
#                     binary_transformed = encoder.transform(df_encoded[[column]])
#                     df_encoded.drop(columns=[column], inplace=True)
#                     df_encoded = pd.concat([df_encoded, binary_transformed], axis=1)
#                 logger.info(f"Applied pre-fitted encoder to column '{column}'.")
#             else:
#                 # Fitting a new encoder (training scenario)
#                 if unique_values <= cardinality_threshold:
#                     one_hot_encoded = pd.get_dummies(df_encoded[column], prefix=column)
#                     df_encoded.drop(columns=[column], inplace=True)
#                     df_encoded = pd.concat([df_encoded, one_hot_encoded], axis=1)
#                     encoder = None
#                 elif target_column and (target_column in df_encoded.columns):
#                     # Use TargetEncoder for high cardinality categories if training
#                     encoder = TargetEncoder().fit(df_encoded[column], df_encoded[target_column])
#                     df_encoded[column] = encoder.transform(df_encoded[column])
#                 elif unique_values > 100:
#                     # If extremely high cardinality, use binary encoding
#                     encoder = BinaryEncoder().fit(df_encoded[[column]])
#                     binary_transformed = encoder.transform(df_encoded[[column]])
#                     df_encoded.drop(columns=[column], inplace=True)
#                     df_encoded = pd.concat([df_encoded, binary_transformed], axis=1)
#                 else:
#                     # Fallback to ordinal encoding
#                     encoder = OrdinalEncoder(
#                         handle_unknown='use_encoded_value',
#                         unknown_value=-1
#                     ).fit(df_encoded[[column]])
#                     df_encoded[column] = encoder.transform(df_encoded[[column]])
# 
#                 if encoder is not None:
#                     encoders[column] = encoder
# 
#         # Convert bool columns to integer
#         bool_columns = df_encoded.select_dtypes(include=['bool']).columns.tolist()
#         for col in bool_columns:
#             logger.info(f"Converting boolean column '{col}' to integers.")
#             df_encoded[col] = df_encoded[col].astype(int)
# 
#         # Align columns with training data if saved_column_names is provided
#         if saved_column_names:
#             logger.info("Aligning columns with training data columns.")
#             if id_column in saved_column_names:
#                 saved_column_names = [c for c in saved_column_names if c != id_column]
#             missing_cols = [c for c in saved_column_names if c not in df_encoded.columns]
#             for mc in missing_cols:
#                 df_encoded[mc] = 0
#             df_encoded = df_encoded[saved_column_names]
# 
#         logger.info("Completed handling of categorical features.")
#         return df_encoded, encoders
# 
#     except Exception as e:
#         logger.error(f"Error in handle_categorical_features: {e}")
#         raise
# 
# =============================================================================



#v3

from sklearn.experimental import enable_iterative_imputer  # Required to enable IterativeImputer
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import BinaryEncoder, TargetEncoder
import pandas as pd
import numpy as np
from src.logging_config import get_logger

logger = get_logger(__name__)

"""
This module handles categorical features with one-hot, ordinal, binary, or target encoding.
Improvements:
 - Explicitly identifies categorical columns, excluding date-like and boolean columns.
 - Prevents data leakage by fitting target encoders only on training data.
 - Ensures reproducibility by aligning with saved column names and handling unknown categories.
 - Optimized for scalability with efficient encoding strategies.
 - Fixed AttributeError by safely handling boolean detection.
"""

def handle_categorical_features(df, target_column=None, cardinality_threshold=3, encoders=None,
                               saved_column_names=None, id_column=None, training=True):
    """
    Encodes categorical features. If 'encoders' is provided, applies pre-fitted encoders 
    (for prediction). Otherwise, fits new encoders (during training).

    Parameters:
    - df (pd.DataFrame): Input DataFrame.
    - target_column (str, optional): Target column name for target encoding.
    - cardinality_threshold (int): Threshold for one-hot encoding (default: 3).
    - encoders (dict, optional): Pre-fitted encoders for prediction.
    - saved_column_names (list, optional): Column names from training to align with.
    - id_column (str, optional): ID column to exclude from encoding.
    - training (bool): Whether in training mode (default: True).

    Returns:
    - tuple: (encoded_df, encoders) - Encoded DataFrame and fitted encoders (if training).
    """
    try:
        logger.info("Starting to handle categorical features...")
        if df.shape[0] == 0:
            raise ValueError("DataFrame is empty; cannot encode categorical features.")

        # Ensure the ID column is not treated as categorical
        potential_categorical = df.columns[df.dtypes == 'object'].tolist()
        if id_column in potential_categorical:
            potential_categorical.remove(id_column)
            logger.info(f"Excluded ID column '{id_column}' from categorical processing.")

        # Identify true categorical columns, excluding date-like and boolean columns
        categorical_columns = []
        for col in potential_categorical:
            # Check if column looks like a date
            if pd.to_datetime(df[col], errors='coerce').notna().all():
                logger.warning(f"Column '{col}' appears to be a date; skipping categorical encoding.")
                continue
            # Safely check for boolean-like columns
            if df[col].dtype.name in ['bool', 'boolean']:
                logger.info(f"Column '{col}' identified as boolean; converting to integer.")
                df[col] = df[col].astype(int)  # Convert True/False to 1/0
                continue
            elif df[col].dtype == 'object':
                # Check for string "true"/"false" without .str if not all strings
                try:
                    if df[col].dropna().astype(str).str.lower().isin(['true', 'false']).all():
                        logger.info(f"Column '{col}' identified as string boolean; converting to integer.")
                        df[col] = df[col].map({'true': 1, 'false': 0}).fillna(0).astype(int)
                        continue
                except AttributeError:
                    logger.warning(f"Column '{col}' contains non-string values; skipping boolean check.")
            categorical_columns.append(col)

        logger.info(f"Identified categorical columns: {categorical_columns}")

        df_encoded = df.copy()
        if encoders is None:
            encoders = {}

        for column in categorical_columns:
            unique_values = df_encoded[column].nunique()
            logger.info(f"Processing column '{column}' with {unique_values} unique values.")

            # Consistent column name
            consistent_col_name = column.replace(' ', '_').replace("'", "").lower()
            df_encoded.rename(columns={column: consistent_col_name}, inplace=True)
            column = consistent_col_name  # Update in loop

            if not training and column in encoders:
                encoder = encoders[column]
                if isinstance(encoder, OrdinalEncoder):
                    df_encoded[column] = encoder.transform(df_encoded[[column]])
                elif isinstance(encoder, TargetEncoder):
                    df_encoded[column] = encoder.transform(df_encoded[column])
                elif isinstance(encoder, BinaryEncoder):
                    binary_transformed = encoder.transform(df_encoded[[column]])
                    df_encoded.drop(columns=[column], inplace=True)
                    df_encoded = pd.concat([df_encoded, binary_transformed], axis=1)
                logger.info(f"Applied pre-fitted encoder to column '{column}'.")
            else:
                # Fitting a new encoder (training scenario)
                if unique_values <= cardinality_threshold:
                    one_hot_encoded = pd.get_dummies(df_encoded[column], prefix=column, dummy_na=True)
                    df_encoded.drop(columns=[column], inplace=True)
                    df_encoded = pd.concat([df_encoded, one_hot_encoded], axis=1)
                    encoder = None
                elif target_column and training and (target_column in df_encoded.columns):
                    # Fit TargetEncoder only on training data to prevent leakage
                    encoder = TargetEncoder(cols=[column], handle_missing='value', handle_unknown='value')
                    encoder.fit(df_encoded[[column]], df_encoded[target_column])
                    df_encoded[column] = encoder.transform(df_encoded[[column]])
                elif unique_values > 100:
                    # Use BinaryEncoder for high cardinality
                    encoder = BinaryEncoder(cols=[column], handle_missing='value', handle_unknown='value')
                    encoder.fit(df_encoded[[column]])
                    binary_transformed = encoder.transform(df_encoded[[column]])
                    df_encoded.drop(columns=[column], inplace=True)
                    df_encoded = pd.concat([df_encoded, binary_transformed], axis=1)
                else:
                    # Use OrdinalEncoder with robust handling
                    encoder = OrdinalEncoder(
                        handle_unknown='use_encoded_value',
                        unknown_value=-1,
                        encoded_missing_value=-2
                    )
                    encoder.fit(df_encoded[[column]])
                    df_encoded[column] = encoder.transform(df_encoded[[column]])

                if encoder is not None and training:
                    encoders[column] = encoder

        # Convert any remaining boolean columns to integer
        bool_columns = df_encoded.select_dtypes(include=['bool']).columns.tolist()
        for col in bool_columns:
            logger.info(f"Converting boolean column '{col}' to integers.")
            df_encoded[col] = df_encoded[col].astype(int)

        # Align columns with training data if provided
        if saved_column_names:
            logger.info("Aligning columns with training data columns.")
            if id_column in saved_column_names:
                saved_column_names = [c for c in saved_column_names if c != id_column]
            current_cols = df_encoded.columns.tolist()
            missing_cols = [c for c in saved_column_names if c not in current_cols]
            extra_cols = [c for c in current_cols if c not in saved_column_names]
            for mc in missing_cols:
                df_encoded[mc] = 0  # Fill missing columns with 0
            for ec in extra_cols:
                if ec not in [id_column, target_column]:  # Preserve id and target if present
                    df_encoded.drop(columns=[ec], inplace=True)
            df_encoded = df_encoded[saved_column_names]

        logger.info("Completed handling of categorical features.")
        return df_encoded, encoders

    except Exception as e:
        logger.error(f"Error in handle_categorical_features: {e}")
        raise