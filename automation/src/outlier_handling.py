# src/outlier_handling.py

import numpy as np
import pandas as pd

def detect_and_handle_outliers_train(df, factor=1.5, cols=None):
    """
    Computes Q1, Q3, and IQR-based bounds for each numeric column, 
    then caps outliers in 'df' (training set) accordingly.

    Returns:
     - df_out: The modified DataFrame with outliers capped.
     - outlier_bounds: A dict containing {col: (lower_bound, upper_bound)} 
                       for each numeric column processed.
    """
    df_out = df.copy()
    outlier_bounds = {}

    # If 'cols' isn't specified, grab all numeric columns
    if cols is None:
        cols = df_out.select_dtypes(include=['int64','float64']).columns.tolist()

    for col in cols:
        # Compute quartiles
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - factor * IQR
        upper_bound = Q3 + factor * IQR

        # Store the bounds
        outlier_bounds[col] = (lower_bound, upper_bound)

        # Cap the values
        df_out[col] = np.where(df_out[col] < lower_bound, lower_bound, df_out[col])
        df_out[col] = np.where(df_out[col] > upper_bound, upper_bound, df_out[col])

    return df_out, outlier_bounds


def apply_outlier_bounds(df, outlier_bounds):
    """
    Applies precomputed outlier bounds (from training) to 'df'.
    Caps values outside (lower_bound, upper_bound) for each column 
    that exists in outlier_bounds.

    Returns:
     - df_capped: DataFrame with outliers capped using the stored training bounds.
    """
    df_capped = df.copy()

    for col, (lower_bound, upper_bound) in outlier_bounds.items():
        if col in df_capped.columns:
            df_capped[col] = np.where(df_capped[col] < lower_bound, lower_bound, df_capped[col])
            df_capped[col] = np.where(df_capped[col] > upper_bound, upper_bound, df_capped[col])

    return df_capped
