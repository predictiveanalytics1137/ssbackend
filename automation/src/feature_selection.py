
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import is_classifier
import joblib
from src.logging_config import get_logger

logger = get_logger(__name__)


# def feature_selection(df, target_column, task="regression", variance_threshold=0.01, corr_threshold=0.9, top_n_features=15, save_path=None, id_column=None):
#     """
#     Performs sophisticated feature selection using filter, embedded, and wrapper methods.

#     Parameters:
#     - df: pandas DataFrame containing the dataset (features + target).
#     - target_column: string, the name of the target column.
#     - task: string, either "regression" or "classification".
#     - variance_threshold: float, threshold to remove low-variance features.
#     - corr_threshold: float, threshold for removing highly correlated features.
#     - top_n_features: int, optional, number of top features to select in the final step.
#     - save_path: string, optional, path to save selected feature names.

#     Returns:
#     - df_selected: pandas DataFrame with selected features.
#     - selected_features: list of selected feature names.
#     """
#     try:
#         logger.info("Starting feature selection...")
        
#         # Exclude the ID column if provided
#         if id_column and id_column in df.columns:
#             id_data = df[id_column].copy()
#             df = df.drop(columns=[id_column])
#             logger.info(f"Excluded entity column '{id_column}' from feature selection.")
        
#         # 1. Separate features and target
#         X = df.drop(columns=[target_column])
#         y = df[target_column]

#         # 2. Filter Method: Remove low-variance features
#         logger.info("Applying VarianceThreshold to remove low-variance features...")
#         variance_selector = VarianceThreshold(threshold=variance_threshold)
#         X_var_filtered = pd.DataFrame(
#             variance_selector.fit_transform(X), 
#             columns=X.columns[variance_selector.get_support()]
#         )
#         logger.info(f"Features remaining after VarianceThreshold: {X_var_filtered.shape[1]}")

#         # 3. Filter Method: Remove highly correlated features
#         logger.info("Removing highly correlated features...")
#         corr_matrix = X_var_filtered.corr().abs()
#         upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
#         high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
#         X_corr_filtered = X_var_filtered.drop(columns=high_corr_features)
#         logger.info(f"Features remaining after correlation filter: {X_corr_filtered.shape[1]}")

#         # 4. Embedded Method: Feature importance using Lasso or RandomForest
#         logger.info("Applying embedded feature selection...")
#         if task == "regression":
#             model = Lasso(alpha=0.01, random_state=42)
#         else:
#             model = RandomForestClassifier(random_state=42)

#         model.fit(X_corr_filtered, y)
#         embedded_selector = SelectFromModel(model, prefit=True)
#         X_embedded_filtered = pd.DataFrame(
#             embedded_selector.transform(X_corr_filtered),
#             columns=X_corr_filtered.columns[embedded_selector.get_support()]
#         )
#         logger.info(f"Features remaining after embedded method: {X_embedded_filtered.shape[1]}")

#         # 5. Wrapper Method: Recursive Feature Elimination (optional for top features)
#         if top_n_features:
#             logger.info(f"Applying wrapper method (RFE) to select top {top_n_features} features...")
#             rfe_model = RandomForestRegressor(random_state=42) if task == "regression" else RandomForestClassifier(random_state=42)
#             rfe_selector = RFE(rfe_model, n_features_to_select=top_n_features, step=1)
#             X_rfe_filtered = pd.DataFrame(
#                 rfe_selector.fit_transform(X_embedded_filtered, y),
#                 columns=X_embedded_filtered.columns[rfe_selector.get_support()]
#             )
#             logger.info(f"Features remaining after RFE: {X_rfe_filtered.shape[1]}")
#             selected_features = X_rfe_filtered.columns.tolist()
#             X_final = X_rfe_filtered
#         else:
#             selected_features = X_embedded_filtered.columns.tolist()
#             X_final = X_embedded_filtered
            
#         # Recombine selected features with the target column
#         df_final = pd.concat([X_final, y.reset_index(drop=True)], axis=1)
        
#         # Add back the ID column if applicable
#         if id_column:
#             df_final[id_column] = id_data
#             logger.info(f"Added back entity column '{id_column}' to the final DataFrame.")

#         # 6. Save selected features (optional)
#         # if save_path:
#         #     logger.info(f"Saving selected features to {save_path}...")
#         #     joblib.dump(selected_features, save_path)

#         logger.info("Feature selection complete.")
#         return df_final, selected_features

#     except Exception as e:
#         logger.error(f"Error during feature selection: {e}")
#         raise



# v2

import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import is_classifier
from sklearn.impute import SimpleImputer
import joblib
from src.logging_config import get_logger

logger = get_logger(__name__)

def feature_selection(df, target_column, task="regression", variance_threshold=0.01,
                      corr_threshold=0.9, top_n_features=15, save_path=None, id_column=None):
    """
    Performs multi-step feature selection: variance, correlation, embedded, RFE.
    """
    try:
        logger.info("Starting feature selection...")

        if df.shape[0] == 0:
            raise ValueError("DataFrame is empty; cannot perform feature selection.")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not in DataFrame.")

        if id_column and id_column in df.columns:
            id_data = df[id_column].copy()
            df = df.drop(columns=[id_column])
            logger.info(f"Excluded ID column '{id_column}' from feature selection.")
        else:
            id_data = None

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 1. Variance Threshold
        logger.info("Applying VarianceThreshold to remove low-variance features...")
        variance_selector = VarianceThreshold(threshold=variance_threshold)
        X_var_filtered = variance_selector.fit_transform(X)
        retained_cols = X.columns[variance_selector.get_support()]
        X_var_filtered = pd.DataFrame(X_var_filtered, columns=retained_cols)
        logger.info(f"Features after variance filter: {X_var_filtered.shape[1]}")

        # 2. Correlation Filter
        logger.info("Removing highly correlated features...")
        corr_matrix = X_var_filtered.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [c for c in upper_tri.columns if any(upper_tri[c] > corr_threshold)]
        X_corr_filtered = X_var_filtered.drop(columns=to_drop)
        logger.info(f"Features after correlation filter: {X_corr_filtered.shape[1]}")

        # 3. Embedded Method (Feature Importances)
        logger.info("Applying embedded feature selection...")
        if task == "regression":
            model = Lasso(alpha=0.01, random_state=42)
            scoring = "neg_mean_squared_error"
        else:
            model = RandomForestClassifier(random_state=42)
            scoring = "accuracy"

        model.fit(X_corr_filtered, y)
        embedded_selector = SelectFromModel(model, prefit=True)
        X_embedded_filtered = embedded_selector.transform(X_corr_filtered)
        embedded_cols = X_corr_filtered.columns[embedded_selector.get_support()]
        X_embedded_filtered = pd.DataFrame(X_embedded_filtered, columns=embedded_cols)
        logger.info(f"Features after embedded method: {X_embedded_filtered.shape[1]}")

        # 4. Wrapper Method (RFE)
        if top_n_features and top_n_features < X_embedded_filtered.shape[1]:
            logger.info(f"Applying RFE for top {top_n_features} features...")
            if task == "regression":
                rfe_model = RandomForestRegressor(random_state=42)
            else:
                rfe_model = RandomForestClassifier(random_state=42)

            rfe_selector = RFE(rfe_model, n_features_to_select=top_n_features, step=1)
            X_rfe = rfe_selector.fit_transform(X_embedded_filtered, y)
            rfe_cols = X_embedded_filtered.columns[rfe_selector.get_support()]
            X_final = pd.DataFrame(X_rfe, columns=rfe_cols)
            selected_features = list(rfe_cols)
        else:
            X_final = X_embedded_filtered
            selected_features = list(X_embedded_filtered.columns)

        # Combine final features with target
        df_final = pd.concat([X_final, y.reset_index(drop=True)], axis=1)

        # Add back ID column
        if id_column and id_data is not None:
            df_final[id_column] = id_data
            logger.info(f"Re-added ID column '{id_column}' to final DataFrame.")

        logger.info(f"Feature selection complete. Final shape: {df_final.shape}")
        return df_final, selected_features

    except Exception as e:
        logger.error(f"Error during feature selection: {e}")
        raise
