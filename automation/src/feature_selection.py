
from typing import List, Optional
import pandas as pd
import numpy as np
from sklearn.discriminant_analysis import StandardScaler
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
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List, Optional
from src.logging_config import get_logger

logger = get_logger(__name__)

def feature_selection(
    df: pd.DataFrame,
    target_column: str,
    time_column: str,
    task: str = "regression",
    variance_threshold: float = 0.01,
    corr_threshold: float = 0.9,
    max_features: int = 20,
    id_column: Optional[str] = None,
    exclude_columns: Optional[List[str]] = None,
    cardinality_threshold: float = 0.5
) -> Tuple[pd.DataFrame, List[str]]:
    try:
        logger.info("Starting feature selection...")

        if df.empty:
            raise ValueError("DataFrame is empty.")
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found.")
        time_data = df[time_column].copy() if time_column in df.columns else None

        high_cardinality_cols = [
            col for col in df.columns 
            if df[col].nunique() > cardinality_threshold * len(df) and col != target_column
        ]
        exclude_columns = list(set((exclude_columns or []) + high_cardinality_cols + [time_column]))
        if id_column and id_column in df.columns:
            exclude_columns.append(id_column)
        logger.info(f"Excluded columns: {exclude_columns}")

        id_data = df[id_column].copy() if id_column in df.columns else None
        columns_to_drop = [target_column] + [col for col in exclude_columns if col in df.columns]
        X = df.drop(columns=columns_to_drop, errors="ignore")
        y = df[target_column]
        logger.info(f"Initial feature count: {X.shape[1]}")

        # Convert categorical features to numeric
        for col in X.select_dtypes(include=["category"]).columns:
            X[col] = X[col].cat.codes
            logger.info(f"Converted categorical column '{col}' to numeric (cat.codes).")

        # Variance Threshold
        X_numeric = X.select_dtypes(include=["float64", "int64"])
        X_non_numeric = X.select_dtypes(exclude=["float64", "int64"])
        if X_numeric.empty:
            X_var_filtered = X_non_numeric
        else:
            variance_selector = VarianceThreshold(threshold=variance_threshold)
            X_var_filtered = variance_selector.fit_transform(X_numeric)
            retained_cols = X_numeric.columns[variance_selector.get_support()]
            X_var_filtered = pd.DataFrame(X_var_filtered, columns=retained_cols, index=X.index)
        X_var_filtered = pd.concat([X_var_filtered, X_non_numeric], axis=1)
        logger.info(f"Features after variance filter: {X_var_filtered.shape[1]}")

        # Correlation Filter
        X_numeric_corr = X_var_filtered.select_dtypes(include=["float64", "int64"])
        if X_numeric_corr.empty:
            X_corr_filtered = X_var_filtered
        else:
            corr_matrix = X_numeric_corr.corr().abs()
            upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
            X_corr_filtered = X_var_filtered.drop(columns=to_drop, errors="ignore")
            logger.info(f"Dropped {len(to_drop)} correlated features: {to_drop}")
        logger.info(f"Features after correlation filter: {X_corr_filtered.shape[1]}")

        # Embedded Method (No Scaling Here)
        model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1) if task == "regression" else \
                RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        selector = SelectFromModel(model, max_features=max_features, threshold="median", prefit=False)
        selector.fit(X_corr_filtered, y)
        selected_cols = X_corr_filtered.columns[selector.get_support()].tolist()
        X_final = X_corr_filtered[selected_cols]
        logger.info(f"Selected {len(selected_cols)} features: {selected_cols}")

        selected_features = [col for col in selected_cols if col not in exclude_columns]
        if not selected_features:
            logger.warning("No features selected; falling back to top 5 by importance.")
            model.fit(X_corr_filtered, y)
            importances = pd.Series(model.feature_importances_, index=X_corr_filtered.columns)
            selected_cols = importances.nlargest(5).index.tolist()
            X_final = X_corr_filtered[selected_cols]
            selected_features = selected_cols

        # Combine with target and reattach time_column and id_column
        df_final = pd.concat([X_final, y.reset_index(drop=True)], axis=1)
        if time_data is not None:
            df_final[time_column] = time_data.reindex(df_final.index)
            logger.info(f"Reattached time column '{time_column}' to final DataFrame.")
        if id_data is not None:
            df_final[id_column] = id_data.reindex(df_final.index)
            logger.info(f"Reattached ID column '{id_column}' to final DataFrame.")

        logger.info(f"Feature selection complete. Final shape: {df_final.shape}")
        return df_final, selected_features

    except Exception as e:
        logger.error(f"Error during feature selection: {str(e)}", exc_info=True)
        raise