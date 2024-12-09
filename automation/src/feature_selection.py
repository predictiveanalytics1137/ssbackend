
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.base import is_classifier
import joblib
from src.logging_config import get_logger

logger = get_logger(__name__)

def feature_selection(df, target_column, task="regression", variance_threshold=0.01, corr_threshold=0.9, top_n_features=15, save_path=None):
    """
    Performs sophisticated feature selection using filter, embedded, and wrapper methods.

    Parameters:
    - df: pandas DataFrame containing the dataset (features + target).
    - target_column: string, the name of the target column.
    - task: string, either "regression" or "classification".
    - variance_threshold: float, threshold to remove low-variance features.
    - corr_threshold: float, threshold for removing highly correlated features.
    - top_n_features: int, optional, number of top features to select in the final step.
    - save_path: string, optional, path to save selected feature names.

    Returns:
    - df_selected: pandas DataFrame with selected features.
    - selected_features: list of selected feature names.
    """
    try:
        logger.info("Starting feature selection...")
        
        # 1. Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 2. Filter Method: Remove low-variance features
        logger.info("Applying VarianceThreshold to remove low-variance features...")
        variance_selector = VarianceThreshold(threshold=variance_threshold)
        X_var_filtered = pd.DataFrame(
            variance_selector.fit_transform(X), 
            columns=X.columns[variance_selector.get_support()]
        )
        logger.info(f"Features remaining after VarianceThreshold: {X_var_filtered.shape[1]}")

        # 3. Filter Method: Remove highly correlated features
        logger.info("Removing highly correlated features...")
        corr_matrix = X_var_filtered.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        high_corr_features = [col for col in upper_tri.columns if any(upper_tri[col] > corr_threshold)]
        X_corr_filtered = X_var_filtered.drop(columns=high_corr_features)
        logger.info(f"Features remaining after correlation filter: {X_corr_filtered.shape[1]}")

        # 4. Embedded Method: Feature importance using Lasso or RandomForest
        logger.info("Applying embedded feature selection...")
        if task == "regression":
            model = Lasso(alpha=0.01, random_state=42)
        else:
            model = RandomForestClassifier(random_state=42)

        model.fit(X_corr_filtered, y)
        embedded_selector = SelectFromModel(model, prefit=True)
        X_embedded_filtered = pd.DataFrame(
            embedded_selector.transform(X_corr_filtered),
            columns=X_corr_filtered.columns[embedded_selector.get_support()]
        )
        logger.info(f"Features remaining after embedded method: {X_embedded_filtered.shape[1]}")

        # 5. Wrapper Method: Recursive Feature Elimination (optional for top features)
        if top_n_features:
            logger.info(f"Applying wrapper method (RFE) to select top {top_n_features} features...")
            rfe_model = RandomForestRegressor(random_state=42) if task == "regression" else RandomForestClassifier(random_state=42)
            rfe_selector = RFE(rfe_model, n_features_to_select=top_n_features, step=1)
            X_rfe_filtered = pd.DataFrame(
                rfe_selector.fit_transform(X_embedded_filtered, y),
                columns=X_embedded_filtered.columns[rfe_selector.get_support()]
            )
            logger.info(f"Features remaining after RFE: {X_rfe_filtered.shape[1]}")
            selected_features = X_rfe_filtered.columns.tolist()
            X_final = X_rfe_filtered
        else:
            selected_features = X_embedded_filtered.columns.tolist()
            X_final = X_embedded_filtered
            
        # Recombine selected features with the target column
        df_final = pd.concat([X_final, y.reset_index(drop=True)], axis=1)

        # 6. Save selected features (optional)
        # if save_path:
        #     logger.info(f"Saving selected features to {save_path}...")
        #     joblib.dump(selected_features, save_path)

        logger.info("Feature selection complete.")
        return df_final, selected_features

    except Exception as e:
        logger.error(f"Error during feature selection: {e}")
        raise
