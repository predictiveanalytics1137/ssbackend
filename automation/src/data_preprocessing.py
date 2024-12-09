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


def handle_categorical_features(df, target_column=None, cardinality_threshold=3, encoders=None, saved_column_names=None):
    logger.info("Starting to handle categorical features...")

    try:
        # Identify categorical columns
        categorical_columns = df.select_dtypes(include=['object']).columns.tolist()
        logger.info(f"Categorical columns identified: {categorical_columns}")

        df_encoded = df.copy()
        if encoders is None:
            encoders = {}

        all_encoded_columns = []

        for column in categorical_columns:
            unique_values = df[column].nunique()
            logger.info(f"Processing column '{column}' with {unique_values} unique values.")

            # Ensure consistent column name (replace spaces with underscores and convert to lowercase)
            consistent_column_name = column.replace(' ', '_').replace("'", "").lower()
            df_encoded.rename(columns={column: consistent_column_name}, inplace=True)
            column = consistent_column_name  # Update the column name in the loop

            # Apply encoding
            if column in encoders:
                encoder = encoders[column]
                # Transform the column using the preloaded encoder
                if isinstance(encoder, OrdinalEncoder):
                    df_encoded[column] = encoder.transform(df_encoded[[column]])
                elif isinstance(encoder, TargetEncoder):
                    df_encoded[column] = encoder.transform(df_encoded[column])
                elif isinstance(encoder, BinaryEncoder):
                    binary_transformed = encoder.transform(df_encoded[[column]])
                    df_encoded = pd.concat([df_encoded.drop(column, axis=1), binary_transformed], axis=1)
                logger.info(f"Applied preloaded encoder to column '{column}'.")
            else:
                # Fit and transform using a new encoder during training
                if unique_values <= cardinality_threshold:
                    one_hot_encoded = pd.get_dummies(df[column], prefix=column)
                    all_encoded_columns.extend(one_hot_encoded.columns)
                    df_encoded = pd.concat([df_encoded.drop(column, axis=1), one_hot_encoded], axis=1)
                    encoder = None
                elif target_column and target_column in df.columns:
                    encoder = TargetEncoder().fit(df[column], df[target_column])
                    df_encoded[column] = encoder.transform(df[column])
                elif unique_values > 100:
                    encoder = BinaryEncoder().fit(df[[column]])
                    binary_transformed = encoder.transform(df[[column]])
                    all_encoded_columns.extend(binary_transformed.columns)
                    df_encoded = pd.concat([df_encoded.drop(column, axis=1), binary_transformed], axis=1)
                else:
                    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1).fit(df[[column]])
                    df_encoded[column] = encoder.transform(df[[column]])

                if encoder is not None:
                    encoders[column] = encoder

        # Convert boolean columns to 0 and 1
        bool_columns = df_encoded.select_dtypes(include=['bool']).columns.tolist()
        if bool_columns:
            df_encoded[bool_columns] = df_encoded[bool_columns].replace({True: 1, False: 0})

        logger.info("Completed handling of categorical features.")

        # Align columns with the training data, ensuring that any missing columns are added
        if saved_column_names:
            logger.info(f"Aligning columns with training data: {saved_column_names}")
            missing_cols = [col for col in saved_column_names if col not in df_encoded.columns]
            for col in missing_cols:
                df_encoded[col] = 0
            df_encoded = df_encoded[saved_column_names]

        return df_encoded, encoders

    except Exception as e:
        logger.error(f"Error while handling categorical features: {e}")
        raise
