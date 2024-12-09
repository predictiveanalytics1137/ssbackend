
def normalize_column_names(df):
    """
    Normalizes column names by replacing special characters with meaningful placeholders.
    Ensures consistency between training and prediction phases.

    Parameters:
    - df: pandas DataFrame with column names to normalize.

    Returns:
    - df: pandas DataFrame with normalized column names.
    """
    normalized_columns = {
        col: col.replace("+", "_plus_")
               .replace("-", "_minus_")
               .replace("/", "_divide_")
               .replace("*", "_multiply_")
               .replace(" ", "_")
               .replace("'", "")
               for col in df.columns
    }
    df.rename(columns=normalized_columns, inplace=True)
    return df

