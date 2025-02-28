# # from typing import Dict, List
# # import boto3
# # import logging
# # import pandas as pd
# # from io import BytesIO
# # import re
# # from django.conf import settings
# # from botocore.exceptions import ClientError
# # from sqlalchemy import create_engine

# # logger = logging.getLogger(__name__)

# # def get_s3_client():
# #     """Create an S3 client with AWS credentials."""
# #     return boto3.client(
# #         's3',
# #         aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
# #         aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
# #         region_name=settings.AWS_S3_REGION_NAME
# #     )

# # def get_glue_client():
# #     """Create a Glue client with AWS credentials."""
# #     return boto3.client(
# #         'glue',
# #         aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
# #         aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
# #         region_name=settings.AWS_S3_REGION_NAME
# #     )

# # def execute_sql_query(query: str) -> pd.DataFrame:
# #     """Execute an Athena SQL query and return results as a DataFrame."""
# #     try:
# #         connection_string = (
# #             f"awsathena+rest://{settings.AWS_ACCESS_KEY_ID}:{settings.AWS_SECRET_ACCESS_KEY}"
# #             f"@athena.{settings.AWS_S3_REGION_NAME}.amazonaws.com:443/pa_user_datafiles_db"
# #             f"?s3_staging_dir={settings.AWS_ATHENA_S3_STAGING_DIR}&catalog_name=AwsDataCatalog"
# #         )
# #         engine = create_engine(connection_string)
# #         df = pd.read_sql_query(query, engine)
# #         logger.info(f"Query executed successfully. Rows returned: {len(df)}")
# #         return df
# #     except Exception as e:
# #         logger.error(f"Failed to execute query: {query}, Error: {str(e)}")
# #         return pd.DataFrame()

# # def infer_column_dtype(series: pd.Series) -> str:
# #     """Infer the data type of a pandas Series (int, bigint, double, boolean, timestamp, string)."""
# #     series = series.dropna()
# #     if series.empty:
# #         return "string"

# #     total_count = len(series)
# #     col_name = series.name.lower() if series.name else ""

# #     # Boolean check
# #     unique_values = set(series.astype(str).str.lower().unique())
# #     boolean_patterns = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}
# #     if len(unique_values) == 2 and unique_values.issubset(boolean_patterns):
# #         return "boolean"

# #     # Date check (timestamp)
# #     date_formats = [
# #         "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
# #         "%m-%d-%Y", "%Y.%m.%d", "%d.%m.%Y",
# #     ]
# #     for fmt in date_formats:
# #         dt_series = pd.to_datetime(series, format=fmt, errors='coerce')
# #         valid_ratio = dt_series.notnull().sum() / total_count
# #         if valid_ratio >= 0.6:
# #             return "timestamp"

# #     # Numeric checks (int, bigint, double)
# #     try:
# #         numeric_series = pd.to_numeric(series, errors='raise')
# #         if (numeric_series % 1 == 0).all():
# #             if numeric_series.min() >= -2147483648 and numeric_series.max() <= 2147483647:
# #                 return "int"
# #             else:
# #                 return "bigint"
# #         else:
# #             return "double"
# #     except ValueError:
# #         pass

# #     return "string"

# # def normalize_column_name(col_name: str) -> str:
# #     """Normalize column names for SQL compatibility."""
# #     normalized = col_name.strip().lower()
# #     normalized = normalized.replace(' ', '_')
# #     normalized = re.sub(r'[^a-z0-9_]', '', normalized)
# #     if normalized and normalized[0].isdigit():
# #         normalized = f'_{normalized}'
# #     return normalized

# # def parse_dates_with_known_formats(series: pd.Series) -> pd.Series:
# #     """Parse dates with known formats, defaulting invalid to '1970-01-01'."""
# #     possible_formats = [
# #         "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
# #         "%m-%d-%Y", "%Y.%m.%d", "%d.%m.%Y",
# #     ]
# #     parsed = pd.to_datetime(series, format=possible_formats[0], errors='coerce')
# #     for fmt in possible_formats[1:]:
# #         parsed = parsed.combine_first(pd.to_datetime(series, format=fmt, errors='coerce'))
# #     return parsed.fillna(pd.Timestamp("1970-01-01"))

# # def standardize_datetime_columns(df: pd.DataFrame, schema: List[Dict]) -> pd.DataFrame:
# #     """Standardize timestamp columns to ISO format."""
# #     for col_info in schema:
# #         if col_info["data_type"] == "timestamp":
# #             col_name = col_info["column_name"]
# #             df[col_name] = parse_dates_with_known_formats(df[col_name]).dt.strftime("%Y-%m-%d %H:%M:%S")
# #     return df




# import os
# import re
# import logging
# import boto3
# import pandas as pd
# from io import BytesIO
# from typing import Dict, List
# from botocore.exceptions import ClientError
# from sqlalchemy import create_engine

# logger = logging.getLogger(__name__)

# # ------------------------------------------------------------------------------
# # 1. Load Environment Variables at the Top
# # ------------------------------------------------------------------------------
# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')

# # If you prefer to store the Athena DB/Schema name in an env var, do so:
# ATHENA_SCHEMA_NAME = os.getenv('ATHENA_SCHEMA_NAME', 'pa_user_datafiles_db')


# # ------------------------------------------------------------------------------
# # 2. Utility Functions
# # ------------------------------------------------------------------------------

# def get_s3_client():
#     """
#     Create an S3 client with AWS credentials from environment variables.
#     """
#     return boto3.client(
#         's3',
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_S3_REGION_NAME
#     )


# def get_glue_client():
#     """
#     Create a Glue client with AWS credentials from environment variables.
#     """
#     return boto3.client(
#         'glue',
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_S3_REGION_NAME
#     )


# def execute_sql_query(query: str) -> pd.DataFrame:
#     """
#     Execute an Athena SQL query and return results as a DataFrame.

#     Relies on environment variables to build the connection string:
#       - AWS_ACCESS_KEY_ID
#       - AWS_SECRET_ACCESS_KEY
#       - AWS_S3_REGION_NAME
#       - AWS_ATHENA_S3_STAGING_DIR
#       - ATHENA_SCHEMA_NAME
#     """
#     try:
#         connection_string = (
#             f"awsathena+rest://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}"
#             f"@athena.{AWS_S3_REGION_NAME}.amazonaws.com:443/{ATHENA_SCHEMA_NAME}"
#             f"?s3_staging_dir={AWS_ATHENA_S3_STAGING_DIR}&catalog_name=AwsDataCatalog"
#         )

#         engine = create_engine(connection_string)
#         df = pd.read_sql_query(query, engine)
#         logger.info(f"Query executed successfully. Rows returned: {len(df)}")
#         return df
#     except Exception as e:
#         logger.error(f"Failed to execute query: {query}, Error: {str(e)}")
#         return pd.DataFrame()


# def infer_column_dtype(series: pd.Series) -> str:
#     """
#     Infer the data type of a pandas Series:
#     - int
#     - bigint
#     - double
#     - boolean
#     - timestamp
#     - string
#     """
#     series = series.dropna()
#     if series.empty:
#         return "string"

#     total_count = len(series)
#     unique_values = set(series.astype(str).str.lower().unique())

#     # Check for boolean-like patterns
#     boolean_patterns = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}
#     if len(unique_values) == 2 and unique_values.issubset(boolean_patterns):
#         return "boolean"

#     # Check for date/time formats
#     date_formats = [
#         "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
#         "%m-%d-%Y", "%Y.%m.%d", "%d.%m.%Y",
#     ]
#     for fmt in date_formats:
#         dt_series = pd.to_datetime(series, format=fmt, errors='coerce')
#         valid_ratio = dt_series.notnull().sum() / total_count
#         if valid_ratio >= 0.6:
#             return "timestamp"

#     # Check for numeric (int/bigint/double)
#     try:
#         numeric_series = pd.to_numeric(series, errors='raise')
#         if (numeric_series % 1 == 0).all():
#             if numeric_series.min() >= -2147483648 and numeric_series.max() <= 2147483647:
#                 return "int"
#             else:
#                 return "bigint"
#         else:
#             return "double"
#     except ValueError:
#         pass

#     # Otherwise treat as string
#     return "string"


# def normalize_column_name(col_name: str) -> str:
#     """
#     Normalize column names for SQL compatibility.
#     """
#     normalized = col_name.strip().lower()
#     normalized = normalized.replace(' ', '_')
#     normalized = re.sub(r'[^a-z0-9_]', '', normalized)
#     if normalized and normalized[0].isdigit():
#         normalized = f'_{normalized}'
#     return normalized


# def parse_dates_with_known_formats(series: pd.Series) -> pd.Series:
#     """
#     Parse dates with known formats, defaulting invalid to '1970-01-01'.
#     """
#     possible_formats = [
#         "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
#         "%m-%d-%Y", "%Y.%m.%d", "%d.%m.%Y",
#     ]
#     # First parse attempt with the first format
#     parsed = pd.to_datetime(series, format=possible_formats[0], errors='coerce')
#     # Then combine_first for each subsequent format
#     for fmt in possible_formats[1:]:
#         parsed = parsed.combine_first(pd.to_datetime(series, format=fmt, errors='coerce'))
#     return parsed.fillna(pd.Timestamp("1970-01-01"))


# def standardize_datetime_columns(df: pd.DataFrame, schema: List[Dict]) -> pd.DataFrame:
#     """
#     Standardize timestamp columns to ISO format (YYYY-MM-DD HH:MM:SS).
#     """
#     for col_info in schema:
#         if col_info["data_type"] == "timestamp":
#             col_name = col_info["column_name"]
#             df[col_name] = parse_dates_with_known_formats(df[col_name]).dt.strftime("%Y-%m-%d %H:%M:%S")
#     return df



import os
import re
import logging
import boto3
import pandas as pd
from io import BytesIO
from typing import Dict, List
from botocore.exceptions import ClientError
from sqlalchemy import create_engine

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# 1. Load Environment Variables at the Top
# ------------------------------------------------------------------------------
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')

# If you prefer to store the Athena DB/Schema name in an env var, do so:
ATHENA_SCHEMA_NAME = os.getenv('ATHENA_SCHEMA_NAME', 'pa_user_datafiles_db')

# ------------------------------------------------------------------------------
# 2. Utility Functions
# ------------------------------------------------------------------------------

def get_s3_client():
    """
    Create an S3 client with AWS credentials from environment variables.
    """
    logger.debug("[DEBUG] Creating S3 client...")
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION_NAME
    )

def get_glue_client():
    """
    Create a Glue client with AWS credentials from environment variables.
    """
    logger.debug("[DEBUG] Creating Glue client...")
    return boto3.client(
        'glue',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION_NAME
    )

def execute_sql_query(query: str) -> pd.DataFrame:
    """
    Execute an Athena SQL query and return results as a DataFrame.

    Relies on environment variables to build the connection string:
      - AWS_ACCESS_KEY_ID
      - AWS_SECRET_ACCESS_KEY
      - AWS_S3_REGION_NAME
      - AWS_ATHENA_S3_STAGING_DIR
      - ATHENA_SCHEMA_NAME
    """
    try:
        if not AWS_ATHENA_S3_STAGING_DIR:
            raise ValueError("AWS_ATHENA_S3_STAGING_DIR is not set.")
        connection_string = (
            f"awsathena+rest://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}"
            f"@athena.{AWS_S3_REGION_NAME}.amazonaws.com:443/{ATHENA_SCHEMA_NAME}"
            f"?s3_staging_dir={AWS_ATHENA_S3_STAGING_DIR}&catalog_name=AwsDataCatalog"
        )
        logger.debug(f"[DEBUG] Athena connection string created: {connection_string}")
        engine = create_engine(connection_string)
        df = pd.read_sql_query(query, engine)
        logger.info(f"Query executed successfully. Rows returned: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Failed to execute query: {query}, Error: {str(e)}")
        return pd.DataFrame()

def infer_column_dtype(series: pd.Series, threshold: float = 0.6) -> str:
    """
    Infer the data type of a pandas Series:
    - int
    - bigint
    - double
    - boolean
    - timestamp
    - string
    """
    series = series.dropna()
    if series.empty:
        logger.debug("[DEBUG] Column is empty after dropna, defaulting to string")
        return "string"
    
    total_count = len(series)
    col_name = series.name.lower() if series.name else ""

    # Boolean check first (numeric 0/1 or true/false equivalents)
    unique_values = set(series.astype(str).str.lower().unique())
    boolean_patterns = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}
    
    # Try numeric conversion to check for 0/1
    try:
        numeric_series = pd.to_numeric(series, errors='raise')
        unique_nums = set(numeric_series.unique())
        if len(unique_nums) == 2 and {0, 1}.issubset(unique_nums) and not (numeric_series < 0).any() and not (numeric_series > 1).any():
            logger.debug("[DEBUG] Detected as boolean (numeric 0/1)")
            return "boolean"
    except ValueError:
        pass  # Proceed to string-based boolean check if not numeric

    # Check string patterns for exactly 2 unique values
    if len(unique_values) == 2 and unique_values.issubset(boolean_patterns):
        if unique_values.issubset({'true', 'false'}):
            logger.debug("[DEBUG] Detected as boolean (true/false)")
            return "boolean"
        if unique_values.issubset({'yes', 'no'}):
            logger.debug("[DEBUG] Detected as boolean (yes/no)")
            return "boolean"
        if unique_values.issubset({'t', 'f'}):
            logger.debug("[DEBUG] Detected as boolean (t/f)")
            return "boolean"
        if unique_values.issubset({'y', 'n'}):
            logger.debug("[DEBUG] Detected as boolean (y/n)")
            return "boolean"
        if unique_values.issubset({'1', '0'}):
            logger.debug("[DEBUG] Detected as boolean (string 0/1)")
            return "boolean"

    # Date formats (only if column name suggests date or high validity)
    date_formats = [
        "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
        "%m-%d-%Y", "%Y.%m.%d", "%d.%m.%Y",
        "%b %d %Y", "%d %b %Y", "%Y %b %d",
        "%B %d %Y", "%d %B %Y", "%Y %B %d",
        "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
        "%Y%m%d", "%d%m%Y", "%m%d%Y"
    ]
    
    valid_masks = {}
    for fmt in date_formats:
        dt_series = pd.to_datetime(series, format=fmt, errors='coerce')
        valid = dt_series.notnull()
        valid_count = valid.sum()
        valid_ratio = valid_count / total_count
        # Ensure dates are within a reasonable range (1900-2100)
        if valid_count > 0:
            valid &= (dt_series >= pd.Timestamp("1900-01-01")) & (dt_series <= pd.Timestamp("2100-12-31"))
            valid_count = valid.sum()
            valid_ratio = valid_count / total_count
        logger.debug(f"[DEBUG] Format {fmt}: Valid {valid_count}/{total_count} ({valid_ratio:.2f})")
        valid_masks[fmt] = valid
        if valid_ratio >= threshold:
            logger.debug(f"[DEBUG] Detected as timestamp with format {fmt}")
            return "timestamp"

    # Combine valid date parses
    combined_valid = pd.Series(False, index=series.index)
    for mask in valid_masks.values():
        combined_valid |= mask
    combined_valid_count = combined_valid.sum()
    combined_valid_ratio = combined_valid_count / total_count
    logger.debug(f"[DEBUG] Combined date formats: Valid {combined_valid_count}/{total_count} ({combined_valid_ratio:.2f})")
    if combined_valid_ratio >= threshold and ("date" in col_name or combined_valid_ratio > 0.9):
        logger.debug("[DEBUG] Detected as timestamp with combined formats")
        return "timestamp"

    # Flexible parse only if column name suggests date
    if "date" in col_name:
        dt_series_flex = pd.to_datetime(series, errors='coerce')
        valid_flex = dt_series_flex.notnull() & (dt_series_flex >= pd.Timestamp("1900-01-01")) & (dt_series_flex <= pd.Timestamp("2100-12-31"))
        valid_count_flex = valid_flex.sum()
        valid_ratio_flex = valid_count_flex / total_count
        logger.debug(f"[DEBUG] Flexible parse: Valid {valid_count_flex}/{total_count} ({valid_ratio_flex:.2f})")
        if valid_ratio_flex >= threshold:
            logger.debug("[DEBUG] Detected as timestamp with flexible parse")
            return "timestamp"

    # Numeric checks (after boolean and date)
    try:
        numeric_series = pd.to_numeric(series, errors='raise')
        if (numeric_series % 1 == 0).all():
            if numeric_series.min() >= -2147483648 and numeric_series.max() <= 2147483647:
                logger.debug(f"[DEBUG] Detected as int ({numeric_series.min()} to {numeric_series.max()})")
                return "int"
            else:
                logger.debug(f"[DEBUG] Detected as bigint ({numeric_series.min()} to {numeric_series.max()})")
                return "bigint"
        else:
            logger.debug(f"[DEBUG] Detected as double ({numeric_series.min()} to {numeric_series.max()})")
            return "double"
    except ValueError:
        logger.debug("[DEBUG] Not numeric")

    logger.debug("[DEBUG] Defaulting to string")
    return "string"

def normalize_column_name(col_name: str) -> str:
    """
    Normalize column names for SQL compatibility.
    """
    normalized = col_name.strip().lower()
    normalized = normalized.replace(' ', '_')
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    if normalized and normalized[0].isdigit():
        normalized = f'_{normalized}'
    return normalized

def parse_dates_with_known_formats(series: pd.Series, possible_formats=None, dayfirst=False) -> pd.Series:
    """
    Parse dates with known formats, defaulting invalid to '1970-01-01'.
    Supports multiple formats and prioritizes day-first or year-first parsing.
    """
    if possible_formats is None:
        possible_formats = [
            "%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d",
            "%m-%d-%Y", "%Y.%m.%d", "%d.%m.%Y",
            "%b %d %Y", "%d %b %Y", "%Y %b %d",
            "%B %d %Y", "%d %B %Y", "%Y %B %d",
            "%Y-%m-%d %H:%M:%S", "%d-%m-%Y %H:%M:%S", "%m/%d/%Y %H:%M:%S",
            "%Y%m%d", "%d%m%Y", "%m%d%Y"
        ]

    total_count = len(series)
    parsed_series = pd.Series(pd.NaT, index=series.index)
    valid_masks = {}

    for fmt in possible_formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce", dayfirst=dayfirst)
        valid = parsed.notnull() & (parsed >= pd.Timestamp("1900-01-01")) & (parsed <= pd.Timestamp("2100-12-31"))
        valid_count = valid.sum()
        logger.debug(f"[DEBUG] Parsing with {fmt} (dayfirst={dayfirst}): Valid {valid_count}/{total_count}")
        valid_masks[fmt] = valid
        parsed_series = parsed_series.where(parsed_series.notnull(), parsed)

    combined_valid = pd.Series(False, index=series.index)
    for mask in valid_masks.values():
        combined_valid |= mask
    nulls = total_count - combined_valid.sum()
    logger.debug(f"[DEBUG] Combined parse: Nulls {nulls}/{total_count}")

    if nulls > (0.5 * total_count) and "date" in series.name.lower():
        parsed_flex = pd.to_datetime(series, errors="coerce", dayfirst=dayfirst)
        valid_flex = parsed_flex.notnull() & (parsed_flex >= pd.Timestamp("1900-01-01")) & (parsed_flex <= pd.Timestamp("2100-12-31"))
        valid_count_flex = valid_flex.sum()
        logger.debug(f"[DEBUG] Flexible parse fallback (dayfirst={dayfirst}): Valid {valid_count_flex}/{total_count}")
        parsed_series = parsed_series.where(parsed_series.notnull(), parsed_flex)

    parsed_final = parsed_series.fillna(pd.Timestamp("1970-01-01"))
    logger.debug(f"[DEBUG] Final nulls filled with '1970-01-01': Nulls {parsed_final.isnull().sum()}/{total_count}")
    return parsed_final

def standardize_datetime_columns(df: pd.DataFrame, schema: List[Dict]) -> pd.DataFrame:
    """
    Standardize timestamp columns to ISO format (YYYY-MM-DD HH:MM:SS).
    Uses multi-format date parsing to handle various input formats.
    """
    for col_info in schema:
        if col_info["data_type"] == "timestamp":
            col_name = col_info["column_name"]
            logger.debug(f"[DEBUG] Parsing dates in column '{col_name}' with multi-format logic...")
            try:
                # Use parse_dates_with_known_formats with dayfirst=True for dd-mm-yyyy preference
                parsed_dates = parse_dates_with_known_formats(df[col_name], dayfirst=True)
                # Convert final datetimes to ISO string
                df[col_name] = parsed_dates.dt.strftime("%Y-%m-%d %H:%M:%S")

                # Log how many rows ended up as 1970-01-01
                num_1970 = (df[col_name] == "1970-01-01 00:00:00").sum()
                logger.debug(f"[DEBUG] Done parsing '{col_name}'. Rows with 1970-01-01 = {num_1970}")
            except Exception as e:
                logger.warning(f"[WARNING] Could not standardize date column '{col_name}': {str(e)}")
    return df