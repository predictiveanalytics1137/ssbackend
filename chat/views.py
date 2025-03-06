



import logging
logger = logging.getLogger(__name__)

import difflib
import os
import uuid
import datetime
import re
import boto3
import nbformat
import pandas as pd
import openai
import requests
import numpy as np
from collections import defaultdict

# Add these two dictionaries at the module or class level.
# They track whether we are awaiting a â€œconfirm yesâ€ response,
# and any pending column changes the user has not yet confirmed.
awaiting_confirmation = defaultdict(bool)
pending_column_updates = defaultdict(dict)
from io import BytesIO
from typing import Any, Dict, List
from django.db import transaction
from django.contrib.auth.models import User
from django.core.exceptions import ValidationError
from rest_framework import status
from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
from rest_framework.response import Response
from rest_framework.views import APIView
from rest_framework.permissions import IsAuthenticated
from botocore.exceptions import ClientError, NoCredentialsError
from django.conf import settings
from sqlalchemy import create_engine
from langchain.chains import ConversationChain
from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage
from django.http import JsonResponse
from django.views.decorators.http import require_GET
from .models import ChatFileInfo, PredictiveSettings
from .serializers import PredictiveSettingsSerializer

import difflib

from chat.utils import classify_ml_type, gpt_suggest_columns, parse_nlu_input

from .models import FileSchema, Notebook, PredictiveSettings, UploadedFile, ChatBackup
from .serializers import UploadedFileSerializer

# AWS and OpenAI environment variables
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')
AWS_REGION_NAME = AWS_S3_REGION_NAME
ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Adjust as needed
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

# Initialize the ChatOpenAI model
llm_chatgpt = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

# In-memory storage for user-specific data
user_conversations = {}
user_schemas = {}
user_confirmations = {}
user_notebook_flags = {}
user_notebooks = {}

SYSTEM_INSTRUCTIONS = (
    "You are a highly intelligent and helpful PACX AI assistant. Your responses should be clear and concise. "
    "Assist the user in forming the predictive question and any corrections they provide. Reflect the confirmed or "
    "corrected schema back to the user before proceeding, asking one question at a time.\n\n"
    "Steps:\n"
    "1. Discuss the subject they want to predict.\n"
    "2. Confirm the target value they want to predict.\n"
    "3. Check if there's a specific time frame for the prediction (e.g., next month, 6 month, year)\n"
    "4. Determine whether the prediction should occur on a recurring basis (e.g., daily, weekly, monthly) or after a specific event.\n"
    "5. Reference the dataset schema if available; if not, ask to upload it.\n"
    "6. Once all necessary information is confirmed, provide a summary and let the user know they can generate the notebook."
)

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_INSTRUCTIONS),
    HumanMessagePromptTemplate.from_template(
        "Conversation so far:\n{history}\nUser input:\n{user_input}"
    )
])

# -----------------------------------------------------------------------------------
# S3 & Glue Helpers
# -----------------------------------------------------------------------------------
def get_s3_client():
    print("[DEBUG] Creating S3 client...")
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION_NAME
    )

def get_glue_client():
    print("[DEBUG] Creating Glue client...")
    return boto3.client(
        'glue',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION_NAME
    )

def execute_sql_query(query: str) -> pd.DataFrame:
    print("[DEBUG] Executing Athena query:", query)
    try:
        if not AWS_ATHENA_S3_STAGING_DIR:
            raise ValueError("AWS_ATHENA_S3_STAGING_DIR is not set.")
        connection_string = (
            f"awsathena+rest://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}"
            f"@athena.{AWS_REGION_NAME}.amazonaws.com:443/{ATHENA_SCHEMA_NAME}"
            f"?s3_staging_dir={AWS_ATHENA_S3_STAGING_DIR}&catalog_name=AwsDataCatalog"
        )
        print("[DEBUG] Athena connection string created.")
        engine = create_engine(connection_string)
        df = pd.read_sql_query(query, engine)
        print(f"[DEBUG] Query executed successfully. Rows returned: {len(df)}")
        return df
    except Exception as e:
        print(f"[ERROR] Failed to execute query: {query}, Error: {str(e)}")
        return pd.DataFrame()

# -----------------------------------------------------------------------------------
# Column Normalization
# -----------------------------------------------------------------------------------
import re

def normalize_column_name(col_name: str) -> str:
    # Convert to lowercase and trim whitespace
    normalized = col_name.strip().lower()
    # Replace spaces with underscores
    normalized = normalized.replace(' ', '_')
    # Remove all special characters except underscores
    normalized = re.sub(r'[^a-z0-9_]', '', normalized)
    # Ensure name starts with letter/underscore (valid Python variable)
    if normalized and normalized[0].isdigit():
        normalized = f'_{normalized}'
    return normalized

# -----------------------------------------------------------------------------------
# Multi-Format Date Parser
# -----------------------------------------------------------------------------------
# def parse_dates_with_known_formats(series: pd.Series, possible_formats=None, dayfirst=False) -> pd.Series:
    """
    Try multiple known formats & pick the one that parses the most rows successfully.
    If more than half remain null, fallback to dateutil guess (infer_datetime_format=True).
    Invalid remain as NaT, which we fill with '1970-01-01' afterwards.
    """
    if possible_formats is None:
        possible_formats = [
            "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d/%m/%Y", "%m/%d/%Y",
            "%Y/%m/%d", "%d.%m.%Y", "%Y.%m.%d"
        ]

    best_format = None
    best_valid_count = -1
    total_count = len(series)

    # Try each format
    for fmt in possible_formats:
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        valid_count = parsed.notnull().sum()
        if valid_count > best_valid_count:
            best_valid_count = valid_count
            best_format = fmt

    # Parse with the best format
    parsed_final = pd.to_datetime(series, format=best_format, errors="coerce")
    fallback_nulls = parsed_final.isnull().sum()

    # If more than half remain null, do a final fallback
    if fallback_nulls > (0.5 * total_count):
        parsed_final = pd.to_datetime(series, infer_datetime_format=True, dayfirst=dayfirst, errors="coerce")

    # Fill any leftover nulls with 1970-01-01
    parsed_final = parsed_final.fillna(pd.Timestamp("1970-01-01"))
    return parsed_final

# -----------------------------------------------------------------------------------
# Data Type Inference (int, bigint, double, boolean, timestamp, string)
# -----------------------------------------------------------------------------------
# def infer_column_dtype(series: pd.Series, threshold: float = 0.6) -> str:
    """
    Attempt to infer the most likely data type of a pandas Series.
    This uses 'errors=coerce' for date parsing + threshold-based classification.
    Handles: timestamp, boolean, int, bigint, double, string
    """
    series = series.dropna().astype(str).str.strip()
    if series.empty:
        return "string"

    # Attempt #1: Flexible parse toggling dayfirst
    for dayfirst_val in [False, True]:
        dt_series = pd.to_datetime(series, dayfirst=dayfirst_val, errors='coerce')
        valid_count = dt_series.notnull().sum()
        total_count = len(series)
        if total_count > 0 and (valid_count / total_count) >= threshold:
            return "timestamp"

    # Attempt #2: Known date formats
    known_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d"]
    for fmt in known_formats:
        dt_series = pd.to_datetime(series, format=fmt, errors='coerce')
        valid_count = dt_series.notnull().sum()
        total_count = len(series)
        if total_count > 0 and (valid_count / total_count) >= threshold:
            return "timestamp"

    # Boolean check
    boolean_values = {'true','false','1','0','yes','no','t','f','y','n'}
    unique_values = set(series.str.lower().unique())
    if unique_values.issubset(boolean_values):
        return "boolean"

    # Integer check
    try:
        numeric_series = pd.to_numeric(series, errors='raise')
        # If they are all whole numbers
        if (numeric_series % 1 == 0).all():
            # check if it fits 32-bit
            if numeric_series.min() >= -2147483648 and numeric_series.max() <= 2147483647:
                return "int"
            else:
                return "bigint"
        else:
            # If not all whole numbers, it's probably a float => double
            return "double"
    except ValueError:
        pass

    # Try a float parse anyway for a double
    try:
        pd.to_numeric(series, errors='raise', downcast='float')
        return "double"
    except ValueError:
        pass

    # Default to string
    return "string"

def parse_dates_with_known_formats(series: pd.Series, possible_formats=None, dayfirst=False) -> pd.Series:
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
        parsed = pd.to_datetime(series, format=fmt, errors="coerce")
        valid = parsed.notnull() & (parsed >= pd.Timestamp("1900-01-01")) & (parsed <= pd.Timestamp("2100-12-31"))
        valid_count = valid.sum()
        print(f"[DEBUG] Parsing with {fmt}: Valid {valid_count}/{total_count}")
        valid_masks[fmt] = valid
        parsed_series = parsed_series.where(parsed_series.notnull(), parsed)

    combined_valid = pd.Series(False, index=series.index)
    for mask in valid_masks.values():
        combined_valid |= mask
    nulls = total_count - combined_valid.sum()
    print(f"[DEBUG] Combined parse: Nulls {nulls}/{total_count}")

    if nulls > (0.5 * total_count) and "date" in series.name.lower():
        parsed_flex = pd.to_datetime(series, errors="coerce")
        valid_flex = parsed_flex.notnull() & (parsed_flex >= pd.Timestamp("1900-01-01")) & (parsed_flex <= pd.Timestamp("2100-12-31"))
        valid_count_flex = valid_flex.sum()
        print(f"[DEBUG] Flexible parse fallback: Valid {valid_count_flex}/{total_count}")
        parsed_series = parsed_series.where(parsed_series.notnull(), parsed_flex)

    parsed_final = parsed_series.fillna(pd.Timestamp("1970-01-01"))
    print(f"[DEBUG] Final nulls filled with '1970-01-01': Nulls {parsed_final.isnull().sum()}/{total_count}")
    return parsed_final





def infer_column_dtype(series: pd.Series, threshold: float = 0.6) -> str:
    series = series.dropna()
    if series.empty:
        print(f"[DEBUG] Column is empty after dropna, defaulting to string")
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
            print(f"[DEBUG] Detected as boolean (numeric 0/1)")
            return "boolean"
    except ValueError:
        pass  # Proceed to string-based boolean check if not numeric

    # Check string patterns for exactly 2 unique values
    if len(unique_values) == 2 and unique_values.issubset(boolean_patterns):
        if unique_values.issubset({'true', 'false'}):
            print(f"[DEBUG] Detected as boolean (true/false)")
            return "boolean"
        if unique_values.issubset({'yes', 'no'}):
            print(f"[DEBUG] Detected as boolean (yes/no)")
            return "boolean"
        if unique_values.issubset({'t', 'f'}):
            print(f"[DEBUG] Detected as boolean (t/f)")
            return "boolean"
        if unique_values.issubset({'y', 'n'}):
            print(f"[DEBUG] Detected as boolean (y/n)")
            return "boolean"
        if unique_values.issubset({'1', '0'}):
            print(f"[DEBUG] Detected as boolean (string 0/1)")
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
        print(f"[DEBUG] Format {fmt}: Valid {valid_count}/{total_count} ({valid_ratio:.2f})")
        valid_masks[fmt] = valid
        if valid_ratio >= threshold:
            print(f"[DEBUG] Detected as timestamp with format {fmt}")
            return "timestamp"

    # Combine valid date parses
    combined_valid = pd.Series(False, index=series.index)
    for mask in valid_masks.values():
        combined_valid |= mask
    combined_valid_count = combined_valid.sum()
    combined_valid_ratio = combined_valid_count / total_count
    print(f"[DEBUG] Combined date formats: Valid {combined_valid_count}/{total_count} ({combined_valid_ratio:.2f})")
    if combined_valid_ratio >= threshold and ("date" in col_name or combined_valid_ratio > 0.9):
        print(f"[DEBUG] Detected as timestamp with combined formats")
        return "timestamp"

    # Flexible parse only if column name suggests date
    if "date" in col_name:
        dt_series_flex = pd.to_datetime(series, errors='coerce')
        valid_flex = dt_series_flex.notnull() & (dt_series_flex >= pd.Timestamp("1900-01-01")) & (dt_series_flex <= pd.Timestamp("2100-12-31"))
        valid_count_flex = valid_flex.sum()
        valid_ratio_flex = valid_count_flex / total_count
        print(f"[DEBUG] Flexible parse: Valid {valid_count_flex}/{total_count} ({valid_ratio_flex:.2f})")
        if valid_ratio_flex >= threshold:
            print(f"[DEBUG] Detected as timestamp with flexible parse")
            return "timestamp"

    # Numeric checks (after boolean and date)
    try:
        numeric_series = pd.to_numeric(series, errors='raise')
        if (numeric_series % 1 == 0).all():
            if numeric_series.min() >= -2147483648 and numeric_series.max() <= 2147483647:
                print(f"[DEBUG] Detected as int ({numeric_series.min()} to {numeric_series.max()})")
                return "int"
            else:
                print(f"[DEBUG] Detected as bigint ({numeric_series.min()} to {numeric_series.max()})")
                return "bigint"
        else:
            print(f"[DEBUG] Detected as double ({numeric_series.min()} to {numeric_series.max()})")
            return "double"
    except ValueError:
        print(f"[DEBUG] Not numeric")

    print(f"[DEBUG] Defaulting to string")
    return "string"


# -----------------------------------------------------------------------------------
# Standardize Timestamp Columns
# -----------------------------------------------------------------------------------
def standardize_datetime_columns(df: pd.DataFrame, schema: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    For columns recognized as 'timestamp', we parse them with multi-format logic,
    then convert to a standard ISO datetime string. Invalid => '1970-01-01'.
    """
    for colinfo in schema:
        if colinfo["data_type"] == "timestamp":
            col_name = colinfo["column_name"]
            try:
                print(f"[DEBUG] Parsing dates in column '{col_name}' with multi-format logic...")
                # Use parse_dates_with_known_formats to handle many possible date formats
                # We'll set dayfirst=True if you prefer dd-mm-yyyy, else dayfirst=False
                parsed_dates = parse_dates_with_known_formats(df[col_name], dayfirst=True)
                # Convert final datetimes to string
                df[col_name] = parsed_dates.dt.strftime("%Y-%m-%d %H:%M:%S")

                # Show how many ended up as 1970-01-01
                num_1970 = (df[col_name] == "1970-01-01 00:00:00").sum()
                print(f"[DEBUG] Done parsing '{col_name}'. Rows with 1970-01-01 = {num_1970}")
            except Exception as e:
                print(f"[WARNING] Could not standardize date column '{col_name}': {str(e)}")
    return df

# -----------------------------------------------------------------------------------
# Suggest columns
# -----------------------------------------------------------------------------------
def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
    return df.columns[-1] if len(df.columns) else None

def suggest_entity_id_column(df: pd.DataFrame) -> Any:
    likely_id_columns = [col for col in df.columns if "id" in col.lower()]
    for col in likely_id_columns:
        if df[col].nunique() / len(df) > 0.95:
            return col
    for col in df.columns:
        if df[col].nunique() / len(df) > 0.95:
            return col
    return None

# -----------------------------------------------------------------------------------
# Update PredictiveSettings
# -----------------------------------------------------------------------------------
from django.db import transaction
import difflib
import logging
logger = logging.getLogger(__name__)

# def update_predictive_settings(ps, parsed_updates, schema_columns):
#     """
#     Update only fields explicitly mentioned in parsed_updates, with improved validation.
#     Now includes machine_learning_type with strict validation.
#     """
#     logger.info(f"[update_predictive_settings] Current PS values: {ps.__dict__}")
#     logger.info(f"[update_predictive_settings] Parsed updates: {parsed_updates}")

#     updateable_fields = [
#         'target_column',
#         'entity_column',
#         'time_column',
#         'predictive_question',
#         'time_frame',
#         'time_frequency',
#         'machine_learning_type'  # Already added in previous example, kept here
#     ]

#     def validate_and_match(proposed_value, current_value):
#         """Case-insensitive exact + fuzzy matching for columns."""
#         if not proposed_value:
#             return current_value
#         for col in schema_columns:
#             if col.lower() == proposed_value.lower():
#                 return col
#         matches = difflib.get_close_matches(proposed_value, schema_columns, n=1, cutoff=0.6)
#         if matches:
#             return matches[0]
#         logger.warning(f"No column match found for '{proposed_value}'. Keeping old value '{current_value}'.")
#         return current_value

#     updated_fields = {}

#     try:
#         with transaction.atomic():
#             for field in updateable_fields:
#                 if field in parsed_updates:
#                     new_value = parsed_updates[field]
#                     current_value = getattr(ps, field)

#                     if field in ['target_column', 'entity_column', 'time_column']:
#                         validated = validate_and_match(new_value, current_value)
#                         if validated != current_value:
#                             setattr(ps, field, validated)
#                             updated_fields[field] = validated
#                     elif field == 'machine_learning_type':
#                         # Strict validation: only accept valid ML types
#                         if new_value in ["classification", "regression"]:
#                             if new_value != current_value:
#                                 setattr(ps, field, new_value)
#                                 updated_fields[field] = new_value
#                         elif new_value is not None:  # Allow None, but log invalid values
#                             logger.warning(f"Invalid machine_learning_type '{new_value}', keeping '{current_value}'")
#                     else:
#                         if new_value != current_value:  # Allow None for other fields
#                             setattr(ps, field, new_value)
#                             updated_fields[field] = new_value

#             # Auto-generate predictive question if needed
#             if ('target_column' in updated_fields or 'time_frequency' in updated_fields 
#                 or 'time_frame' in updated_fields or 'entity_column' in updated_fields):
#                 if not parsed_updates.get('predictive_question'):
#                     freq = ps.time_frequency if ps.time_frequency else "weekly"
#                     frame = ps.time_frame if ps.time_frame else "30 days"
#                     target = ps.target_column if ps.target_column else "target"
#                     entity = ps.entity_column if ps.entity_column else "each record"
#                     ps.predictive_question = f"Predict {freq}, the {target} for {entity} in the next {frame}"
#                     updated_fields['predictive_question'] = ps.predictive_question

#             if updated_fields:
#                 ps.save(update_fields=list(updated_fields.keys()))

#         logger.info(f"[update_predictive_settings] Updated fields: {updated_fields}")
#     except Exception as e:
#         logger.error(f"[update_predictive_settings] Error occurred: {e}", exc_info=True)

#     return updated_fields


def update_predictive_settings(ps, parsed_updates, schema_columns):
    """
    Update only fields explicitly mentioned in parsed_updates, with improved validation.
    Now includes machine_learning_type and features with strict validation.
    """
    logger.info(f"[update_predictive_settings] Current PS values: {ps.__dict__}")
    logger.info(f"[update_predictive_settings] Parsed updates: {parsed_updates}")

    updateable_fields = [
        'target_column',
        'entity_column',
        'time_column',
        'predictive_question',
        'time_frame',
        'time_frequency',
        'machine_learning_type',
        'features'  # NEW: Add features to updateable fields
    ]

    def validate_and_match(proposed_value, current_value):
        """Case-insensitive exact + fuzzy matching for columns."""
        if not proposed_value:
            return current_value
        for col in schema_columns:
            if col.lower() == proposed_value.lower():
                return col
        matches = difflib.get_close_matches(proposed_value, schema_columns, n=1, cutoff=0.6)
        if matches:
            return matches[0]
        logger.warning(f"No column match found for '{proposed_value}'. Keeping old value '{current_value}'.")
        return current_value

    updated_fields = {}

    try:
        with transaction.atomic():
            for field in updateable_fields:
                if field in parsed_updates:
                    new_value = parsed_updates[field]
                    current_value = getattr(ps, field)

                    if field in ['target_column', 'entity_column', 'time_column']:
                        validated = validate_and_match(new_value, current_value)
                        if validated != current_value:
                            setattr(ps, field, validated)
                            updated_fields[field] = validated
                    elif field == 'machine_learning_type':
                        if new_value in ["classification", "regression"]:
                            if new_value != current_value:
                                setattr(ps, field, new_value)
                                updated_fields[field] = new_value
                        elif new_value is not None:
                            logger.warning(f"Invalid machine_learning_type '{new_value}', keeping '{current_value}'")
                    elif field == 'features':  # NEW: Handle explicit feature updates
                        if isinstance(new_value, list) and all(col in schema_columns for col in new_value):
                            if new_value != current_value:
                                setattr(ps, field, new_value)
                                updated_fields[field] = new_value
                        else:
                            logger.warning(f"Invalid features list '{new_value}', keeping '{current_value}'")
                    else:
                        if new_value != current_value:
                            setattr(ps, field, new_value)
                            updated_fields[field] = new_value

            # Auto-generate predictive question if needed
            if ('target_column' in updated_fields or 'time_frequency' in updated_fields 
                or 'time_frame' in updated_fields or 'entity_column' in updated_fields):
                if not parsed_updates.get('predictive_question'):
                    freq = ps.time_frequency if ps.time_frequency else "weekly"
                    frame = ps.time_frame if ps.time_frame else "30 days"
                    target = ps.target_column if ps.target_column else "target"
                    entity = ps.entity_column if ps.entity_column else "each record"
                    ps.predictive_question = f"Predict {freq}, the {target} for {entity} in the next {frame}"
                    updated_fields['predictive_question'] = ps.predictive_question

            # NEW: Auto-generate feature list if not explicitly provided
            if 'features' not in parsed_updates and schema_columns:
                exclude_cols = {ps.target_column, ps.entity_column}
                feature_list = [col for col in schema_columns if col not in exclude_cols and col is not None]
                if feature_list != ps.features:
                    ps.features = feature_list
                    updated_fields['features'] = feature_list
                    logger.info(f"[update_predictive_settings] Auto-generated features: {feature_list}")

            if updated_fields:
                ps.save(update_fields=list(updated_fields.keys()))

        logger.info(f"[update_predictive_settings] Updated fields: {updated_fields}")
    except Exception as e:
        logger.error(f"[update_predictive_settings] Error occurred: {e}", exc_info=True)

    return updated_fields

# -----------------------------------------------------------------------------------
# UnifiedChatGPTAPI
# -----------------------------------------------------------------------------------
class UnifiedChatGPTAPI(APIView):
    parser_classes = [MultiPartParser, FormParser, JSONParser]

    def post(self, request):
        action = request.data.get('action', '')
        if action == 'reset':
            return self.reset_conversation(request)
        if action == 'generate_notebook':
            return self.generate_notebook(request)
        if "file" in request.FILES:
            return self.handle_file_upload(request, request.FILES.getlist("file"))
        return self.handle_chat(request)

    def handle_file_upload(self, request, files):
        """
        1) Ingests CSV/XLSX, normalizes columns, and infers schema.
        2) Uploads to S3 and updates Glue.
        3) Persists file metadata (schema, suggestions, etc.) along with the chat ID.
        """
        user_id = request.data.get("user_id", "default_user")
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        # Retrieve or create the chat based on the provided chat_id.
        chat_id = request.data.get("chat_id", "")
        if chat_id:
            try:
                chat_obj = ChatBackup.objects.get(user=user, chat_id=chat_id)
            except ChatBackup.DoesNotExist:
                # If the chat_id is invalid, create a new chat.
                chat_obj, chat_id = self.create_new_chat(user, "Initial file upload")
        else:
            chat_obj, chat_id = self.create_new_chat(user, "Initial file upload")

        s3 = get_s3_client()
        glue = get_glue_client()
        uploaded_files_info = []

        for file in files:
            print(f"[DEBUG] Processing file: {file.name}")  # DEBUG PRINT

            try:
                # READ CSV or Excel with on_bad_lines='error' => no row skipping
                if file.name.lower().endswith('.csv'):
                    df = pd.read_csv(
                        file,
                        low_memory=False,
                        encoding='utf-8',
                        delimiter=',',
                        na_values=['NA', 'N/A', ''],
                        on_bad_lines='error'  # CRITICAL to avoid skipping
                    )
                else:
                    df = pd.read_excel(file, engine='openpyxl')

                print(f"[DEBUG] Shape after reading file: {df.shape}")
                print("[DEBUG] Head of DataFrame:\n", df.head(5))

                if df.empty:
                    print("[ERROR] File is empty:", file.name)
                    return Response(
                        {"error": f"Uploaded file {file.name} is empty."},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                if not df.columns.any():
                    print("[ERROR] File has no columns:", file.name)
                    return Response(
                        {"error": f"Uploaded file {file.name} has no columns."},
                        status=status.HTTP_400_BAD_REQUEST
                    )

            except pd.errors.ParserError as e:
                print("[ERROR] CSV parsing error:", e)
                return Response(
                    {"error": f"CSV parsing error for file {file.name}: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            except Exception as e:
                print("[ERROR] Error reading file:", e)
                return Response(
                    {"error": f"Error reading file {file.name}: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Normalize column names
            old_cols = df.columns.tolist()
            normalized_columns = [normalize_column_name(c) for c in df.columns]
            if len(normalized_columns) != len(set(normalized_columns)):
                print("[ERROR] Duplicate columns after normalization.")
                return Response({"error": "Duplicate columns detected after normalization."},
                                status=status.HTTP_400_BAD_REQUEST)
            if any(col == '' for col in normalized_columns):
                print("[ERROR] Empty column names after normalization.")
                return Response({"error": "Some columns have empty names after normalization."},
                                status=status.HTTP_400_BAD_REQUEST)
            df.columns = normalized_columns

            print("[DEBUG] Old columns -> Normalized columns:")
            for oc, nc in zip(old_cols, normalized_columns):
                print(f"   {oc} -> {nc}")

            # Infer schema
            raw_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]
            print("[DEBUG] Raw schema inferred:", raw_schema)

            # Standardize date columns
            rows_before_std = df.shape[0]
            df = standardize_datetime_columns(df, raw_schema)
            rows_after_std = df.shape[0]
            print(f"[DEBUG] Rows before date standardization: {rows_before_std}, after: {rows_after_std}")

            # Re-check final schema
            final_schema = []
            for col in df.columns:
                final_schema.append({
                    "column_name": col,
                    "data_type": infer_column_dtype(df[col])
                })
            print("[DEBUG] Final schema after standardizing date columns:", final_schema)

            has_date_column = any(c["data_type"] == "timestamp" for c in final_schema)
            possible_date_cols = [c["column_name"] for c in final_schema if c["data_type"] == "timestamp"]

            # Fix boolean columns
            boolean_columns = [c['column_name'] for c in final_schema if c['data_type'] == 'boolean']
            replacement_dict = {
                '1': 'true', '0': 'false',
                'yes': 'true', 'no': 'false',
                't': 'true', 'f': 'false',
                'y': 'true', 'n': 'false',
                'true': 'true', 'false': 'false',
            }
            for col_name in boolean_columns:
                df[col_name] = (
                    df[col_name].astype(str)
                    .str.strip()
                    .str.lower()
                    .replace(replacement_dict)
                )
                unexpected_values = [v for v in df[col_name].unique() if v not in ['true', 'false']]
                if unexpected_values:
                    print("[ERROR] Unexpected boolean values:", unexpected_values)
                    return Response(
                        {"error": f"Unexpected boolean values in column {col_name}: {unexpected_values}"},
                        status=status.HTTP_400_BAD_REQUEST
                    )

            # Build unique file key for S3
            file_name_base, file_extension = os.path.splitext(file.name)
            file_name_base = file_name_base.lower().replace(' ', '_')
            unique_id = uuid.uuid4().hex[:8]
            new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
            s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
            file_key = f"uploads/{unique_id}/{s3_file_name}"
            print("[DEBUG] Uploading file to S3 at key:", file_key)

            try:
                with transaction.atomic():
                    # Save the file record in Django
                    file.seek(0)
                    file_serializer = UploadedFileSerializer(data={'name': new_file_name, 'file': file})
                    if file_serializer.is_valid():
                        file_instance = file_serializer.save()

                        # Convert DF to CSV in memory
                        csv_buffer = BytesIO()
                        df.to_csv(csv_buffer, index=False, encoding='utf-8')
                        csv_buffer.seek(0)

                        # Upload to S3
                        s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
                        print("[DEBUG] S3 upload successful:", file_key)
                        s3.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=file_key)

                        file_url = f"s3://{AWS_STORAGE_BUCKET_NAME}/{file_key}"
                        file_instance.file_url = file_url
                        file_instance.save()

                        # Store the schema in DB
                        FileSchema.objects.create(file=file_instance, schema=final_schema)

                        file_size_mb = file.size / (1024 * 1024)
                        self.trigger_glue_update(new_file_name, final_schema, file_key, file_size_mb)

                        # Build suggestions
                        target_suggestion = suggest_target_column(df, [])
                        entity_suggestion = suggest_entity_id_column(df)
                        feature_candidates = [c for c in df.columns if c not in [entity_suggestion, target_suggestion]]

                        # Optionally refine with GPT columns (skipped here)

                        file_info = {
                            'id': file_instance.id,
                            'name': file_instance.name,
                            'file_url': file_instance.file_url,
                            'schema': final_schema,
                            'file_size_mb': file_size_mb,
                            'has_date_column': has_date_column,
                            'date_columns': possible_date_cols,
                            'suggestions': {
                                'target_column': target_suggestion,
                                'entity_column': entity_suggestion,
                                'feature_columns': feature_candidates
                            }
                        }
                        uploaded_files_info.append(file_info)

                        # Persist ChatFileInfo
                        ChatFileInfo.objects.create(
                            user=user,
                            chat=chat_obj,
                            file=file_instance,
                            name=file_instance.name,
                            file_url=file_instance.file_url,
                            schema=final_schema,
                            suggestions=file_info['suggestions'],
                            has_date_column=has_date_column,
                            date_columns=possible_date_cols,
                        )
                    else:
                        print("[ERROR] File serializer errors:", file_serializer.errors)
                        return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            except ClientError as e:
                print("[ERROR] AWS ClientError:", e)
                return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                print("[ERROR] Unexpected error during file processing:", e)
                return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        # (Optionally, initialize conversation chain if needed)
        memory_key = f"{user_id}_{chat_id}"
        if memory_key not in user_conversations:
            from langchain.chains import ConversationChain
            from langchain.memory import ConversationBufferMemory
            conversation_chain = ConversationChain(
                llm=llm_chatgpt,
                prompt=chat_prompt,
                input_key="user_input",
                memory=ConversationBufferMemory()
            )
            user_conversations[memory_key] = conversation_chain
        else:
            conversation_chain = user_conversations[memory_key]

        if uploaded_files_info:
            schema_discussion = self.format_schema_message(uploaded_files_info[0])
            conversation_chain.memory.chat_memory.messages.append(AIMessage(content=schema_discussion))
        else:
            schema_discussion = "No file info available."

        print("[DEBUG] Files uploaded and schema discussion initiated.")
        return Response({
            "message": "Files uploaded and processed successfully.",
            "uploaded_files": uploaded_files_info,
            "chat_message": schema_discussion,
            "chat_id": chat_id
        }, status=status.HTTP_201_CREATED)

    def create_new_chat(self, user, initial_text):
        chat_id = str(uuid.uuid4())
        chat_title = initial_text[:50]
        chat_obj = ChatBackup.objects.create(
            user=user,
            chat_id=chat_id,
            title=chat_title,
            messages=[]
        )
        memory_key = f"{user.id}_{chat_id}"
        conversation_chain = ConversationChain(
            llm=llm_chatgpt,
            prompt=chat_prompt,
            input_key="user_input",
            memory=ConversationBufferMemory(return_messages=True)
        )
        user_conversations[memory_key] = conversation_chain
        return chat_obj, chat_id
    

    def handle_chat(self, request):
        user_input = request.data.get("message", "").strip()
        user_id = request.data.get("user_id", "default_user")
        chat_id = request.data.get("chat_id")
        new_chat_flag = request.data.get("new_chat", False)

        if not user_input:
            return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        # Create or get the chat
        if new_chat_flag or not chat_id:
            chat_obj, chat_id = self.create_new_chat(user, user_input)
        else:
            chat_obj, _ = ChatBackup.objects.get_or_create(
                user=user,
                chat_id=chat_id,
                defaults={"title": user_input[:50], "messages": []}
            )
            if not chat_obj.messages:
                memory_key = f"{user_id}_{chat_id}"
                conversation_chain = ConversationChain(
                    llm=llm_chatgpt,
                    prompt=chat_prompt,
                    input_key="user_input",
                    memory=ConversationBufferMemory(return_messages=True)
                )
                user_conversations[memory_key] = conversation_chain

        # Get schema and file info
        from .models import ChatFileInfo
        uploaded_file_info = ChatFileInfo.objects.filter(user=user, chat=chat_obj).order_by('-created_at').first()
        schema_columns = []
        if uploaded_file_info:
            schema_columns = [col["column_name"] for col in uploaded_file_info.schema]
        else:
            logger.warning("No uploaded file info found for this chat; schema_columns will be empty.")

        # Parse user input
        parsed_result = parse_nlu_input(
            system_prompt="Extract user instructions about target/entity/time_frame/time_column/time_frequency/predictive_question.",
            user_message=user_input,
            schema_columns=schema_columns
        )
        is_confirmation = parsed_result["is_confirmation"]
        explicit_updates = parsed_result["updates"]

        # Get or create PredictiveSettings with no default machine_learning_type
        ps, created = PredictiveSettings.objects.get_or_create(
            user=user,
            chat_id=chat_id,
            defaults={
                'target_column': None,
                'entity_column': None,
                'time_column': None,
                'predictive_question': None,
                'time_frame': None,
                'time_frequency': None,
                'machine_learning_type': None  # CHANGED: No default, starts as None
            }
        )

        # Prepare updates and determine machine_learning_type
        parsed_updates = {}
        if explicit_updates:
            parsed_updates = explicit_updates
            logger.info(f"[handle_chat] Applying explicit updates: {parsed_updates}")
        elif is_confirmation:
            # Fallback suggestions
            if not ps.entity_column and uploaded_file_info:
                suggestions = uploaded_file_info.suggestions or {}
                suggested_entity = suggestions.get("entity_column")
                if suggested_entity:
                    parsed_updates["entity_column"] = suggested_entity
                    logger.info(f"[handle_chat] Setting entity_column to '{suggested_entity}' by fallback suggestion.")
            if not ps.target_column and uploaded_file_info:
                suggestions = uploaded_file_info.suggestions or {}
                suggested_target = suggestions.get("target_column")
                if suggested_target:
                    parsed_updates["target_column"] = suggested_target
                    logger.info(f"[handle_chat] Setting target_column to '{suggested_target}' by fallback suggestion.")

        # --- NEW: Determine machine_learning_type with high accuracy ---
        predictive_question = parsed_updates.get('predictive_question') or ps.predictive_question
        target_column = parsed_updates.get('target_column') or ps.target_column

        if predictive_question or target_column:
            ml_type_from_question = None
            ml_type_from_data = None

            # Step 1: Classify based on predictive question if available
            if predictive_question:
                ml_type_from_question = classify_ml_type(predictive_question)
                logger.info(f"[handle_chat] Classified ML type from question '{predictive_question}': {ml_type_from_question}")

            # Step 2: Infer from target column data if available
            if target_column and uploaded_file_info:
                schema_dict = {col["column_name"]: col["data_type"] for col in uploaded_file_info.schema}
                target_dtype = schema_dict.get(target_column)
                if target_dtype:
                    if target_dtype in ["int", "bigint", "double"]:
                        ml_type_from_data = "regression"
                    elif target_dtype in ["string", "boolean"]:
                        ml_type_from_data = "classification"
                    else:
                        ml_type_from_data = None  # Unknown or unsupported type
                    logger.info(f"[handle_chat] Inferred ML type from target column '{target_column}' ({target_dtype}): {ml_type_from_data}")

            # Step 3: Validate and reconcile ML type
            final_ml_type = None
            if ml_type_from_question and ml_type_from_data:
                if ml_type_from_question == ml_type_from_data:
                    final_ml_type = ml_type_from_question
                    logger.info(f"[handle_chat] ML type confirmed: '{final_ml_type}' (question and data agree)")
                else:
                    logger.warning(f"[handle_chat] Conflict: Question suggests '{ml_type_from_question}', data suggests '{ml_type_from_data}'. Prioritizing data.")
                    final_ml_type = ml_type_from_data  # Data-driven type takes precedence for accuracy
            elif ml_type_from_question and not ml_type_from_data:
                final_ml_type = ml_type_from_question if ml_type_from_question != "unknown" else None
                logger.info(f"[handle_chat] ML type set to '{final_ml_type}' from question (no data validation possible)")
            elif ml_type_from_data and not ml_type_from_question:
                final_ml_type = ml_type_from_data
                logger.info(f"[handle_chat] ML type set to '{final_ml_type}' from data (no question provided)")
            else:
                logger.info("[handle_chat] Insufficient info to determine ML type, keeping as None")

            # Step 4: Only set if we have a valid, confirmed type
            if final_ml_type in ["classification", "regression"]:
                parsed_updates["machine_learning_type"] = final_ml_type
            else:
                parsed_updates["machine_learning_type"] = None  # Remains None if unclear

        # --- END NEW SECTION ---

        logger.info(f"[handle_chat] Before update => {ps.__dict__}")
        logger.info(f"[handle_chat] parsed_updates => {parsed_updates}")

        updated_fields = update_predictive_settings(ps, parsed_updates, schema_columns)
        ps.refresh_from_db()

        logger.info(f"[handle_chat] After update => {ps.__dict__}")
        logger.info(f"[handle_chat] Updated fields => {updated_fields}")

        # Rest of conversation handling
        memory_key = f"{user_id}_{chat_id}"
        if memory_key not in user_conversations:
            restored_memory = ConversationBufferMemory(return_messages=True)
            for msg in chat_obj.messages:
                if msg["sender"] == "assistant":
                    restored_memory.chat_memory.add_message(AIMessage(content=msg["text"]))
                else:
                    restored_memory.chat_memory.add_message(HumanMessage(content=msg["text"]))
            conversation_chain = ConversationChain(
                llm=llm_chatgpt,
                prompt=chat_prompt,
                input_key="user_input",
                memory=restored_memory
            )
            user_conversations[memory_key] = conversation_chain
        else:
            conversation_chain = user_conversations[memory_key]

        assistant_response = conversation_chain.run(user_input=user_input)

        timestamp = datetime.datetime.now().isoformat()
        chat_obj.messages.append({"sender": "user", "text": user_input, "timestamp": timestamp})
        chat_obj.messages.append({"sender": "assistant", "text": assistant_response, "timestamp": timestamp})
        chat_obj.save()

        show_generate_notebook = bool(ps.target_column and ps.entity_column)

        return Response({
            "response": assistant_response,
            "chat_id": chat_id,
            "show_generate_notebook": show_generate_notebook,
            "corrected_target_column": ps.target_column,
            "corrected_entity_column": ps.entity_column,
            "corrected_time_column": ps.time_column,
            "settings_updated": bool(updated_fields),
            "predictive_question": ps.predictive_question,
            "machine_learning_type": ps.machine_learning_type  # Include in response for visibility
        })




    def reset_conversation(self, request):
        user_id = request.data.get("user_id", "default_user")
        keys_to_delete = [key for key in user_conversations if key == str(user_id) or key.startswith(f"{user_id}_")]
        for key in keys_to_delete:
            del user_conversations[key]
        if user_id in user_schemas:
            del user_schemas[user_id]
        if user_id in user_notebooks:
            del user_notebooks[user_id]

        PredictiveSettings.objects.filter(user_id=user_id).delete()
        ChatBackup.objects.filter(user_id=user_id).delete()

        return Response({"message": "Conversation reset successful."})

    def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
        schema = uploaded_file['schema']
        target_column = uploaded_file['suggestions'].get('target_column')
        entity_id_column = uploaded_file['suggestions'].get('entity_id_column',
                            uploaded_file['suggestions'].get('entity_column'))
        feature_columns = uploaded_file['suggestions'].get('feature_columns', [])
        has_date_column = uploaded_file.get('has_date_column', False)
        date_cols = uploaded_file.get('date_columns', [])

        schema_text = (
            f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
            "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
            "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
            f"Suggested Target Column: {target_column or 'None'}\n"
            f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
            f"Suggested Feature Columns: {', '.join(feature_columns) if feature_columns else 'None'}\n\n"
        )
        if has_date_column:
            if len(date_cols) == 1:
                schema_text += (
                    f"We detected a single date column: {date_cols[0]}.\n"
                    "We'll use it for time-based modeling unless you specify otherwise.\n"
                    "You can also specify a 'Time Frame: <X>' (e.g., 'Time Frame: 1 WEEK') or 'Time Frequency: <daily|weekly|monthly>'.\n\n"
                )
            elif len(date_cols) > 1:
                schema_text += (
                    "We detected multiple date columns. Please specify which one to use as the time column:\n"
                    f"{date_cols}\n\n"
                    "And also specify 'Time Frame: <X>' or 'Time Frequency: <daily|weekly|monthly>' if youâ€™d like a time-based approach.\n"
                )
        else:
            schema_text += (
                "No date column detected, so by default we'll proceed with a non-time-based approach.\n\n"
            )

        schema_text += (
            "Please confirm:\n"
            "- Is the Target Column correct?\n"
            "- Is the Entity ID Column correct?\n"
            "- If a date column is detected, specify 'Time Column: <column>' if you want a time-based approach.\n"
            "- Optionally specify 'Time Frame: <X>' (e.g., 'Time Frame: 2 WEEKS') and 'Time Frequency: <daily|weekly|monthly>'.\n\n"
            "(Reply 'yes' to confirm or provide corrections in the format:\n"
            "'Entity ID Column: <column>, Target Column: <column>, Time Column: <column>, Time Frame: <X>, Time Frequency: <Y>')"
        )
        return schema_text

    def generate_notebook(self, request):
        user_id = request.data.get("user_id")
        chat_id = request.data.get("chat_id")

        if not user_id or not chat_id:
            return Response({"error": "user_id and chat_id are required."},
                            status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

        try:
            chat_obj = ChatBackup.objects.get(user=user, chat_id=chat_id)
        except ChatBackup.DoesNotExist:
            return Response({"error": "Chat not found."}, status=status.HTTP_404_NOT_FOUND)

        try:
            settings_obj = PredictiveSettings.objects.get(user=user, chat_id=chat_id)
        except PredictiveSettings.DoesNotExist:
            return Response({"error": "No predictive settings found. Please confirm columns first."},
                            status=status.HTTP_400_BAD_REQUEST)

        entity_id_column = settings_obj.entity_column
        target_column = settings_obj.target_column
        time_column = settings_obj.time_column
        time_frame = settings_obj.time_frame
        time_frequency = settings_obj.time_frequency

        from .models import ChatFileInfo
        uploaded_file_info = ChatFileInfo.objects.filter(user=user, chat=chat_obj).order_by('-created_at').first()
        if not uploaded_file_info:
            return Response({"error": "Uploaded file info not found."},
                            status=status.HTTP_400_BAD_REQUEST)

        feature_columns = []
        if uploaded_file_info.suggestions:
            #feature_columns = uploaded_file_info.suggestions.get("feature_columns", [])
            feature_columns = settings_obj.features or []

        columns_list = [col['column_name'] for col in uploaded_file_info.schema]
        if entity_id_column and entity_id_column not in columns_list:
            return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."},
                            status=status.HTTP_400_BAD_REQUEST)
        if target_column and target_column not in columns_list:
            return Response({"error": f"Target column '{target_column}' does not exist."},
                            status=status.HTTP_400_BAD_REQUEST)

        has_date_column = uploaded_file_info.has_date_column
        file_url = uploaded_file_info.file_url
        table_name_raw = os.path.splitext(uploaded_file_info.name)[0]
        sanitized_table_name = self.sanitize_identifier(table_name_raw)


        # Determine prediction_type based on notebook type
        prediction_type = True if has_date_column and time_column else False
        # Update PredictiveSettings with prediction_type
        settings_obj.prediction_type = prediction_type
        settings_obj.save()

        if has_date_column and time_column:
            final_time_frame = time_frame if time_frame else "1 WEEK"
            final_notebook, new_target_column = self.create_dynamic_time_based_notebook(
                entity_id_column=entity_id_column,
                time_column=time_column,
                target_column=target_column,
                table_name=sanitized_table_name,
                time_horizon=final_time_frame,
                extra_features=feature_columns,
                time_frequency=time_frequency
            )

             # Update PredictiveSettings with the new target column
            settings_obj.new_target_column = new_target_column
            settings_obj.save()

            notebook_sanitized = self.sanitize_notebook(final_notebook)
            notebook_json = nbformat.writes(notebook_sanitized, version=4)

            notebook = Notebook.objects.create(
                user=user,
                chat=chat_id,
                entity_column=entity_id_column,
                target_column=target_column,
                # target_column=new_target_column,  # Use the new target column
                time_column=time_column,
                time_frame=final_time_frame,
                time_frequency=time_frequency,
                features=feature_columns,
                file_url=file_url,
                notebook_json=notebook_json
            )

            user_notebooks[user_id] = {'time_based_notebook': notebook_json}
            return Response({
                "message": "Notebook generated and saved successfully.",
                "notebook_id": notebook.id,
                "notebook_data": notebook_json,
                "prediction_type": prediction_type
            }, status=status.HTTP_200_OK)
        else:
            notebook_non_time_based = self.create_non_time_based_notebook(
                entity_id_column,
                target_column,
                feature_columns,
                sanitized_table_name,
                columns_list
            )
            notebook_non_time_based_sanitized = self.sanitize_notebook(notebook_non_time_based)
            notebook_non_time_based_json = nbformat.writes(notebook_non_time_based_sanitized, version=4)
            notebook_record = Notebook.objects.create(
                user=user,
                chat=chat_id,
                entity_column=entity_id_column,
                target_column=target_column,
                features=feature_columns,
                file_url=file_url,
                notebook_json=notebook_non_time_based_json
            )

            user_notebooks[user_id] = {'non_time_based_notebook': notebook_non_time_based_json}

            return Response({
                "message": "Notebook generated and saved successfully (non-time-based).",
                "non_time_based_notebook_id": notebook_record.id,
                "non_time_based_notebook": notebook_non_time_based_json,
                "prediction_type": prediction_type
            }, status=status.HTTP_200_OK)

    def create_dynamic_time_based_notebook(
        self,
        entity_id_column: str,
        time_column: str,
        target_column: str,
        table_name: str,
        time_horizon: str,
        extra_features: list,
        time_frequency: str = None
):
        import nbformat
        import datetime
        import numpy as np
        import pandas as pd
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

        def parse_time_horizon(th: str):
            parts = th.strip().split()
            if len(parts) < 2:
                return 1, "week"
            number = parts[0]
            unit = parts[1].lower().rstrip('s')
            return int(number), unit

        def _stringify_timestamps(rows):
            for row in rows:
                for k, v in row.items():
                    if isinstance(v, (pd.Timestamp, datetime.datetime)):
                        row[k] = v.strftime("%Y-%m-%d %H:%M:%S")

        horizon_number, horizon_unit = parse_time_horizon(time_horizon)

        if time_frequency:
            freq_lower = time_frequency.strip().lower()
            if freq_lower.startswith("day"):
                increment_unit = "day"
            elif freq_lower.startswith("week"):
                increment_unit = "week"
            else:
                increment_unit = "month"
        else:
            increment_unit = "month"

        nb = new_notebook()
        cells = []

        step1_markdown = new_markdown_cell("### Step 1: Determine relevant timestamps based on Time Frequency")
        cells.append(step1_markdown)

        query_step1 = f"""
            WITH minmax AS (
                SELECT
                    MIN({time_column}) AS min_ts,
                    MAX({time_column}) AS max_ts
                FROM {table_name}
            ),
            all_relevant_dates AS (
                SELECT
                    date_trunc('{increment_unit}', day_in_range) AS relevant_time
                FROM minmax
                CROSS JOIN UNNEST(
                    SEQUENCE(min_ts, max_ts, INTERVAL '1' DAY)
                ) AS t(day_in_range)
            )
            SELECT DISTINCT relevant_time
            FROM all_relevant_dates
            ORDER BY relevant_time
        """.strip()

        from nbformat.v4 import new_markdown_cell, new_code_cell, new_output
        step1_cell = new_code_cell(query_step1)
        df_step1 = execute_sql_query(query_step1)
        step1_cell['execution_count'] = 1

        if df_step1.empty:
            step1_cell.outputs = [
                new_output(
                    output_type='execute_result',
                    data={'text/plain': f"No data returned for query:\n{query_step1}"},
                    execution_count=1
                )
            ]
        else:
            df_step1 = df_step1.replace([None, float('inf'), float('-inf')], None)
            result_json = df_step1.to_dict(orient='records')
            _stringify_timestamps(result_json)

            columns = []
            for col in df_step1.columns:
                guessed_type = infer_column_dtype(df_step1[col])
                columns.append({"name": col, "type": guessed_type})

            text_repr = df_step1.head().to_string(index=False)
            step1_cell.outputs = [
                new_output(
                    output_type='execute_result',
                    data={
                        'application/json': {'rows': result_json, 'columns': columns},
                        'text/plain': text_repr
                    },
                    execution_count=1
                )
            ]
        cells.append(step1_cell)

        step2_markdown = new_markdown_cell("### Step 2: For each entity, gather relevant times after earliest record")
        cells.append(step2_markdown)

        step1_sub = query_step1.strip().rstrip(';')
        query_step2 = f"""
            WITH entity_earliest_time AS (
                SELECT
                    {entity_id_column} AS {entity_id_column},
                    MIN({time_column}) AS first_seen_time
                FROM {table_name}
                GROUP BY {entity_id_column}
            ),
            relevant_times_in_dataset AS (
                {step1_sub}
            )
            SELECT
                entity_earliest_time.{entity_id_column},
                relevant_times_in_dataset.relevant_time AS analysis_time
            FROM entity_earliest_time
            JOIN relevant_times_in_dataset
                ON relevant_times_in_dataset.relevant_time >= entity_earliest_time.first_seen_time
            ORDER BY analysis_time, {entity_id_column}
        """.strip()

        step2_cell = new_code_cell(query_step2)
        df_step2 = execute_sql_query(query_step2)
        step2_cell['execution_count'] = 1

        if df_step2.empty:
            step2_cell.outputs = [
                new_output(
                    output_type='execute_result',
                    data={'text/plain': f"No data returned for query:\n{query_step2}"},
                    execution_count=1
                )
            ]
        else:
            df_step2 = df_step2.replace([None, float('inf'), float('-inf')], None)
            result_json = df_step2.to_dict(orient='records')
            _stringify_timestamps(result_json)

            columns = []
            for col in df_step2.columns:
                guessed_type = infer_column_dtype(df_step2[col])
                columns.append({"name": col, "type": guessed_type})

            text_repr = df_step2.head().to_string(index=False)
            step2_cell.outputs = [
                new_output(
                    output_type='execute_result',
                    data={'application/json': {'rows': result_json, 'columns': columns},
                        'text/plain': text_repr},
                    execution_count=1
                )
            ]
        cells.append(step2_cell)

        step3_markdown = new_markdown_cell("### Step 3: Summarize the target measure over the chosen time horizon")
        cells.append(step3_markdown)

        step2_sub = query_step2.strip().rstrip(';')
        horizon_label = time_horizon.replace(' ', '_')
        new_target_column = f"target_within_{horizon_label}_after"  # Extract the new target column name
        query_step3 = f"""
            WITH last_time AS (
                SELECT MAX({time_column}) AS max_ts
                FROM {table_name}
            ),
            entity_times AS (
                {step2_sub}
            )
            SELECT
                entity_times.{entity_id_column},
                entity_times.analysis_time,
                COALESCE(SUM(tbl.{target_column}), 0) AS {new_target_column}
            FROM entity_times
            LEFT JOIN {table_name} AS tbl
                ON tbl.{entity_id_column} = entity_times.{entity_id_column}
                AND tbl.{time_column} >= entity_times.analysis_time
                AND tbl.{time_column} < date_add('{horizon_unit}', {horizon_number}, entity_times.analysis_time)
            WHERE entity_times.analysis_time <= date_add('{horizon_unit}', -{horizon_number}, (SELECT max_ts FROM last_time))
            GROUP BY
                entity_times.{entity_id_column},
                entity_times.analysis_time
            ORDER BY
                entity_times.analysis_time,
                entity_times.{entity_id_column}
        """.strip()

        step3_cell = new_code_cell(query_step3)
        df_step3 = execute_sql_query(query_step3)
        step3_cell['execution_count'] = 1

        if df_step3.empty:
            step3_cell.outputs = [
                new_output(
                    output_type='execute_result',
                    data={'text/plain': f"No data returned for query:\n{query_step3}"},
                    execution_count=1
                )
            ]
        else:
            df_step3 = df_step3.replace([None, float('inf'), float('-inf')], None)
            result_json = df_step3.to_dict(orient='records')
            _stringify_timestamps(result_json)

            columns = []
            for col in df_step3.columns:
                guessed_type = infer_column_dtype(df_step3[col])
                columns.append({"name": col, "type": guessed_type})

            text_repr = df_step3.head().to_string(index=False)
            step3_cell.outputs = [
                new_output(
                    output_type='execute_result',
                    data={
                        'application/json': {'rows': result_json, 'columns': columns},
                        'text/plain': text_repr
                    },
                    execution_count=1
                )
            ]
        cells.append(step3_cell)

        step4_markdown = new_markdown_cell("### Step 4: Join additional features (1-year lookback)")
        cells.append(step4_markdown)

        if not extra_features:
            feature_selects = ""
        else:
            feature_selects = ",\n    " + ",\n    ".join([f"tbl.{col}" for col in extra_features])

        step3_sub = query_step3.strip().rstrip(';')
        query_step4 = f"""
            WITH core_set AS (
                {step3_sub}
            )
            SELECT
                core_set.{entity_id_column},
                core_set.analysis_time,
                core_set.{new_target_column}
                {feature_selects}
            FROM core_set
            INNER JOIN {table_name} AS tbl
                ON tbl.{entity_id_column} = core_set.{entity_id_column}
                AND tbl.{time_column} < core_set.analysis_time
                AND tbl.{time_column} >= date_add('year', -1, core_set.analysis_time)
            ORDER BY
                core_set.analysis_time,
                core_set.{entity_id_column}
        """.strip()

        step4_cell = new_code_cell(query_step4)
        df_step4 = execute_sql_query(query_step4)
        step4_cell['execution_count'] = 1

        if df_step4.empty:
            step4_cell.outputs = [
                new_output(
                    output_type='execute_result',
                    data={'text/plain': f"No data returned for query:\n{query_step4}"},
                    execution_count=1
                )
            ]
        else:
            df_step4 = df_step4.replace([None, float('inf'), float('-inf')], None)
            result_json = df_step4.to_dict(orient='records')
            _stringify_timestamps(result_json)

            columns = []
            for col in df_step4.columns:
                guessed_type = infer_column_dtype(df_step4[col])
                columns.append({"name": col, "type": guessed_type})

            text_repr = df_step4.head().to_string(index=False)
            step4_cell.outputs = [
                new_output(
                    output_type='execute_result',
                    data={'application/json': {'rows': result_json, 'columns': columns},
                        'text/plain': text_repr},
                    execution_count=1
                )
            ]
        cells.append(step4_cell)

        nb['cells'] = cells
        return nb, new_target_column  # Return the notebook and the new target column name

    def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str, file_size_mb: float):
        print("[DEBUG] Triggering Glue update for table:", table_name)
        glue = get_glue_client()
        unique_id = file_key.split('/')[1]
        s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/{unique_id}/"
        glue_table_name = self.sanitize_identifier(os.path.splitext(table_name)[0])

        storage_descriptor = {
            'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
            'Location': s3_location,
            'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
            'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
            'SerdeInfo': {
                'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
                'Parameters': {
                    'field.delim': ',',
                    'skip.header.line.count': '1'
                }
            }
        }

        try:
            glue.update_table(
                DatabaseName=ATHENA_SCHEMA_NAME,
                TableInput={
                    'Name': glue_table_name,
                    'StorageDescriptor': storage_descriptor,
                    'TableType': 'EXTERNAL_TABLE'
                }
            )
            print("[DEBUG] Glue table updated successfully:", glue_table_name)
        except glue.exceptions.EntityNotFoundException:
            print("[DEBUG] Glue table not found, creating a new one:", glue_table_name)
            glue.create_table(
                DatabaseName=ATHENA_SCHEMA_NAME,
                TableInput={
                    'Name': glue_table_name,
                    'StorageDescriptor': storage_descriptor,
                    'TableType': 'EXTERNAL_TABLE'
                }
            )
            print("[DEBUG] Glue table created successfully:", glue_table_name)

        user_id = self.get_user_id_from_file_key(file_key)
        if user_id in user_schemas:
            user_schemas[user_id][0]["glue_table_name"] = glue_table_name
            print(f"[DEBUG] Stored Glue table name '{glue_table_name}' for user '{user_id}'.")

        base_timeout = 80
        additional_timeout_per_mb = 5
        dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
        self.wait_for_table_creation(glue_table_name, timeout=dynamic_timeout)

    def wait_for_table_creation(self, table_name, timeout):
        import time
        glue_client = get_glue_client()
        start_time = time.time()
        glue_table_ready = False
        athena_table_ready = False

        print("[DEBUG] Waiting for Glue table creation:", table_name)
        while time.time() - start_time < timeout:
            try:
                glue_client.get_table(DatabaseName=ATHENA_SCHEMA_NAME, Name=table_name)
                print("[DEBUG] Glue table is now available:", table_name)
                glue_table_ready = True
                break
            except glue_client.exceptions.EntityNotFoundException:
                time.sleep(5)
            except Exception as e:
                print("[ERROR] Unexpected error while checking Glue table availability:", e)
                return False

        if not glue_table_ready:
            print(f"[ERROR] Glue table '{table_name}' not available within {timeout} seconds.")
            return False

        print("[DEBUG] Checking Athena table availability:", table_name)
        while time.time() - start_time < timeout:
            try:
                query = f"SELECT 1 FROM {ATHENA_SCHEMA_NAME}.{table_name} LIMIT 1;"
                df = execute_sql_query(query)
                if df.empty:
                    print("[DEBUG] Athena recognizes the table (no error), table ready:", table_name)
                    athena_table_ready = True
                    break
                else:
                    print("[DEBUG] Athena table ready with data:", table_name)
                    athena_table_ready = True
                    break
            except Exception as e:
                error_message = str(e)
                if "TableNotFoundException" in error_message or "TABLE_NOT_FOUND" in error_message:
                    print("[DEBUG] Still waiting for Athena to recognize table:", table_name)
                    time.sleep(10)
                else:
                    print("[ERROR] Unexpected error while checking Athena table availability:", e)
                    return False

        if not athena_table_ready:
            print(f"[ERROR] Table '{table_name}' not available in Athena within {timeout} seconds.")
            return False

        return True

    def get_user_id_from_file_key(self, file_key: str) -> str:
        try:
            return file_key.split("/")[1]
        except IndexError:
            print("[WARNING] Unable to extract user ID, defaulting to 'default_user'.")
            return "default_user"

    def sanitize_identifier(self, name):
        return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())

    def validate_column_exists(self, column_name, columns_list):
        if not column_name:
            return True
        print("[DEBUG] Validating column existence:", column_name)
        print("[DEBUG] Available columns:", columns_list)
        norm_col = normalize_column_name(column_name)
        norm_list = [normalize_column_name(c) for c in columns_list]
        if norm_col in norm_list:
            return True
        else:
            print("[DEBUG] Column not found after normalization:", norm_col)
            return False

    def sanitize_notebook(self, nb):
        """
        Clears the outputs and execution counts from all code cells,
        recursively sanitizes the notebook (e.g., replacing NaN/infs),
        and converts the result back into a NotebookNode.
        This ensures that only the SQL queries (cell sources) are saved
        and that the notebook is in the correct format for nbformat.writes.
        """
        # First, clear outputs and execution counts from code cells.
        for cell in nb.get("cells", []):
            if cell.get("cell_type") == "code":
                cell["outputs"] = []           # Remove query results
                cell["execution_count"] = None # Clear execution count

        # Now, recursively sanitize the notebook to replace any invalid float values.
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [sanitize(item) for item in obj]
            elif isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                return obj
            else:
                return obj

        sanitized_nb = sanitize(nb)
        # Convert the sanitized dictionary back into a NotebookNode
        return nbformat.from_dict(sanitized_nb)


    def create_non_time_based_notebook(
        self,
        entity_id_column: str,
        target_column: str,
        feature_columns: list,
        table_name: str,
        columns_list: list
    ):
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

        print("[DEBUG] Creating combined nonâ€“timeâ€“based notebook...")
        nb = new_notebook()
        cells = []

        # --- Entity & Target Section ---
        md_entity_target = new_markdown_cell("### Entity & Target Query")
        cells.append(md_entity_target)

        sanitized_entity = self.sanitize_identifier(entity_id_column) if entity_id_column.strip() else "*"
        sanitized_target = self.sanitize_identifier(target_column) if target_column.strip() else "*"
        query_entity_target = f"SELECT {sanitized_entity}, {sanitized_target} FROM {table_name} LIMIT 10;"
        if query_entity_target.strip():
            code_entity_target = new_code_cell(query_entity_target)
            cells.append(code_entity_target)

        # --- Features Section ---
        md_features = new_markdown_cell("### Features or Attributes Query")
        cells.append(md_features)

        if feature_columns and any(f.strip() for f in feature_columns):
            sanitized_features = [self.sanitize_identifier(f) for f in feature_columns if f.strip()]
            if sanitized_features:
                query_features = f"SELECT {', '.join(sanitized_features)} FROM {table_name} LIMIT 10;"
                if query_features.strip():
                    code_features = new_code_cell(query_features)
                    cells.append(code_features)
        else:
            md_no_features = new_markdown_cell("No feature columns specified.")
            cells.append(md_no_features)

        filtered_cells = []
        for cell in cells:
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if source.strip() != "":
                filtered_cells.append(cell)

        unique_cells = []
        seen_sources = set()
        for cell in filtered_cells:
            source = cell.get("source", "")
            if isinstance(source, list):
                source = "".join(source)
            if source not in seen_sources:
                unique_cells.append(cell)
                seen_sources.add(source)

        nb['cells'] = unique_cells
        return nb


# -----------------------------------------------------------------------------------
# ChatHistoryByUserView
# -----------------------------------------------------------------------------------
class ChatHistoryByUserView(APIView):
    """
    API to retrieve chat history for a specific user.
    """
    def get(self, request):
        print("DEBUG: ChatHistoryByUserView GET method called")
        user_id = request.GET.get('user_id')
        print(f"DEBUG: Received user_id: {user_id}")

        if not user_id:
            print("ERROR: user_id is missing in the request")
            return Response({"error": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            print(f"DEBUG: Querying chats for user_id: {user_id}")
            chats = ChatBackup.objects.filter(user_id=user_id)
            if not chats.exists():
                print(f"WARNING: No chats found for user_id: {user_id}")
                return Response(
                    {"error": f"No chats found for the given user_id: {user_id}"},
                    status=status.HTTP_404_NOT_FOUND,
                )

            response_data = []
            for chat in chats:
                messages = chat.messages
                user_messages = [msg for msg in messages if msg.get("sender") == "user"]
                assistant_messages = [msg for msg in messages if msg.get("sender") == "assistant"]

                response_data.append({
                    "chat_id": chat.chat_id,
                    "title": chat.title,
                    "user_messages": user_messages,
                    "assistant_messages": assistant_messages,
                })

            print("DEBUG: Successfully prepared response data")
            return Response(response_data, status=status.HTTP_200_OK)

        except ChatBackup.DoesNotExist:
            print(f"ERROR: No records found for user_id: {user_id}")
            return Response(
                {"error": f"No chat records found for user_id={user_id}"},
                status=status.HTTP_404_NOT_FOUND,
            )
        except Exception as e:
            print(f"ERROR: Unexpected error occurred: {e}")
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# -----------------------------------------------------------------------------------
# NotebookView
# -----------------------------------------------------------------------------------
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class NotebookView(APIView):
    def get(self, request):
        user_id = request.query_params.get("user_id")
        chat_id = request.query_params.get("chat_id")

        if not user_id and not chat_id:
            return Response(
                {"error": "At least one of 'user_id' or 'chat_id' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            filters = {}
            if user_id:
                filters["user_id"] = user_id
            if chat_id:
                filters["chat"] = chat_id

            notebooks = Notebook.objects.filter(**filters)
            if not notebooks.exists():
                return Response(
                    {"error": "No notebooks found for the given criteria."},
                    status=status.HTTP_404_NOT_FOUND,
                )

            notebook_data = [
                {
                    "id": notebook.id,
                    "user_id": notebook.user_id,
                    "chat_id": notebook.chat,
                    "entity_column": notebook.entity_column,
                    "target_column": notebook.target_column,
                    "time_column": notebook.time_column,
                    "time_frame": notebook.time_frame,
                    "time_frequency": notebook.time_frequency,
                    "features": notebook.features,
                    "file_url": notebook.file_url,
                    "notebook_json": notebook.notebook_json,
                    "cell_s3_links": notebook.cell_s3_links,
                    "created_at": notebook.created_at,
                }
                for notebook in notebooks
            ]

            return Response(
                {
                    "message": "Notebooks retrieved successfully.",
                    "user_id": user_id,
                    "notebooks": notebook_data,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# -----------------------------------------------------------------------------------
# PredictiveSettingsDetailView
# -----------------------------------------------------------------------------------
from .models import PredictiveSettings  # Adjust the import if your model is in a different module

class PredictiveSettingsDetailView(APIView):
    def get(self, request, user_id, chat_id):
        try:
            ps = PredictiveSettings.objects.get(user_id=user_id, chat_id=chat_id)
            
            # Validate values against schema
            if user_id in user_schemas and user_schemas[user_id]:
                schema_columns = [col["column_name"] for col in user_schemas[user_id][0]["schema"]]
                if ps.target_column and ps.target_column not in schema_columns:
                    logger.error(f"Invalid target column {ps.target_column} for user {user_id}")
                    
            data = {
                "target_column": ps.target_column if ps.target_column else "Null",
                "entity_column": ps.entity_column if ps.entity_column else "Null",
                "time_column": ps.time_column if ps.time_column else "Null",
                "predictive_question": ps.predictive_question if ps.predictive_question else "Null",
                "time_frame": ps.time_frame if ps.time_frame else "Null",
                "time_frequency": ps.time_frequency if ps.time_frequency else "Null",
                "machine_learning_type": ps.machine_learning_type if ps.machine_learning_type else "Null",
                "features": ps.features if ps.features else [],
                "prediction_type": ps.prediction_type if ps.prediction_type else "Null",
                "new_target_column": ps.new_target_column if ps.new_target_column else "Null",
                # ... other fields ...
            }
            
            return Response(data, status=status.HTTP_200_OK)
            
        except PredictiveSettings.DoesNotExist:
            return Response(
                {"error": "Settings not found"},
                status=status.HTTP_404_NOT_FOUND
            )




