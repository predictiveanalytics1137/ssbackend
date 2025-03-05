

import os
import uuid
import datetime
import re
import boto3
import pandas as pd
import openai
import requests
import numpy as np
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

from .models import FileSchema, Notebook, UploadedFile, ChatBackup
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

# Initialize the ChatOpenAI model
llm_chatgpt = ChatOpenAI(
    model="gpt-3.5-turbo-16k",
    temperature=0,
    openai_api_key=OPENAI_API_KEY,
)

# In-memory storage for user-specific data
user_conversations = {}
user_schemas = {}
user_confirmations = {}
user_notebook_flags = {}
user_notebooks = {}

# Prompt template for ChatGPT
prompt_chatgpt = PromptTemplate(
    input_variables=["history", "user_input"],
    template=(
        "You are a highly intelligent and helpful AI assistant. Your "
        "responses should be clear and concise. Assist the user in confirming "
        "the dataset schema and any corrections they provide. Reflect the "
        "confirmed or corrected schema back to the user before proceeding.\n\n"
        "Steps:\n"
        "1. Discuss the subject they want to predict.\n"
        "2. Confirm the target value they want to predict.\n"
        "3. Check if there's a specific time frame for the prediction.\n"
        "4. Reference the dataset schema if available.\n"
        "5. Once you have confirmed all necessary information with the user, "
        "provide a summary of the inputs and let them know they can generate the notebook.\n\n"
        "Conversation history:\n{history}\n"
        "User input:\n{user_input}\n"
        "Assistant:"
    ),
)


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

def normalize_column_name(col_name: str) -> str:
    return col_name.strip().lower().replace(' ', '_')

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


# ----------------------------------------------------------------------
# Flexible date/time detection
# ----------------------------------------------------------------------
def infer_column_dtype(series: pd.Series, threshold: float = 0.8) -> str:
    """
    Attempt to infer the most likely data type of a pandas Series.
    This version uses 'errors=coerce' for date parsing to avoid skipping rows
    and a threshold-based approach to classify a column as timestamp.
    """

    # Drop NA, convert to string, strip whitespace
    series = series.dropna().astype(str).str.strip()
    if series.empty:
        return "string"

    # Attempt #1: flexible parse toggling dayfirst, errors='coerce'
    # If at least `threshold` fraction of non-null values are valid dates, call it timestamp.
    for dayfirst_val in [False, True]:
        dt_series = pd.to_datetime(series, dayfirst=dayfirst_val, errors='coerce')
        valid_count = dt_series.notnull().sum()
        total_count = len(series)
        if total_count > 0 and (valid_count / total_count) >= threshold:
            return "timestamp"

    # Attempt #2: known common formats
    known_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y"]
    for fmt in known_formats:
        dt_series = pd.to_datetime(series, format=fmt, errors='coerce')
        valid_count = dt_series.notnull().sum()
        total_count = len(series)
        if total_count > 0 and (valid_count / total_count) >= threshold:
            return "timestamp"

    # Check for boolean
    boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}
    unique_values = set(series.str.lower().unique())
    if unique_values.issubset(boolean_values):
        return "boolean"

    # Check integer
    try:
        int_series = pd.to_numeric(series, errors='raise')
        # if all integer (no fraction), decide between int and bigint
        if (int_series % 1 == 0).all():
            int_min = int_series.min()
            int_max = int_series.max()
            if int_min >= -2147483648 and int_max <= 2147483647:
                return "int"
            else:
                return "bigint"
    except ValueError:
        pass

    # Check double
    try:
        pd.to_numeric(series, errors='raise', downcast='float')
        return "double"
    except ValueError:
        pass

    return "string"


def standardize_datetime_columns(df: pd.DataFrame, schema: list) -> pd.DataFrame:
    """
    Given a DataFrame and a schema (list of {'column_name', 'data_type'}),
    convert columns recognized as 'timestamp' into a standard ISO datetime string:
    'YYYY-MM-DD HH:MM:SS'.
    """
    for colinfo in schema:
        if colinfo["data_type"] == "timestamp":
            col_name = colinfo["column_name"]
            try:
                df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
                df[col_name] = df[col_name].dt.strftime("%Y-%m-%d %H:%M:%S")
                print(f"[DEBUG] Standardized datetime column: {col_name}")
            except Exception as e:
                print(f"[WARNING] Could not standardize date column '{col_name}': {str(e)}")
    return df


def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
    return df.columns[-1]


def suggest_entity_id_column(df: pd.DataFrame) -> Any:
    likely_id_columns = [col for col in df.columns if "id" in col.lower()]
    for col in likely_id_columns:
        if df[col].nunique() / len(df) > 0.95:
            return col
    for col in df.columns:
        if df[col].nunique() / len(df) > 0.95:
            return col
    return None

def parse_user_adjustments(user_input, uploaded_file_info):
    print("[DEBUG] Parsing user adjustments...")
    columns_list = [col['column_name'] for col in uploaded_file_info['schema']]
    normalized_columns = [normalize_column_name(c) for c in columns_list]
    print("[DEBUG] Current columns_list:", columns_list)
    print("[DEBUG] Normalized columns_list:", normalized_columns)
    print("[DEBUG] User input:", user_input)

    adjustments = {}
    lines = user_input.strip().split(',')
    for line in lines:
        if ':' in line:
            key, value = line.split(':', 1)
            key = key.strip().lower()
            value = value.strip()
            val_norm = normalize_column_name(value)

            if 'entity' in key and 'column' in key:
                if val_norm in normalized_columns:
                    match_col = columns_list[normalized_columns.index(val_norm)]
                    adjustments['entity_id_column'] = match_col
                else:
                    print("[DEBUG] Entity ID column not found:", val_norm)

            elif 'target' in key and 'column' in key:
                if val_norm in normalized_columns:
                    match_col = columns_list[normalized_columns.index(val_norm)]
                    adjustments['target_column'] = match_col
                else:
                    print("[DEBUG] Target column not found:", val_norm)

            elif 'time' in key and 'column' in key:
                if val_norm in normalized_columns:
                    match_col = columns_list[normalized_columns.index(val_norm)]
                    adjustments['time_column'] = match_col
                else:
                    print("[DEBUG] Time column not found:", val_norm)

            elif 'time' in key and 'frame' in key:
                adjustments['time_frame'] = value

            # [ADDED CODE FOR FREQUENCY]
            elif 'time' in key and 'frequency' in key:
                # e.g. "Time Frequency: monthly"
                adjustments['time_frequency'] = value

    # If entity/target => auto-build features
    if adjustments.get('entity_id_column') and adjustments.get('target_column'):
        entity_id = adjustments['entity_id_column']
        target_col = adjustments['target_column']
        feature_columns = [
            {'column_name': col} for col in columns_list
            if col not in [entity_id, target_col]
        ]
        adjustments['feature_columns'] = feature_columns
        print("[DEBUG] Adjustments found:", adjustments)

    return adjustments if adjustments else None


class UnifiedChatGPTAPI(APIView):
    # import pdb; pdb.set_trace()
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
  

    def handle_file_upload(self, request, files: List[Any]):
        user_id = request.data.get("user_id", "default_user")
        s3 = get_s3_client()
        glue = get_glue_client()
        uploaded_files_info = []

        for file in files:
            print(f"[DEBUG] Processing file: {file.name}")
            try:
                if file.name.lower().endswith('.csv'):
                    df = pd.read_csv(
                        file,
                        low_memory=False,
                        encoding='utf-8',
                        delimiter=',',
                        na_values=['NA', 'N/A', ''],
                        on_bad_lines='warn'
                    )
                else:
                    df = pd.read_excel(file, engine='openpyxl')

                if df.empty:
                    print("[ERROR] File is empty:", file.name)
                    return Response({"error": f"Uploaded file {file.name} is empty."}, 
                                    status=status.HTTP_400_BAD_REQUEST)

                if not df.columns.any():
                    print("[ERROR] File has no columns:", file.name)
                    return Response({"error": f"Uploaded file {file.name} has no columns."}, 
                                    status=status.HTTP_400_BAD_REQUEST)

            except pd.errors.ParserError as e:
                print("[ERROR] CSV parsing error:", e)
                return Response({"error": f"CSV parsing error for file {file.name}: {str(e)}"}, 
                                status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                print("[ERROR] Error reading file:", e)
                return Response({"error": f"Error reading file {file.name}: {str(e)}"}, 
                                status=status.HTTP_400_BAD_REQUEST)

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

            raw_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]
            df = standardize_datetime_columns(df, raw_schema)
            final_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]

            has_date_column = any(c["data_type"] == "timestamp" for c in final_schema)
            possible_date_cols = [c["column_name"] for c in final_schema if c["data_type"] == "timestamp"]

            boolean_columns = [c['column_name'] for c in final_schema if c['data_type'] == 'boolean']
            replacement_dict = {
                '1': 'true','0': 'false',
                'yes':'true','no':'false',
                't':'true','f':'false',
                'y':'true','n':'false',
                'true':'true','false':'false',
            }
            for col_name in boolean_columns:
                df[col_name] = df[col_name].astype(str).str.strip().str.lower().replace(replacement_dict)
                unexpected_values = [v for v in df[col_name].unique() if v not in ['true','false']]
                if unexpected_values:
                    print("[ERROR] Unexpected boolean values:", unexpected_values)
                    return Response({"error": f"Unexpected boolean values in column {col_name}: {unexpected_values}"}, 
                                    status=status.HTTP_400_BAD_REQUEST)

            file_name_base, file_extension = os.path.splitext(file.name)
            file_name_base = file_name_base.lower().replace(' ', '_')
            unique_id = uuid.uuid4().hex[:8]
            new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
            s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
            file_key = f"uploads/{unique_id}/{s3_file_name}"
            print("[DEBUG] Uploading file to S3 at key:", file_key)

            try:
                with transaction.atomic():
                    file.seek(0)
                    file_serializer = UploadedFileSerializer(data={'name': new_file_name, 'file': file})
                    if file_serializer.is_valid():
                        file_instance = file_serializer.save()

                        csv_buffer = BytesIO()
                        df.to_csv(csv_buffer, index=False, encoding='utf-8')
                        csv_buffer.seek(0)
                        s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, file_key)
                        print("[DEBUG] S3 upload successful:", file_key)
                        s3.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=file_key)
                        file_url = f"s3://{AWS_STORAGE_BUCKET_NAME}/{file_key}"
                        file_instance.file_url = file_url
                        file_instance.save()

                        FileSchema.objects.create(file=file_instance, schema=final_schema)

                        file_size_mb = file.size / (1024 * 1024)
                        self.trigger_glue_update(new_file_name, final_schema, file_key, file_size_mb)

                        uploaded_files_info.append({
                            'id': file_instance.id,
                            'name': file_instance.name,
                            'file_url': file_instance.file_url,
                            'schema': final_schema,
                            'file_size_mb': file_size_mb,
                            'has_date_column': has_date_column,
                            'date_columns': possible_date_cols,
                            'suggestions': {
                                'target_column': suggest_target_column(df, []),
                                'entity_id_column': suggest_entity_id_column(df),
                                'feature_columns': [
                                    c for c in df.columns
                                    if c not in [
                                        suggest_entity_id_column(df),
                                        suggest_target_column(df, [])
                                    ]
                                ]
                            }
                        })
                    else:
                        print("[ERROR] File serializer errors:", file_serializer.errors)
                        return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)

            except ClientError as e:
                print("[ERROR] AWS ClientError:", e)
                return Response({'error': f'AWS error: {str(e)}'}, 
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            except Exception as e:
                print("[ERROR] Unexpected error during file processing:", e)
                return Response({'error': f'File processing failed: {str(e)}'}, 
                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)

        user_schemas[user_id] = uploaded_files_info

        chat_id = request.data.get("chat_id", "")
        memory_key = f"{user_id}_{chat_id}" if chat_id else user_id

        if memory_key not in user_conversations:
            conversation_chain = ConversationChain(
                llm=llm_chatgpt,
                prompt=prompt_chatgpt,
                input_key="user_input",
                memory=ConversationBufferMemory()
            )
            user_conversations[memory_key] = conversation_chain
        else:
            conversation_chain = user_conversations[memory_key]

        schema_discussion = self.format_schema_message(uploaded_files_info[0])
        conversation_chain.memory.chat_memory.messages.append(
            AIMessage(content=schema_discussion)
        )

        print("[DEBUG] Files uploaded and schema discussion initiated.")
        return Response({
            "message": "Files uploaded and processed successfully.",
            "uploaded_files": uploaded_files_info,
            "chat_message": schema_discussion
        }, status=status.HTTP_201_CREATED)

    def handle_chat(self, request):
        user_input = request.data.get("message", "").strip()
        user_id = request.data.get("user_id", "default_user")
        chat_id = request.data.get("chat_id")

        if not user_input:
            return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Ensure user exists
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        # If no chat_id => create a new one
        if not chat_id:
            chat_id = str(uuid.uuid4())
            print(f"[DEBUG] New chat created with chat_id: {chat_id}")
            chat_title = user_input[:50]
            print(f"[DEBUG] New chat title: {chat_title}")

            ChatBackup.objects.create(
                user=user,
                chat_id=chat_id,
                title=chat_title,
                messages=[]
            )

        chat_obj, created = ChatBackup.objects.get_or_create(
            user=user, chat_id=chat_id,
            defaults={"title": user_input[:50], "messages": []}
        )

        memory_key = f"{user_id}_{chat_id}"

        # If conversation chain missing, restore from ChatBackup
        if memory_key not in user_conversations:
            print(f"[DEBUG] Restoring memory for chat_id: {chat_id}")
            restored_memory = ConversationBufferMemory()
            for msg in chat_obj.messages:
                sender = msg["sender"]
                text = msg["text"]
                if sender == "assistant":
                    restored_memory.chat_memory.add_message(AIMessage(content=text))
                else:
                    restored_memory.chat_memory.add_message(HumanMessage(content=text))

            user_conversations[memory_key] = ConversationChain(
                llm=llm_chatgpt,
                prompt=prompt_chatgpt,
                input_key="user_input",
                memory=restored_memory
            )

        conversation_chain = user_conversations[memory_key]

        show_generate_notebook = False
        confirmation_response, confirmed_flag = self.process_schema_confirmation(user_input, user_id)

        if confirmed_flag:
            assistant_response = confirmation_response
            show_generate_notebook = True
        else:
            assistant_response = conversation_chain.run(user_input=user_input)

        # Save to DB
        chat_obj.messages.append({
            "sender": "user",
            "text": user_input,
            "timestamp": datetime.datetime.now().isoformat()
        })
        chat_obj.messages.append({
            "sender": "assistant",
            "text": assistant_response,
            "timestamp": datetime.datetime.now().isoformat()
        })
        chat_obj.save()

        return Response({
            "response": assistant_response,
            "chat_id": chat_id,
            "show_generate_notebook": show_generate_notebook
        })

    def process_schema_confirmation(self, user_input, user_id):
        print("[DEBUG] Processing schema confirmation for user:", user_id)

        if user_id not in user_schemas or not user_schemas[user_id]:
            return ("", False)

        uploaded_file_info = user_schemas[user_id][0]
        suggestions = uploaded_file_info['suggestions']
        has_date_column = uploaded_file_info.get('has_date_column', False)
        date_cols = uploaded_file_info.get('date_columns', [])

        if user_input.lower() == 'yes':
            print("[DEBUG] User confirmed suggested schema.")
            user_confirmations[user_id] = {
                'entity_id_column': suggestions['entity_id_column'],
                'target_column': suggestions['target_column'],
                'feature_columns': [{'column_name': col} for col in suggestions['feature_columns']],
                'time_column': date_cols[0] if (has_date_column and len(date_cols) == 1) else None,
                'time_frame': None,
                # [ADDED CODE FOR FREQUENCY] - Initialize to None
                'time_frequency': None
            }

            assistant_response = (
                f"Great! You've confirmed the schema for '{uploaded_file_info['name']}':\n\n"
                f"- Entity ID Column: {suggestions['entity_id_column']}\n"
                f"- Target Column: {suggestions['target_column']}\n"
                f"- Feature Columns: {', '.join(suggestions['feature_columns'])}\n"
            )

            if has_date_column:
                if len(date_cols) == 1:
                    assistant_response += f"- Time Column: {date_cols[0]}\n\n"
                    assistant_response += (
                        "You have a time-based dataset. We'll use the single detected date column.\n"
                        "Now please specify the Time Frame if you'd like, e.g. 'Time Frame: 1 WEEK'.\n"
                        "Or you can proceed to generate the notebook."
                    )
                else:
                    assistant_response += (
                        "\nWe detected multiple date columns in your dataset. Please specify the Time Column:\n"
                        f"{date_cols}\n\n"
                        "Example: 'Time Column: <column_name>'\n"
                        "Then also specify the Time Frame if needed, e.g. 'Time Frame: 2 WEEKS'."
                    )
            else:
                assistant_response += (
                    "\nNo date column detected. We'll proceed with a non-time-based approach unless you specify otherwise.\n"
                )
            return (assistant_response, True)

        # If not simple "yes", parse user adjustments
        adjusted_columns = parse_user_adjustments(user_input, uploaded_file_info)
        if adjusted_columns:
            existing_conf = user_confirmations.get(user_id, {
                'entity_id_column': suggestions['entity_id_column'],
                'target_column': suggestions['target_column'],
                'feature_columns': [{'column_name': c} for c in suggestions['feature_columns']],
                'time_column': None,
                'time_frame': None,
                # [ADDED CODE FOR FREQUENCY]
                'time_frequency': None
            })

            if 'entity_id_column' in adjusted_columns:
                existing_conf['entity_id_column'] = adjusted_columns['entity_id_column']
            if 'target_column' in adjusted_columns:
                existing_conf['target_column'] = adjusted_columns['target_column']
                columns_list = [col['column_name'] for col in uploaded_file_info['schema']]
                entity_id_col = existing_conf['entity_id_column']
                target_col = existing_conf['target_column']
                feature_names = [c for c in columns_list if c not in [entity_id_col, target_col]]
                existing_conf['feature_columns'] = [{'column_name': c} for c in feature_names]

            if 'time_column' in adjusted_columns:
                if adjusted_columns['time_column'] in date_cols:
                    existing_conf['time_column'] = adjusted_columns['time_column']
                else:
                    print("[DEBUG] Provided time column is not recognized among:", date_cols)

            if 'time_frame' in adjusted_columns:
                existing_conf['time_frame'] = adjusted_columns['time_frame']

            # [ADDED CODE FOR FREQUENCY]
            if 'time_frequency' in adjusted_columns:
                existing_conf['time_frequency'] = adjusted_columns['time_frequency']

            user_confirmations[user_id] = existing_conf

            entity_id_col = existing_conf['entity_id_column']
            target_col = existing_conf['target_column']
            feature_col_objs = existing_conf['feature_columns']
            feature_names = [obj['column_name'] for obj in feature_col_objs]
            time_col = existing_conf.get('time_column', None)
            time_frame = existing_conf.get('time_frame', None)
            # [ADDED CODE FOR FREQUENCY]
            time_frequency = existing_conf.get('time_frequency', None)

            assistant_response = (
                f"Thanks for the corrections! The updated schema for '{uploaded_file_info['name']}' is:\n\n"
                f"- Entity ID Column: {entity_id_col}\n"
                f"- Target Column: {target_col}\n"
                f"- Feature Columns: {', '.join(feature_names) if feature_names else 'None'}\n"
            )
            if has_date_column:
                if time_col:
                    assistant_response += f"- Time Column: {time_col}\n"
                    if time_frame:
                        assistant_response += f"- Time Frame: {time_frame}\n"
                    # [ADDED CODE FOR FREQUENCY]
                    if time_frequency:
                        assistant_response += f"- Time Frequency: {time_frequency}\n"

                    if time_frame:
                        assistant_response += (
                            "\nTime-based approach is now fully confirmed. You can proceed to generate the notebook."
                        )
                    else:
                        assistant_response += (
                            "\nWe have the date column now. Please also specify the Time Frame in the format 'Time Frame: <X>' if you'd like a time-based window.\n"
                            "Or proceed to generate a partial time-based approach."
                        )
                else:
                    assistant_response += (
                        "\nWe detected a date column, but you haven't specified which to use yet.\n"
                        f"Available date columns: {date_cols}\n"
                        "Please provide the time column in the format: 'Time Column: <column_name>'.\n"
                        "Also optionally specify 'Time Frame: <X>'."
                    )
            else:
                assistant_response += "\nNo date column found, so we'll proceed non-time-based."

            return (assistant_response, True)

        return ("", False)

    def reset_conversation(self, request):
        user_id = request.data.get("user_id", "default_user")
        print("[DEBUG] Resetting conversation for user:", user_id)
        if user_id in user_conversations:
            del user_conversations[user_id]
        if user_id in user_schemas:
            del user_schemas[user_id]
        if user_id in user_confirmations:
            del user_confirmations[user_id]
        if user_id in user_notebook_flags:
            del user_notebook_flags[user_id]
        if user_id in user_notebooks:
            del user_notebooks[user_id]
        return Response({"message": "Conversation reset successful."})

    def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
        schema = uploaded_file['schema']
        target_column = uploaded_file['suggestions']['target_column']
        entity_id_column = uploaded_file['suggestions']['entity_id_column']
        feature_columns = uploaded_file['suggestions']['feature_columns']
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
                    "You can also specify a 'Time Frame: <X>' (e.g., 'Time Frame: 1 WEEK') or a 'Time Frequency: <daily|weekly|monthly>' for how to increment.\n\n"
                )
            elif len(date_cols) > 1:
                schema_text += (
                    "We detected multiple date columns. Please specify which one to use as the time column:\n"
                    f"{date_cols}\n\n"
                    "And also specify 'Time Frame: <X>' or 'Time Frequency: <daily|weekly|monthly>' if you’d like a time-based approach.\n"
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
            "- Optionally specify 'Time Frame: <X>' (e.g. 'Time Frame: 2 WEEKS') and 'Time Frequency: <daily|weekly|monthly>'.\n\n"
            "(Reply 'yes' to confirm or provide corrections in the format:\n"
            "'Entity ID Column: <column>, Target Column: <column>, Time Column: <column>, Time Frame: <X>, Time Frequency: <Y>')"
        )
        return schema_text

    def generate_notebook(self, request):
        user_id = request.data.get("user_id")
        chat_id = request.data.get("chat_id")
        print("[DEBUG] Generating notebook for user_id:", user_id, "and chat_id:", chat_id)

        if not user_id or not chat_id:
            print("[ERROR] user_id or chat_id missing in the request.")
            return Response({"error": "user_id and chat_id are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            print(f"[ERROR] User with id {user_id} not found.")
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

        try:
            chat = ChatBackup.objects.get(chat_id=chat_id, user=user)
        except ChatBackup.DoesNotExist:
            print(f"[ERROR] Chat with id {chat_id} not found for user {user_id}.")
            return Response({"error": "Chat not found."}, status=status.HTTP_404_NOT_FOUND)

        if user_id not in user_confirmations:
            print("[ERROR] Schema not confirmed yet.")
            return Response({"error": "Schema not confirmed yet."}, status=status.HTTP_400_BAD_REQUEST)

        confirmation = user_confirmations[user_id]
        entity_id_column = confirmation.get('entity_id_column')
        target_column = confirmation.get('target_column')
        feature_columns = [col['column_name'] for col in confirmation.get('feature_columns', [])]
        time_column = confirmation.get('time_column', None)
        time_frame = confirmation.get('time_frame', None)
        # [ADDED CODE FOR FREQUENCY]
        time_frequency = confirmation.get('time_frequency', None)

        if user_id in user_schemas:
            uploaded_file_info = user_schemas[user_id][0]
            table_name_raw = os.path.splitext(uploaded_file_info['name'])[0]
            sanitized_table_name = self.sanitize_identifier(table_name_raw)
            file_url = uploaded_file_info.get('file_url')
            has_date_column = uploaded_file_info.get('has_date_column', False)
        else:
            print("[ERROR] Uploaded file info not found.")
            return Response({"error": "Uploaded file info not found."}, status=status.HTTP_400_BAD_REQUEST)

        columns_list = [col['column_name'] for col in uploaded_file_info['schema']]

        if entity_id_column and not self.validate_column_exists(entity_id_column, columns_list):
            return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)
        if target_column and not self.validate_column_exists(target_column, columns_list):
            return Response({"error": f"Target column '{target_column}' does not exist."}, status=status.HTTP_400_BAD_REQUEST)

        import nbformat

        # TIME-BASED APPROACH
        if has_date_column and time_column:
            print("[DEBUG] Time-based approach triggered.")
            final_time_frame = time_frame if time_frame else "1 WEEK"

            final_notebook = self.create_dynamic_time_based_notebook(
                entity_id_column=entity_id_column,
                time_column=time_column,
                target_column=target_column,
                table_name=sanitized_table_name,
                time_horizon=final_time_frame,
                extra_features=feature_columns,
                # cross_join_limit=1000,
                # [ADDED CODE FOR FREQUENCY]
                time_frequency=time_frequency
            )
            notebook_sanitized = self.sanitize_notebook(final_notebook)
            notebook_json = nbformat.writes(notebook_sanitized, version=4)

            # Save to database
            notebook = Notebook.objects.create(
                user=user,
                chat=chat_id,
                entity_column=entity_id_column,
                target_column=target_column,
                time_column=time_column,
                time_frame=final_time_frame,
                time_frequency=time_frequency,
                features=feature_columns,
                file_url=file_url,
                notebook_json=notebook_json
            )

            user_notebooks[user_id] = {
                'time_based_notebook': notebook_json
            }
            # response_data = {
            #     "message": "Notebooks generated successfully (time-based).",
            #     "notebooks": user_notebooks[user_id],
            #     "file_url": file_url,
            #     "entity_column": entity_id_column,
            #     "target_column": target_column,
            #     "time_column": time_column,
            #     "time_frame": final_time_frame,
            #     # [ADDED CODE FOR FREQUENCY]
            #     "time_frequency": time_frequency,
            #     "features": feature_columns,
            #     "user_id": user_id,
            #     "chat_id": chat_id,
            # }
            # Return response
            response_data = {
                "message": "Notebook generated and saved successfully.",
                "notebook_id": notebook.id,
                "notebook_data": notebook_json
            }
            print(response_data)
            print("[DEBUG] Returning data to the frontend (time-based approach)...", response_data)
            return Response(response_data, status=status.HTTP_200_OK)

        # NON-TIME-BASED APPROACH
        else:
            print("[DEBUG] Non-time-based approach triggered.")
            notebook_entity_target = self.create_entity_target_notebook(
                entity_id_column,
                target_column,
                sanitized_table_name,
                columns_list
            )
            notebook_features = self.create_features_notebook(
                feature_columns,
                sanitized_table_name,
                columns_list
            )

            notebook_entity_target_sanitized = self.sanitize_notebook(notebook_entity_target)
            notebook_features_sanitized = self.sanitize_notebook(notebook_features)
            notebook_entity_target_json = nbformat.writes(notebook_entity_target_sanitized, version=4)
            notebook_features_json = nbformat.writes(notebook_features_sanitized, version=4)

            # Save to database as two separate records
            notebook_entity_target_record = Notebook.objects.create(
                user=user,
                chat=chat_id,
                entity_column=entity_id_column,
                target_column=target_column,
                features=feature_columns,
                file_url=file_url,
                notebook_json=notebook_entity_target_json
            )

            notebook_features_record = Notebook.objects.create(
                user=user,
                chat=chat_id,
                entity_column=entity_id_column,
                target_column=target_column,
                features=feature_columns,
                file_url=file_url,
                notebook_json=notebook_features_json
            )

            user_notebooks[user_id] = {
                'entity_target_notebook': notebook_entity_target_json,
                'features_notebook': notebook_features_json
            }

            # response_data = {
            #     "message": "Notebooks generated successfully (non-time-based).",
            #     "notebooks": user_notebooks[user_id],
            #     "file_url": file_url,
            #     "entity_column": entity_id_column,
            #     "target_column": target_column,
            #     "features": feature_columns,
            #     "user_id": user_id,
            #     "chat_id": chat_id,
            # }
            response_data = {
            "message": "Notebooks generated and saved successfully (non-time-based).",
            "entity_target_notebook_id": notebook_entity_target_record.id,
            "features_notebook_id": notebook_features_record.id,
            "entity_target_notebook": notebook_entity_target_json,
            "features_notebook": notebook_features_json
        }
            print("[DEBUG] Returning data to the frontend (non-time-based approach)...", response_data)
            return Response(response_data, status=status.HTTP_200_OK)

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
        """
        Creates a notebook with 4 steps (similar to Pecan's approach):

        1) Generate daily increments from earliest to latest date, then
        truncate by user’s time frequency (day/week/month).
        2) For each entity, keep only times after it first appears.
        3) Summarize the target within the next X time units after that
        analysis_time (partial window filter included).
        4) Join in features from the prior 1 year of data (lookback).

        Updated to ensure we gather *all* monthly snapshots from each
        entity's earliest date and do not limit to only the last date.
        """
        import nbformat
        import datetime
        import numpy as np
        import pandas as pd
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

        # Helper to parse the "6 months", "3 weeks", etc.
        def parse_time_horizon(th: str):
            parts = th.strip().split()
            if len(parts) < 2:
                return 1, "week"
            number = parts[0]
            unit = parts[1].lower().rstrip('s')
            return int(number), unit

        # Helper to format Timestamps for output
        def _stringify_timestamps(rows):
            for row in rows:
                for k, v in row.items():
                    if isinstance(v, (pd.Timestamp, datetime.datetime)):
                        row[k] = v.strftime("%Y-%m-%d %H:%M:%S")

        horizon_number, horizon_unit = parse_time_horizon(time_horizon)

        # Decide the `date_trunc` unit based on time_frequency
        if time_frequency:
            freq_lower = time_frequency.strip().lower()
            if freq_lower.startswith("day"):
                increment_unit = "day"
            elif freq_lower.startswith("week"):
                increment_unit = "week"
            else:
                # default to month
                increment_unit = "month"
        else:
            increment_unit = "month"

        nb = new_notebook()
        cells = []

        ############################################################################
        # STEP 1: Generate daily increments, then truncate by time_frequency
        ############################################################################
        step1_markdown = new_markdown_cell(
            "### Step 1: Determine relevant timestamps based on Time Frequency"
        )
        cells.append(step1_markdown)

        # Removed the small LIMIT entirely so we get *all* relevant timestamps.
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
            LIMIT 1000
            
        """.strip()

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
                        'application/json': {
                            'rows': result_json,
                            'columns': columns
                        },
                        'text/plain': text_repr
                    },
                    execution_count=1
                )
            ]
        cells.append(step1_cell)

        ############################################################################
        # STEP 2: For each entity, gather relevant times after earliest record
        ############################################################################
        step2_markdown = new_markdown_cell(
            "### Step 2: For each entity, gather relevant times after earliest record"
        )
        cells.append(step2_markdown)

        # Use *all* relevant times in ascending order, no small limit that cuts off earlier months.
        step1_sub = query_step1.strip().rstrip(';')
        query_step2 = f"""
            WITH entity_earliest_time AS (
                SELECT
                    {entity_id_column} AS entity_id,
                    MIN({time_column}) AS first_seen_time
                FROM {table_name}
                GROUP BY {entity_id_column}
            ),
            relevant_times_in_dataset AS (
                {step1_sub}
            )
            SELECT
                entity_earliest_time.entity_id,
                relevant_times_in_dataset.relevant_time AS analysis_time
            FROM entity_earliest_time
            JOIN relevant_times_in_dataset
                ON relevant_times_in_dataset.relevant_time >= entity_earliest_time.first_seen_time
            ORDER BY analysis_time, entity_id
            LIMIT 1000
            
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
                    data={
                        'application/json': {'rows': result_json, 'columns': columns},
                        'text/plain': text_repr
                    },
                    execution_count=1
                )
            ]
        cells.append(step2_cell)

        ############################################################################
        # STEP 3: Summarize the target measure over the chosen time horizon
        #         (exclude partial windows after the last full horizon)
        ############################################################################
        step3_markdown = new_markdown_cell(
            "### Step 3: Summarize the target measure over the chosen time horizon (partial window filter)"
        )
        cells.append(step3_markdown)

        step2_sub = query_step2.strip().rstrip(';')
        horizon_label = time_horizon.replace(' ', '_')
        query_step3 = f"""
            WITH last_time AS (
                SELECT MAX({time_column}) AS max_ts
                FROM {table_name}
            ),
            entity_times AS (
                {step2_sub}
            )
            SELECT
                entity_times.entity_id,
                entity_times.analysis_time,
                COALESCE(SUM(tbl.{target_column}), 0) AS target_within_{horizon_label}_after
            FROM entity_times
            LEFT JOIN {table_name} AS tbl
                ON tbl.{entity_id_column} = entity_times.entity_id
                AND tbl.{time_column} >= entity_times.analysis_time
                AND tbl.{time_column} < date_add('{horizon_unit}', {horizon_number}, entity_times.analysis_time)
            WHERE entity_times.analysis_time <= date_add('{horizon_unit}', -{horizon_number}, (SELECT max_ts FROM last_time))
            GROUP BY
                entity_times.entity_id,
                entity_times.analysis_time
            ORDER BY
                entity_times.analysis_time,
                entity_times.entity_id
                LIMIT 1000
                
        """.strip()

#         query_step3 = f""" WITH last_time AS (
#     SELECT MAX({time_column}) AS max_ts
#     FROM {table_name}
# ),
# entity_times AS (
#     {step2_sub}
# )
# SELECT
#     entity_times.entity_id,
#     entity_times.analysis_time,
#     COALESCE(SUM(tbl.{target_column}), 0) AS target_within_{horizon_label}_after
# FROM entity_times
# LEFT JOIN {table_name} AS tbl
#     ON tbl.{entity_id_column} = entity_times.entity_id
#     AND tbl.{time_column} >= entity_times.analysis_time
#     AND tbl.{time_column} < entity_times.analysis_time + INTERVAL '{horizon_number}' {horizon_unit}
# WHERE entity_times.analysis_time <= (SELECT max_ts - INTERVAL '{horizon_number}' {horizon_unit} FROM last_time)
# GROUP BY
#     entity_times.entity_id,
#     entity_times.analysis_time
# ORDER BY
#     entity_times.analysis_time,
#     entity_times.entity_id
#     LIMIT 1000;
# """.strip()

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

        ############################################################################
        # STEP 4: Join additional features (1-year lookback)
        ############################################################################
        step4_markdown = new_markdown_cell(
            "### Step 4: Join additional features (1-year lookback before analysis_time)"
        )
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
                core_set.entity_id,
                core_set.analysis_time,
                core_set.target_within_{horizon_label}_after
                {feature_selects}
            FROM core_set
            INNER JOIN {table_name} AS tbl
                ON tbl.{entity_id_column} = core_set.entity_id
                AND tbl.{time_column} < core_set.analysis_time
                AND tbl.{time_column} >= date_add('year', -1, core_set.analysis_time)
            ORDER BY
                core_set.analysis_time,
                core_set.entity_id
                LIMIT 1000
                
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
                    data={
                        'application/json': {'rows': result_json, 'columns': columns},
                        'text/plain': text_repr
                    },
                    execution_count=1
                )
            ]
        cells.append(step4_cell)

        nb['cells'] = cells
        return nb




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
            return True  # If user never specified, no need to block
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
        def sanitize(obj):
            if isinstance(obj, dict):
                for k in obj:
                    obj[k] = sanitize(obj[k])
                return obj
            elif isinstance(obj, list):
                return [sanitize(v) for v in obj]
            elif isinstance(obj, float):
                if np.isnan(obj) or np.isinf(obj):
                    return None
                else:
                    return obj
            else:
                return obj

        sanitize(nb)
        return nb

    def create_entity_target_notebook(self, entity_id_column, target_column, table_name, columns_list):
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

        print("[DEBUG] Creating entity-target notebook...")
        nb = new_notebook()
        cells = []
        cells.append(new_markdown_cell("Core Set"))

        sanitized_entity_id_column = self.sanitize_identifier(entity_id_column) if entity_id_column else "*"
        sanitized_target_column = self.sanitize_identifier(target_column) if target_column else "*"
        sql_query_entity_target = (
            f"SELECT {sanitized_entity_id_column}, {sanitized_target_column} "
            f"FROM {table_name} LIMIT 10;"
        )
        # # df_result = execute_sql_query(sql_query_entity_target)
        # if df_result.empty:
        #     error_message = f"No data returned for query: {sql_query_entity_target}"
        #     cells.append(new_markdown_cell(f"**Error:** {error_message}"))
        # else:
        #     df_result = df_result.replace([np.nan, np.inf, -np.inf], None)
        #     result_json = df_result.to_dict(orient='records')
        #     columns = []
        #     for col in df_result.columns:
        #         non_null_series = pd.Series([x for x in df_result[col] if x is not None])
        #         if non_null_series.empty:
        #             col_type = "string"
        #         else:
        #             col_type = infer_column_dtype(non_null_series)
        #         columns.append({'name': col, 'type': col_type})

        #     text_repr = df_result.head().to_string(index=False)
        #     code_cell = new_code_cell(sql_query_entity_target)
        #     code_cell['execution_count'] = 1
        #     code_cell.outputs = [
        #         new_output(
        #             output_type='execute_result',
        #             data={
        #                 'application/json': {
        #                     'rows': result_json,
        #                     'columns': columns
        #                 },
        #                 'text/plain': text_repr
        #             },
        #             metadata={},
        #             execution_count=1
        #         )
        #     ]
        #     cells.append(code_cell)

        nb['cells'] = cells
        return nb

    def create_features_notebook(self, feature_columns, table_name, columns_list):
        import nbformat
        from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

        print("[DEBUG] Creating features notebook...")
        nb = new_notebook()
        cells = []
        cells.append(new_markdown_cell("Features or Attributes Test"))

        sanitized_features = [self.sanitize_identifier(feature) for feature in feature_columns]
        missing_columns = [feature for feature in sanitized_features if feature not in columns_list]
        if missing_columns:
            error_message = f"The following feature columns do not exist in the dataset: {', '.join(missing_columns)}"
            cells.append(new_markdown_cell(f"**Error:** {error_message}"))
            nb['cells'] = cells
            return nb

        feature_query = (
            f"SELECT\n    " + ",\n    ".join(sanitized_features) +
            f"\nFROM {table_name}\nLIMIT 10;"
        )

        # df_result = execute_sql_query(feature_query)
        # if df_result.empty:
        #     error_message = f"No data returned for query: {feature_query}"
        #     cells.append(new_markdown_cell(f"**Error:** {error_message}"))
        # else:
        #     df_result = df_result.replace([np.nan, np.inf, -np.inf], None)
        #     result_json = df_result.to_dict(orient='records')
        #     columns = []
        #     for col in df_result.columns:
        #         non_null_series = pd.Series([x for x in df_result[col] if x is not None])
        #         if non_null_series.empty:
        #             col_type = "string"
        #         else:
        #             col_type = infer_column_dtype(non_null_series)
        #         columns.append({'name': col, 'type': col_type})

            # text_repr = df_result.head().to_string(index=False)
            # code_cell = new_code_cell(feature_query)
            # code_cell['execution_count'] = 1
            # code_cell.outputs = [
            #     new_output(
            #         output_type='execute_result',
            #         data={
            #             'application/json': {
            #                 'rows': result_json,
            #                 'columns': columns
            #             },
            #             'text/plain': text_repr
            #         },
            #         metadata={},
            #         execution_count=1
                # )
            # ]
            # cells.append(code_cell)

        nb['cells'] = cells
        return nb

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
                # df = execute_sql_query(query)
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





















# for getting sql queries data



from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class NotebookView(APIView):

    def get(self, request):
        user_id = request.query_params.get("user_id")
        chat_id = request.query_params.get("chat_id")

        if not user_id or not chat_id:
            return Response({"error": "user_id and chat_id are required."}, status=status.HTTP_400_BAD_REQUEST)

        try:
            # notebooks = Notebook.objects.filter(user_id=user_id, chat__chat_id=chat_id)
            notebooks = Notebook.objects.filter(chat    =chat_id, user_id=user_id)


            if not notebooks.exists():
                return Response({"error": "No notebooks found for the given user_id and chat_id."}, status=status.HTTP_404_NOT_FOUND)

            # Serialize the data
            notebook_data = [
                {
                    "id": notebook.id,
                    "entity_column": notebook.entity_column,
                    "target_column": notebook.target_column,
                    "time_column": notebook.time_column,
                    "time_frame": notebook.time_frame,
                    "time_frequency": notebook.time_frequency,
                    "features": notebook.features,
                    "file_url": notebook.file_url,
                    "notebook_json": notebook.notebook_json,
                    "created_at": notebook.created_at,
                }
                for notebook in notebooks
            ]

            return Response({"message": "Notebooks retrieved successfully.", "notebooks": notebook_data}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)