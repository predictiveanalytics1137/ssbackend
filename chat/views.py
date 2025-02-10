




# import os
# import uuid
# import datetime
# import re
# import boto3
# import nbformat
# import pandas as pd
# import openai
# import requests
# import numpy as np
# from io import BytesIO
# from typing import Any, Dict, List
# from django.db import transaction
# from django.contrib.auth.models import User
# from django.core.exceptions import ValidationError
# from rest_framework import status
# from rest_framework.parsers import FormParser, MultiPartParser, JSONParser
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from rest_framework.permissions import IsAuthenticated
# from botocore.exceptions import ClientError, NoCredentialsError
# from django.conf import settings
# from sqlalchemy import create_engine
# from langchain.chains import ConversationChain
# from langchain_community.chat_models import ChatOpenAI
# from langchain.prompts import PromptTemplate
# from langchain.memory import ConversationBufferMemory
# from langchain.schema import AIMessage, HumanMessage

# from chat.utils import classify_ml_type, parse_nlu_input

# from .models import FileSchema, Notebook, PredictiveSettings, UploadedFile, ChatBackup
# from .serializers import UploadedFileSerializer

# # AWS and OpenAI environment variables
# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')
# AWS_REGION_NAME = AWS_S3_REGION_NAME
# ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Adjust as needed
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# openai.api_key = OPENAI_API_KEY


# from langchain.prompts.chat import (
#     ChatPromptTemplate,
#     SystemMessagePromptTemplate,
#     HumanMessagePromptTemplate,
# )

# # Initialize the ChatOpenAI model
# llm_chatgpt = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
#     openai_api_key=OPENAI_API_KEY,
# )

# # In-memory storage for user-specific data
# user_conversations = {}
# user_schemas = {}
# user_confirmations = {}
# user_notebook_flags = {}
# user_notebooks = {}


# SYSTEM_INSTRUCTIONS = (
#     "You are a highly intelligent and helpful PACX AI assistant. Your responses should be clear and concise. "
#     "Assist the user in forming the predictive question and any corrections they provide. Reflect the confirmed or "
#     "corrected schema back to the user before proceeding, asking one question at a time.\n\n"
#     "Steps:\n"
#     "1. Discuss the subject they want to predict.\n"
#     "2. Confirm the target value they want to predict.\n"
#     "3. Check if there's a specific time frame for the prediction.\n"
#     "4. Determine whether the prediction should occur on a recurring basis (e.g., daily, weekly, monthly) or after a specific event.\n"
#     "5. Reference the dataset schema if available; if not, ask to upload it.\n"
#     "6. Once all necessary information is confirmed, provide a summary and let the user know they can generate the notebook."
# )

# # # Prompt template for AI conversation
# # SYSTEM_INSTRUCTIONS = (
# #     "You are a highly intelligent and helpful AI assistant. Your responses should be clear and concise. "
# #     "Assist the user in forming a predictive question and any corrections they provide. "
# # )

# chat_prompt = ChatPromptTemplate.from_messages([
#     SystemMessagePromptTemplate.from_template(SYSTEM_INSTRUCTIONS),
#     HumanMessagePromptTemplate.from_template(
#         "Conversation so far:\n{history}\nUser input:\n{user_input}"
#     )
# ])

# def get_s3_client():
#     print("[DEBUG] Creating S3 client...")
#     return boto3.client(
#         's3',
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_S3_REGION_NAME
#     )

# def get_glue_client():
#     print("[DEBUG] Creating Glue client...")
#     return boto3.client(
#         'glue',
#         aws_access_key_id=AWS_ACCESS_KEY_ID,
#         aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
#         region_name=AWS_S3_REGION_NAME
#     )

# def execute_sql_query(query: str) -> pd.DataFrame:
#     print("[DEBUG] Executing Athena query:", query)
#     try:
#         if not AWS_ATHENA_S3_STAGING_DIR:
#             raise ValueError("AWS_ATHENA_S3_STAGING_DIR is not set.")
#         connection_string = (
#             f"awsathena+rest://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}"
#             f"@athena.{AWS_REGION_NAME}.amazonaws.com:443/{ATHENA_SCHEMA_NAME}"
#             f"?s3_staging_dir={AWS_ATHENA_S3_STAGING_DIR}&catalog_name=AwsDataCatalog"
#         )
#         print("[DEBUG] Athena connection string created.")
#         engine = create_engine(connection_string)
#         df = pd.read_sql_query(query, engine)
#         print(f"[DEBUG] Query executed successfully. Rows returned: {len(df)}")
#         return df
#     except Exception as e:
#         print(f"[ERROR] Failed to execute query: {query}, Error: {str(e)}")
#         return pd.DataFrame()

# import re

# def normalize_column_name(col_name: str) -> str:
#     # Convert to lowercase and trim whitespace
#     normalized = col_name.strip().lower()
#     # Replace spaces with underscores
#     normalized = normalized.replace(' ', '_')
#     # Remove all special characters except underscores
#     normalized = re.sub(r'[^a-z0-9_]', '', normalized)
#     # Ensure name starts with letter/underscore (valid Python variable)
#     if normalized and normalized[0].isdigit():
#         normalized = f'_{normalized}'
#     return normalized

# def infer_column_dtype(series: pd.Series, threshold: float = 0.8) -> str:
#     """
#     Attempt to infer the most likely data type of a pandas Series.
#     This uses 'errors=coerce' for date parsing + threshold-based classification.
#     """
#     series = series.dropna().astype(str).str.strip()
#     if series.empty:
#         return "string"

#     # Attempt #1: flexible parse toggling dayfirst
#     for dayfirst_val in [False, True]:
#         dt_series = pd.to_datetime(series, dayfirst=dayfirst_val, errors='coerce')
#         valid_count = dt_series.notnull().sum()
#         total_count = len(series)
#         if total_count > 0 and (valid_count / total_count) >= threshold:
#             return "timestamp"

#     # Attempt #2: known date formats
#     known_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y"]
#     for fmt in known_formats:
#         dt_series = pd.to_datetime(series, format=fmt, errors='coerce')
#         valid_count = dt_series.notnull().sum()
#         total_count = len(series)
#         if total_count > 0 and (valid_count / total_count) >= threshold:
#             return "timestamp"

#     # Boolean check
#     boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}
#     unique_values = set(series.str.lower().unique())
#     if unique_values.issubset(boolean_values):
#         return "boolean"

#     # Integer check
#     try:
#         int_series = pd.to_numeric(series, errors='raise')
#         if (int_series % 1 == 0).all():
#             int_min = int_series.min()
#             int_max = int_series.max()
#             if int_min >= -2147483648 and int_max <= 2147483647:
#                 return "int"
#             else:
#                 return "bigint"
#     except ValueError:
#         pass

#     # Double check
#     try:
#         pd.to_numeric(series, errors='raise', downcast='float')
#         return "double"
#     except ValueError:
#         pass

#     return "string"

# def standardize_datetime_columns(df: pd.DataFrame, schema: list) -> pd.DataFrame:
#     """
#     For columns recognized as 'timestamp', convert them to a standard ISO datetime string.
#     """
#     for colinfo in schema:
#         if colinfo["data_type"] == "timestamp":
#             col_name = colinfo["column_name"]
#             try:
#                 df[col_name] = pd.to_datetime(df[col_name], errors="coerce")
#                 df[col_name] = df[col_name].dt.strftime("%Y-%m-%d %H:%M:%S")
#                 print(f"[DEBUG] Standardized datetime column: {col_name}")
#             except Exception as e:
#                 print(f"[WARNING] Could not standardize date column '{col_name}': {str(e)}")
#     return df

# def suggest_target_column(df: pd.DataFrame, chat_history: List[Any]) -> Any:
#     return df.columns[-1]

# def suggest_entity_id_column(df: pd.DataFrame) -> Any:
#     likely_id_columns = [col for col in df.columns if "id" in col.lower()]
#     for col in likely_id_columns:
#         if df[col].nunique() / len(df) > 0.95:
#             return col
#     for col in df.columns:
#         if df[col].nunique() / len(df) > 0.95:
#             return col
#     return None

# def parse_user_adjustments(user_input, uploaded_file_info):
#     print("[DEBUG] Parsing user adjustments...")
#     columns_list = [col['column_name'] for col in uploaded_file_info['schema']]
#     normalized_columns = [normalize_column_name(c) for c in columns_list]
#     user_input = user_input.strip().lower()

#     pattern = re.compile(
#         r'(target|entity\s*id|time\s*column|time\s*frame|time\s*freq)\D*(\b\w+\b)',
#         re.IGNORECASE
#     )
    
#     adjustments = {}
#     matches = pattern.findall(user_input)
#     for key_phrase, value in matches:
#         key = key_phrase.lower().strip()
#         val_norm = normalize_column_name(value)
        
#         if 'target' in key:
#             if val_norm in normalized_columns:
#                 match_col = columns_list[normalized_columns.index(val_norm)]
#                 adjustments['target_column'] = match_col
#         elif 'entity' in key and 'id' in key:
#             if val_norm in normalized_columns:
#                 match_col = columns_list[normalized_columns.index(val_norm)]
#                 adjustments['entity_id_column'] = match_col
#         elif 'time' in key and 'column' in key:
#             if val_norm in normalized_columns:
#                 adjustments['time_column'] = columns_list[normalized_columns.index(val_norm)]
#         elif 'time' in key:
#             if 'frame' in key:
#                 adjustments['time_frame'] = value
#             elif 'freq' in key:
#                 adjustments['time_frequency'] = value

#     return adjustments or None

# class UnifiedChatGPTAPI(APIView):
#     parser_classes = [MultiPartParser, FormParser, JSONParser]

#     def post(self, request):
#         action = request.data.get('action', '')
#         if action == 'reset':
#             return self.reset_conversation(request)
#         if action == 'generate_notebook':
#             return self.generate_notebook(request)
#         if "file" in request.FILES:
#             return self.handle_file_upload(request, request.FILES.getlist("file"))
#         return self.handle_chat(request)
    


#     def handle_file_upload(self, request, files):
#         """
#         1) Ingests CSV/XLSX, normalizes columns
#         2) Infers schema
#         3) Uploads to S3
#         4) Creates or updates table in Glue
#         5) Returns 'uploaded_files_info' including suggestions
#         """
#         user_id = request.data.get("user_id", "default_user")
#         s3 = get_s3_client()
#         glue = get_glue_client()
#         uploaded_files_info = []

#         for file in files:
#             print(f"[DEBUG] Processing file: {file.name}")
#             try:
#                 if file.name.lower().endswith('.csv'):
#                     df = pd.read_csv(
#                         file,
#                         low_memory=False,
#                         encoding='utf-8',
#                         delimiter=',',
#                         na_values=['NA', 'N/A', ''],
#                         on_bad_lines='warn'
#                     )
#                 else:
#                     df = pd.read_excel(file, engine='openpyxl')

#                 if df.empty:
#                     print("[ERROR] File is empty:", file.name)
#                     return Response({"error": f"Uploaded file {file.name} is empty."},
#                                     status=status.HTTP_400_BAD_REQUEST)
#                 if not df.columns.any():
#                     print("[ERROR] File has no columns:", file.name)
#                     return Response({"error": f"Uploaded file {file.name} has no columns."},
#                                     status=status.HTTP_400_BAD_REQUEST)
#             except pd.errors.ParserError as e:
#                 print("[ERROR] CSV parsing error:", e)
#                 return Response({"error": f"CSV parsing error for file {file.name}: {str(e)}"},
#                                 status=status.HTTP_400_BAD_REQUEST)
#             except Exception as e:
#                 print("[ERROR] Error reading file:", e)
#                 return Response({"error": f"Error reading file {file.name}: {str(e)}"},
#                                 status=status.HTTP_400_BAD_REQUEST)

#             # Normalize column names
#             normalized_columns = [normalize_column_name(c) for c in df.columns]
#             if len(normalized_columns) != len(set(normalized_columns)):
#                 print("[ERROR] Duplicate columns after normalization.")
#                 return Response({"error": "Duplicate columns detected after normalization."},
#                                 status=status.HTTP_400_BAD_REQUEST)
#             if any(col == '' for col in normalized_columns):
#                 print("[ERROR] Empty column names after normalization.")
#                 return Response({"error": "Some columns have empty names after normalization."},
#                                 status=status.HTTP_400_BAD_REQUEST)

#             df.columns = normalized_columns

#             # Infer schema
#             raw_schema = [
#                 {"column_name": col, "data_type": infer_column_dtype(df[col])}
#                 for col in df.columns
#             ]
#             df = standardize_datetime_columns(df, raw_schema)
#             final_schema = [
#                 {"column_name": col, "data_type": infer_column_dtype(df[col])}
#                 for col in df.columns
#             ]

#             has_date_column = any(c["data_type"] == "timestamp" for c in final_schema)
#             possible_date_cols = [
#                 c["column_name"] for c in final_schema if c["data_type"] == "timestamp"
#             ]

#             # Fix boolean columns
#             boolean_columns = [
#                 c['column_name'] for c in final_schema if c['data_type'] == 'boolean'
#             ]
#             replacement_dict = {
#                 '1': 'true','0': 'false',
#                 'yes':'true','no':'false',
#                 't':'true','f':'false',
#                 'y':'true','n':'false',
#                 'true':'true','false':'false',
#             }
#             for col_name in boolean_columns:
#                 df[col_name] = (
#                     df[col_name].astype(str).str.strip().str.lower().replace(replacement_dict)
#                 )
#                 unexpected_values = [v for v in df[col_name].unique() if v not in ['true','false']]
#                 if unexpected_values:
#                     print("[ERROR] Unexpected boolean values:", unexpected_values)
#                     return Response(
#                         {"error": f"Unexpected boolean values in column {col_name}: {unexpected_values}"},
#                         status=status.HTTP_400_BAD_REQUEST
#                     )

#             # Build unique file key for S3
#             file_name_base, file_extension = os.path.splitext(file.name)
#             file_name_base = file_name_base.lower().replace(' ', '_')
#             unique_id = uuid.uuid4().hex[:8]
#             new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
#             s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
#             file_key = f"uploads/{unique_id}/{s3_file_name}"
#             print("[DEBUG] Uploading file to S3 at key:", file_key)

#             try:
#                 with transaction.atomic():
#                     # Save the file record in Django
#                     file.seek(0)
#                     file_serializer = UploadedFileSerializer(data={'name': new_file_name, 'file': file})
#                     if file_serializer.is_valid():
#                         file_instance = file_serializer.save()
#                         # Re-upload as CSV to S3
#                         csv_buffer = BytesIO()
#                         df.to_csv(csv_buffer, index=False, encoding='utf-8')
#                         csv_buffer.seek(0)
#                         s3.upload_fileobj(csv_buffer, os.getenv('AWS_STORAGE_BUCKET_NAME'), file_key)
#                         print("[DEBUG] S3 upload successful:", file_key)
#                         s3.head_object(
#                             Bucket=os.getenv('AWS_STORAGE_BUCKET_NAME'),
#                             Key=file_key
#                         )
#                         file_url = f"s3://{os.getenv('AWS_STORAGE_BUCKET_NAME')}/{file_key}"
#                         file_instance.file_url = file_url
#                         file_instance.save()

#                         # Store the schema in DB
#                         FileSchema.objects.create(file=file_instance, schema=final_schema)

#                         file_size_mb = file.size / (1024 * 1024)
#                         self.trigger_glue_update(new_file_name, final_schema, file_key, file_size_mb)

#                         # Build suggestions
#                         target_suggestion = suggest_target_column(df, [])
#                         entity_suggestion = suggest_entity_id_column(df)
#                         feature_candidates = [
#                             c for c in df.columns
#                             if c not in [entity_suggestion, target_suggestion]
#                         ]

#                         uploaded_files_info.append({
#                             'id': file_instance.id,
#                             'name': file_instance.name,
#                             'file_url': file_instance.file_url,
#                             'schema': final_schema,
#                             'file_size_mb': file_size_mb,
#                             'has_date_column': has_date_column,
#                             'date_columns': possible_date_cols,
#                             'suggestions': {
#                                 'target_column': target_suggestion,
#                                 'entity_id_column': entity_suggestion,
#                                 'feature_columns': feature_candidates
#                             }
#                         })
#                     else:
#                         print("[ERROR] File serializer errors:", file_serializer.errors)
#                         return Response(file_serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#             except ClientError as e:
#                 print("[ERROR] AWS ClientError:", e)
#                 return Response({'error': f'AWS error: {str(e)}'},
#                                 status=status.HTTP_500_INTERNAL_SERVER_ERROR)
#             except Exception as e:
#                 print("[ERROR] Unexpected error during file processing:", e)
#                 return Response({'error': f'File processing failed: {str(e)}'},
#                                 status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#         # Once done, return info about the uploaded files
#         # (You might store this in the DB if you like, but your code keeps it in memory.)
#         # This is the existing approach
#         user_schemas[user_id] = uploaded_files_info  # (In practice you might remove this if fully DB-based)

#         chat_id = request.data.get("chat_id", "")
#         memory_key = f"{user_id}_{chat_id}" if chat_id else user_id

#         # Create or reuse a conversation chain
#         if memory_key not in user_conversations:
#             conversation_chain = ConversationChain(
#                 llm=llm_chatgpt,
#                 prompt=chat_prompt,
#                 input_key="user_input",
#                 memory=ConversationBufferMemory()
#             )
#             user_conversations[memory_key] = conversation_chain
#         else:
#             conversation_chain = user_conversations[memory_key]

#         # Insert a message about the newly discovered schema
#         schema_discussion = self.format_schema_message(uploaded_files_info[0])
#         conversation_chain.memory.chat_memory.messages.append(
#             AIMessage(content=schema_discussion)
#         )

#         print("[DEBUG] Files uploaded and schema discussion initiated.")
#         return Response({
#             "message": "Files uploaded and processed successfully.",
#             "uploaded_files": uploaded_files_info,
#             "chat_message": schema_discussion
#         }, status=status.HTTP_201_CREATED)



# # ok working
#     def handle_chat(self, request):
#         user_input = request.data.get("message", "").strip()
#         user_id = request.data.get("user_id", "default_user")
#         chat_id = request.data.get("chat_id")

#         if not user_input:
#             return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

#         # Validate user
#         try:
#             user = User.objects.get(id=user_id)
#         except User.DoesNotExist:
#             return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

#         # If no chat_id, create a new chat
#         if not chat_id:
#             chat_id = str(uuid.uuid4())
#             chat_title = user_input[:50]
#             ChatBackup.objects.create(
#                 user=user,
#                 chat_id=chat_id,
#                 title=chat_title,
#                 messages=[]
#             )

#         chat_obj, _ = ChatBackup.objects.get_or_create(
#             user=user,
#             chat_id=chat_id,
#             defaults={"title": user_input[:50], "messages": []}
#         )

#         ############################################
#         # 1) parse the user_input
#         ############################################
#         schema_columns = []
#         if user_id in user_schemas and user_schemas[user_id]:
#             schema_columns = [col["column_name"] for col in user_schemas[user_id][0]["schema"]]

#         system_prompt = "Extract any user instructions about target/entity/time_frame/time_column/time_frequency."
#         parsed_updates = parse_nlu_input(system_prompt, user_input, schema_columns)

#         ############################################
#         # 2) fetch or create PredictiveSettings
#         ############################################
#         ps, _ = PredictiveSettings.objects.get_or_create(user=user, chat_id=chat_id)

#         # For each recognized key in parsed_updates, update only that field
#         if "predictive_question" in parsed_updates:
#             ps.predictive_question = parsed_updates["predictive_question"]
#         if "target_column" in parsed_updates:
#             ps.target_column = parsed_updates["target_column"]
#         if "entity_column" in parsed_updates:
#             ps.entity_column = parsed_updates["entity_column"]
#         if "time_frame" in parsed_updates:
#             ps.time_frame = parsed_updates["time_frame"]
#         if "time_column" in parsed_updates:
#             ps.time_column = parsed_updates["time_column"]
#         if "time_frequency" in parsed_updates:
#             ps.time_frequency = parsed_updates["time_frequency"]

#         # 2b) Optionally, auto-set machine_learning_type
#         # For instance, if the user says "churn" => classification
#         # You can do a simple heuristic check on the user_input or the stored question
#         if ps.predictive_question:
#             lower_question = ps.predictive_question.lower()
#             if "churn" in lower_question or "yes/no" in lower_question:
#                 ps.machine_learning_type = "classification"
#             else:
#                 # for demonstration, assume regression otherwise
#                 ps.machine_learning_type = "regression"

#         # If user_input itself says "I want to predict churn next month," you might parse that as well
#         if "churn" in user_input.lower():
#             ps.machine_learning_type = "classification"

#         # Save the partial updates
#         ps.save()

#         ############################################
#         # 3) Re-hydrate LLM chain, run conversation
#         ############################################
#         memory_key = f"{user_id}_{chat_id}"
#         if memory_key not in user_conversations:
#             restored_memory = ConversationBufferMemory(return_messages=True)
#             for msg in chat_obj.messages:
#                 if msg["sender"] == "assistant":
#                     restored_memory.chat_memory.add_message(AIMessage(content=msg["text"]))
#                 else:
#                     restored_memory.chat_memory.add_message(HumanMessage(content=msg["text"]))

#             user_conversations[memory_key] = ConversationChain(
#                 llm=llm_chatgpt,
#                 prompt=chat_prompt,
#                 input_key="user_input",
#                 memory=restored_memory
#             )

#         conversation_chain = user_conversations[memory_key]
#         assistant_response = conversation_chain.run(user_input=user_input)

#         ############################################
#         # 4) Save messages to ChatBackup
#         ############################################
#         timestamp = datetime.datetime.now().isoformat()
#         chat_obj.messages.append({"sender": "user", "text": user_input, "timestamp": timestamp})
#         chat_obj.messages.append({"sender": "assistant", "text": assistant_response, "timestamp": timestamp})
#         chat_obj.save()

#         ############################################
#         # 5) Decide if show_generate_notebook
#         ############################################
#         show_generate_notebook = False
#         # If we have both target_column & entity_column, we show the button
#         if ps.target_column and ps.entity_column:
#             show_generate_notebook = True

#         return Response({
#             "response": assistant_response,
#             "chat_id": chat_id,
#             "show_generate_notebook": show_generate_notebook
#         })





    




    
    
    
#     def process_schema_confirmation(self, user_input, user_id):
#         print("[DEBUG] Processing schema confirmation for user:", user_id)
#         if user_id not in user_schemas or not user_schemas[user_id]:
#             return ("", False)

#         uploaded_file_info = user_schemas[user_id][0]
#         suggestions = uploaded_file_info['suggestions']
#         has_date_column = uploaded_file_info.get('has_date_column', False)
#         date_cols = uploaded_file_info.get('date_columns', [])

#         # If no confirmation record yet, initialize
#         if user_id not in user_confirmations:
#             user_confirmations[user_id] = {
#                 'entity_id_column': suggestions['entity_id_column'],
#                 'target_column': suggestions['target_column'],
#                 'feature_columns': [{'column_name': col} for col in suggestions['feature_columns']],
#                 'time_column': date_cols[0] if (has_date_column and len(date_cols) == 1) else None,
#                 'time_frame': None,
#                 'time_frequency': None,
#                 'step': 1,
#                 'all_confirmed': False  # [ADDED for final confirmation fix]
#             }

#         conf = user_confirmations[user_id]

#         # [ADDED for final confirmation fix]: If we've already got all_confirmed, do nothing
#         if conf.get('all_confirmed'):
#             # Already finalized. Return an empty string or a short reminder.
#             # We'll treat it as "confirmed" so handle_chat won't do chain
#             return ("All columns are already confirmed. You can generate the notebook now!", True)

#         # Check user input => "yes" or corrections
#         if user_input.lower() == 'yes':
#             if conf.get('step', 1) == 1:
#                 conf['step'] = 2
#                 user_confirmations[user_id] = conf
#                 return (
#                     f"Great! You've confirmed the target column as '{conf['target_column']}'.\n"
#                     f"Next, please confirm the Entity ID Column. Suggested: '{conf['entity_id_column']}'.",
#                     True
#                 )
#             elif conf.get('step') == 2:
#                 conf['step'] = 3
#                 user_confirmations[user_id] = conf
#                 return (
#                     f"Great! You've confirmed the Entity ID Column as '{conf['entity_id_column']}'.\n"
#                     f"Next, please confirm the Feature Columns: "
#                     f"{', '.join([col['column_name'] for col in conf.get('feature_columns', [])])}.",
#                     True
#                 )
#             elif conf.get('step') == 3:
#                 # [ADDED for final confirmation fix] => Mark all_confirmed = True and build final summary
#                 conf['all_confirmed'] = True
#                 final_msg = self.build_final_confirmation_msg(conf, has_date_column)
#                 user_confirmations[user_id] = conf
#                 return (final_msg, True)

#         # If user typed corrections
#         adjusted_columns = parse_user_adjustments(user_input, uploaded_file_info)
#         if adjusted_columns:
#             existing_conf = user_confirmations.get(user_id, {})
#             # Overwrite or create fields
#             if 'target_column' in adjusted_columns:
#                 existing_conf['target_column'] = adjusted_columns['target_column']
#                 schema_cols = [
#                     normalize_column_name(col_info['column_name'])
#                     for col_info in uploaded_file_info['schema']
#                 ]
#                 if normalize_column_name(existing_conf['target_column']) not in schema_cols:
#                     response = (
#                         f"Your provided target column '{existing_conf['target_column']}' is not in the dataset. "
#                         f"Suggested target column is '{suggestions['target_column']}'. Please confirm."
#                     )
#                     existing_conf['target_column'] = suggestions['target_column']
#                     user_confirmations[user_id] = existing_conf
#                     return (response, True)
#                 else:
#                     existing_conf['step'] = 2

#             if 'entity_id_column' in adjusted_columns:
#                 existing_conf['entity_id_column'] = adjusted_columns['entity_id_column']
#                 existing_conf['step'] = 3

#             if 'time_column' in adjusted_columns:
#                 if adjusted_columns['time_column'] in date_cols:
#                     existing_conf['time_column'] = adjusted_columns['time_column']
#                 else:
#                     print("[DEBUG] Provided time column is not recognized among:", date_cols)

#             if 'time_frame' in adjusted_columns:
#                 existing_conf['time_frame'] = adjusted_columns['time_frame']
#             if 'time_frequency' in adjusted_columns:
#                 existing_conf['time_frequency'] = adjusted_columns['time_frequency']

#             user_confirmations[user_id] = existing_conf

#             # Now see if everything is complete => if so, set all_confirmed True
#             if existing_conf['step'] >= 3:
#                 # Check if user truly has all columns set (time_column only if needed)
#                 # We'll consider "has_date_column => require time_column"
#                 if existing_conf.get('target_column') and existing_conf.get('entity_id_column'):
#                     if not has_date_column or existing_conf.get('time_column'):
#                         existing_conf['all_confirmed'] = True
#                         final_msg = self.build_final_confirmation_msg(existing_conf, has_date_column)
#                         user_confirmations[user_id] = existing_conf
#                         return (final_msg, True)

#             # Otherwise, partial summary
#             entity_id_col = existing_conf['entity_id_column']
#             target_col = existing_conf['target_column']
#             feature_names = [obj['column_name'] for obj in existing_conf['feature_columns']]
#             time_col = existing_conf.get('time_column', None)
#             time_frame = existing_conf.get('time_frame', None)
#             time_freq = existing_conf.get('time_frequency', None)

#             partial_msg = (
#                 f"Thanks for the corrections! The updated schema is now:\n\n"
#                 f"- Entity ID Column: {entity_id_col}\n"
#                 f"- Target Column: {target_col}\n"
#                 f"- Feature Columns: {', '.join(feature_names) if feature_names else 'None'}\n"
#             )
#             if has_date_column:
#                 if time_col:
#                     partial_msg += f"- Time Column: {time_col}\n"
#                     if time_frame:
#                         partial_msg += f"- Time Frame: {time_frame}\n"
#                     if time_freq:
#                         partial_msg += f"- Time Frequency: {time_freq}\n"
#                     if time_frame:
#                         partial_msg += "\nTime-based approach confirmed. You can proceed to generate the notebook."
#                     else:
#                         partial_msg += (
#                             "\nWe have the date column set, but please specify Time Frame, e.g. 'Time Frame: 3 MONTHS'."
#                         )
#                 else:
#                     partial_msg += (
#                         "\nWe detected a date column, but you haven't specified which to use yet.\n"
#                         f"Available date columns: {date_cols}\n"
#                         "Please provide it like 'Time Column: <column_name>'."
#                     )
#             else:
#                 partial_msg += "\nNo date column found; proceeding with a non-time-based approach."
#             return (partial_msg, True)

#         # If nothing changed and not a "yes", just return no message
#         return ("", False)

#     # [ADDED for final confirmation fix]: Helper to build final summary
#     def build_final_confirmation_msg(self, conf, has_date_column: bool):
#         """
#         Once all columns are known or step=3 is done, produce final summary
#         so the user can generate the notebook.
#         """
#         entity_id_col = conf.get('entity_id_column')
#         target_col = conf.get('target_column')
#         feature_cols = [obj['column_name'] for obj in conf.get('feature_columns', [])]
#         time_col = conf.get('time_column')
#         time_frame = conf.get('time_frame')
#         time_freq = conf.get('time_frequency')

#         final_msg = (
#             "Great! Here’s the final summary of your prediction setup:\n\n"
#             f"- **Entity ID Column:** {entity_id_col}\n"
#             f"- **Target Column:** {target_col}\n"
#         )
#         if has_date_column:
#             final_msg += f"- **Time Column:** {time_col if time_col else '(none)'}\n"
#             final_msg += f"- **Time Frame:** {time_frame if time_frame else '(none)'}\n"
#             final_msg += f"- **Time Frequency:** {time_freq if time_freq else '(none)'}\n"
#         if feature_cols:
#             final_msg += f"- **Feature Columns:** {', '.join(feature_cols)}\n\n"
#         else:
#             final_msg += "- **Feature Columns:** (none)\n\n"

#         final_msg += (
#             "You can now generate the notebook with these final settings. "
#             "If you need any further adjustments, feel free to let me know!"
#         )
#         return final_msg
    

#     def reset_conversation(self, request):
#         """
#         Clears in-memory conversation. If you also want to remove rows from PredictiveSettings
#         or ChatBackup, do that here.
#         """
#         user_id = request.data.get("user_id", "default_user")
#         if user_id in user_conversations:
#             del user_conversations[user_id]
#         if user_id in user_schemas:
#             del user_schemas[user_id]
#         if user_id in user_notebooks:
#             del user_notebooks[user_id]
#         # Optional: Clear from DB
#         PredictiveSettings.objects.filter(user_id=user_id).delete()
#         ChatBackup.objects.filter(user_id=user_id).delete()

#         return Response({"message": "Conversation reset successful."})



#     def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
#         schema = uploaded_file['schema']
#         target_column = uploaded_file['suggestions']['target_column']
#         entity_id_column = uploaded_file['suggestions']['entity_id_column']
#         feature_columns = uploaded_file['suggestions']['feature_columns']
#         has_date_column = uploaded_file.get('has_date_column', False)
#         date_cols = uploaded_file.get('date_columns', [])

#         schema_text = (
#             f"Dataset '{uploaded_file['name']}' uploaded successfully!\n\n"
#             "Columns:\n" + ", ".join([col['column_name'] for col in schema]) + "\n\n"
#             "Data Types:\n" + "\n".join([f"{col['column_name']}: {col['data_type']}" for col in schema]) + "\n\n"
#             f"Suggested Target Column: {target_column or 'None'}\n"
#             f"Suggested Entity ID Column: {entity_id_column or 'None'}\n"
#             f"Suggested Feature Columns: {', '.join(feature_columns) if feature_columns else 'None'}\n\n"
#         )
#         if has_date_column:
#             if len(date_cols) == 1:
#                 schema_text += (
#                     f"We detected a single date column: {date_cols[0]}.\n"
#                     "We'll use it for time-based modeling unless you specify otherwise.\n"
#                     "You can also specify a 'Time Frame: <X>' (e.g., 'Time Frame: 1 WEEK') or 'Time Frequency: <daily|weekly|monthly>'.\n\n"
#                 )
#             elif len(date_cols) > 1:
#                 schema_text += (
#                     "We detected multiple date columns. Please specify which one to use as the time column:\n"
#                     f"{date_cols}\n\n"
#                     "And also specify 'Time Frame: <X>' or 'Time Frequency: <daily|weekly|monthly>' if you’d like a time-based approach.\n"
#                 )
#         else:
#             schema_text += (
#                 "No date column detected, so by default we'll proceed with a non-time-based approach.\n\n"
#             )

#         schema_text += (
#             "Please confirm:\n"
#             "- Is the Target Column correct?\n"
#             "- Is the Entity ID Column correct?\n"
#             "- If a date column is detected, specify 'Time Column: <column>' if you want a time-based approach.\n"
#             "- Optionally specify 'Time Frame: <X>' (e.g. 'Time Frame: 2 WEEKS') and 'Time Frequency: <daily|weekly|monthly>'.\n\n"
#             "(Reply 'yes' to confirm or provide corrections in the format:\n"
#             "'Entity ID Column: <column>, Target Column: <column>, Time Column: <column>, Time Frame: <X>, Time Frequency: <Y>')"
#         )
#         return schema_text
    




#     def generate_notebook(self, request):
#         """
#         Reads final columns from PredictiveSettings. If we have a date column, do time-based approach; otherwise not.
#         """
#         user_id = request.data.get("user_id")
#         chat_id = request.data.get("chat_id")

#         if not user_id or not chat_id:
#             return Response({"error": "user_id and chat_id are required."},
#                             status=status.HTTP_400_BAD_REQUEST)

#         # Validate user
#         try:
#             user = User.objects.get(id=user_id)
#         except User.DoesNotExist:
#             return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

#         # Validate chat
#         try:
#             ChatBackup.objects.get(user=user, chat_id=chat_id)
#         except ChatBackup.DoesNotExist:
#             return Response({"error": "Chat not found."}, status=status.HTTP_404_NOT_FOUND)

#         # Pull from PredictiveSettings
#         try:
#             settings_obj = PredictiveSettings.objects.get(user=user, chat_id=chat_id)
#         except PredictiveSettings.DoesNotExist:
#             return Response({"error": "No predictive settings found. Please confirm columns first."},
#                             status=status.HTTP_400_BAD_REQUEST)

#         entity_id_column = settings_obj.entity_column
#         target_column = settings_obj.target_column
#         time_column = settings_obj.time_column
#         time_frame = settings_obj.time_frame
#         time_frequency = settings_obj.time_frequency

#         # Retrieve the file info from in-memory user_schemas (you could store it in DB if you prefer)
#         uploaded_file_info = None
#         if user_id in user_schemas and len(user_schemas[user_id]) > 0:
#             uploaded_file_info = user_schemas[user_id][0]
#         else:
#             return Response({"error": "Uploaded file info not found."},
#                             status=status.HTTP_400_BAD_REQUEST)

#         # Prepare feature columns from suggestions or user input
#         feature_columns = []
#         if "suggestions" in uploaded_file_info:
#             feature_columns = uploaded_file_info["suggestions"].get("feature_columns", [])

#         # Basic validations
#         columns_list = [col['column_name'] for col in uploaded_file_info['schema']]
#         if entity_id_column and entity_id_column not in columns_list:
#             return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."},
#                             status=status.HTTP_400_BAD_REQUEST)
#         if target_column and target_column not in columns_list:
#             return Response({"error": f"Target column '{target_column}' does not exist."},
#                             status=status.HTTP_400_BAD_REQUEST)

#         # Has date column?
#         has_date_column = uploaded_file_info.get('has_date_column', False)
#         file_url = uploaded_file_info.get('file_url')
#         table_name_raw = os.path.splitext(uploaded_file_info['name'])[0]
#         sanitized_table_name = self.sanitize_identifier(table_name_raw)

#         # Now the actual notebook creation (kept from your original code)...
#         if has_date_column and time_column:
#             # Time-based approach
#             final_time_frame = time_frame if time_frame else "1 WEEK"
#             final_notebook = self.create_dynamic_time_based_notebook(
#                 entity_id_column=entity_id_column,
#                 time_column=time_column,
#                 target_column=target_column,
#                 table_name=sanitized_table_name,
#                 time_horizon=final_time_frame,
#                 extra_features=feature_columns,
#                 time_frequency=time_frequency
#             )
#             notebook_sanitized = self.sanitize_notebook(final_notebook)
#             notebook_json = nbformat.writes(notebook_sanitized, version=4)

#             # Save in DB
#             notebook = Notebook.objects.create(
#                 user=user,
#                 chat=chat_id,
#                 entity_column=entity_id_column,
#                 target_column=target_column,
#                 time_column=time_column,
#                 time_frame=final_time_frame,
#                 time_frequency=time_frequency,
#                 features=feature_columns,
#                 file_url=file_url,
#                 notebook_json=notebook_json
#             )

#             user_notebooks[user_id] = {'time_based_notebook': notebook_json}
#             return Response({
#                 "message": "Notebook generated and saved successfully.",
#                 "notebook_id": notebook.id,
#                 "notebook_data": notebook_json
#             }, status=status.HTTP_200_OK)
#         else:
#             # Non time-based approach
#             notebook_entity_target = self.create_entity_target_notebook(
#                 entity_id_column,
#                 target_column,
#                 sanitized_table_name,
#                 columns_list
#             )
#             notebook_features = self.create_features_notebook(
#                 feature_columns,
#                 sanitized_table_name,
#                 columns_list
#             )
#             notebook_entity_target_sanitized = self.sanitize_notebook(notebook_entity_target)
#             notebook_features_sanitized = self.sanitize_notebook(notebook_features)

#             notebook_entity_target_json = nbformat.writes(notebook_entity_target_sanitized, version=4)
#             notebook_features_json = nbformat.writes(notebook_features_sanitized, version=4)

#             notebook_entity_target_record = Notebook.objects.create(
#                 user=user,
#                 chat=chat_id,
#                 entity_column=entity_id_column,
#                 target_column=target_column,
#                 features=feature_columns,
#                 file_url=file_url,
#                 notebook_json=notebook_entity_target_json
#             )

#             notebook_features_record = Notebook.objects.create(
#                 user=user,
#                 chat=chat_id,
#                 entity_column=entity_id_column,
#                 target_column=target_column,
#                 features=feature_columns,
#                 file_url=file_url,
#                 notebook_json=notebook_features_json
#             )

#             user_notebooks[user_id] = {
#                 'entity_target_notebook': notebook_entity_target_json,
#                 'features_notebook': notebook_features_json
#             }

#             return Response({
#                 "message": "Notebooks generated and saved successfully (non-time-based).",
#                 "entity_target_notebook_id": notebook_entity_target_record.id,
#                 "features_notebook_id": notebook_features_record.id,
#                 "entity_target_notebook": notebook_entity_target_json,
#                 "features_notebook": notebook_features_json
#             }, status=status.HTTP_200_OK)
    



#     def create_dynamic_time_based_notebook(
#         self,
#         entity_id_column: str,
#         time_column: str,
#         target_column: str,
#         table_name: str,
#         time_horizon: str,
#         extra_features: list,
#         time_frequency: str = None
#     ):
#         import nbformat
#         import datetime
#         import numpy as np
#         import pandas as pd
#         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

#         def parse_time_horizon(th: str):
#             parts = th.strip().split()
#             if len(parts) < 2:
#                 return 1, "week"
#             number = parts[0]
#             unit = parts[1].lower().rstrip('s')
#             return int(number), unit

#         def _stringify_timestamps(rows):
#             for row in rows:
#                 for k, v in row.items():
#                     if isinstance(v, (pd.Timestamp, datetime.datetime)):
#                         row[k] = v.strftime("%Y-%m-%d %H:%M:%S")

#         horizon_number, horizon_unit = parse_time_horizon(time_horizon)

#         if time_frequency:
#             freq_lower = time_frequency.strip().lower()
#             if freq_lower.startswith("day"):
#                 increment_unit = "day"
#             elif freq_lower.startswith("week"):
#                 increment_unit = "week"
#             else:
#                 increment_unit = "month"
#         else:
#             increment_unit = "month"

#         nb = new_notebook()
#         cells = []

#         step1_markdown = new_markdown_cell(
#             "### Step 1: Determine relevant timestamps based on Time Frequency"
#         )
#         cells.append(step1_markdown)

#         query_step1 = f"""
#             WITH minmax AS (
#                 SELECT
#                     MIN({time_column}) AS min_ts,
#                     MAX({time_column}) AS max_ts
#                 FROM {table_name}
#             ),
#             all_relevant_dates AS (
#                 SELECT
#                     date_trunc('{increment_unit}', day_in_range) AS relevant_time
#                 FROM minmax
#                 CROSS JOIN UNNEST(
#                     SEQUENCE(min_ts, max_ts, INTERVAL '1' DAY)
#                 ) AS t(day_in_range)
#             )
#             SELECT DISTINCT relevant_time
#             FROM all_relevant_dates
#             ORDER BY relevant_time
#             LIMIT 1000
#         """.strip()

#         from nbformat.v4 import new_markdown_cell, new_code_cell, new_output

#         step1_cell = new_code_cell(query_step1)
#         df_step1 = execute_sql_query(query_step1)
#         step1_cell['execution_count'] = 1

#         if df_step1.empty:
#             step1_cell.outputs = [
#                 new_output(
#                     output_type='execute_result',
#                     data={'text/plain': f"No data returned for query:\n{query_step1}"},
#                     execution_count=1
#                 )
#             ]
#         else:
#             df_step1 = df_step1.replace([None, float('inf'), float('-inf')], None)
#             result_json = df_step1.to_dict(orient='records')
#             _stringify_timestamps(result_json)

#             columns = []
#             for col in df_step1.columns:
#                 guessed_type = infer_column_dtype(df_step1[col])
#                 columns.append({"name": col, "type": guessed_type})

#             text_repr = df_step1.head().to_string(index=False)
#             step1_cell.outputs = [
#                 new_output(
#                     output_type='execute_result',
#                     data={
#                         'application/json': {
#                             'rows': result_json,
#                             'columns': columns
#                         },
#                         'text/plain': text_repr
#                     },
#                     execution_count=1
#                 )
#             ]
#         cells.append(step1_cell)

#         step2_markdown = new_markdown_cell(
#             "### Step 2: For each entity, gather relevant times after earliest record"
#         )
#         cells.append(step2_markdown)

#         step1_sub = query_step1.strip().rstrip(';')
#         query_step2 = f"""
#             WITH entity_earliest_time AS (
#                 SELECT
#                     {entity_id_column} AS entity_id,
#                     MIN({time_column}) AS first_seen_time
#                 FROM {table_name}
#                 GROUP BY {entity_id_column}
#             ),
#             relevant_times_in_dataset AS (
#                 {step1_sub}
#             )
#             SELECT
#                 entity_earliest_time.entity_id,
#                 relevant_times_in_dataset.relevant_time AS analysis_time
#             FROM entity_earliest_time
#             JOIN relevant_times_in_dataset
#                 ON relevant_times_in_dataset.relevant_time >= entity_earliest_time.first_seen_time
#             ORDER BY analysis_time, entity_id
#             LIMIT 1000
#         """.strip()

#         step2_cell = new_code_cell(query_step2)
#         df_step2 = execute_sql_query(query_step2)
#         step2_cell['execution_count'] = 1

#         if df_step2.empty:
#             step2_cell.outputs = [
#                 new_output(
#                     output_type='execute_result',
#                     data={'text/plain': f"No data returned for query:\n{query_step2}"},
#                     execution_count=1
#                 )
#             ]
#         else:
#             df_step2 = df_step2.replace([None, float('inf'), float('-inf')], None)
#             result_json = df_step2.to_dict(orient='records')
#             _stringify_timestamps(result_json)

#             columns = []
#             for col in df_step2.columns:
#                 guessed_type = infer_column_dtype(df_step2[col])
#                 columns.append({"name": col, "type": guessed_type})

#             text_repr = df_step2.head().to_string(index=False)
#             step2_cell.outputs = [
#                 new_output(
#                     output_type='execute_result',
#                     data={
#                         'application/json': {'rows': result_json, 'columns': columns},
#                         'text/plain': text_repr
#                     },
#                     execution_count=1
#                 )
#             ]
#         cells.append(step2_cell)

#         step3_markdown = new_markdown_cell(
#             "### Step 3: Summarize the target measure over the chosen time horizon (partial window filter)"
#         )
#         cells.append(step3_markdown)

#         step2_sub = query_step2.strip().rstrip(';')
#         horizon_label = time_horizon.replace(' ', '_')
#         query_step3 = f"""
#             WITH last_time AS (
#                 SELECT MAX({time_column}) AS max_ts
#                 FROM {table_name}
#             ),
#             entity_times AS (
#                 {step2_sub}
#             )
#             SELECT
#                 entity_times.entity_id,
#                 entity_times.analysis_time,
#                 COALESCE(SUM(tbl.{target_column}), 0) AS target_within_{horizon_label}_after
#             FROM entity_times
#             LEFT JOIN {table_name} AS tbl
#                 ON tbl.{entity_id_column} = entity_times.entity_id
#                 AND tbl.{time_column} >= entity_times.analysis_time
#                 AND tbl.{time_column} < date_add('{horizon_unit}', {horizon_number}, entity_times.analysis_time)
#             WHERE entity_times.analysis_time <= date_add('{horizon_unit}', -{horizon_number}, (SELECT max_ts FROM last_time))
#             GROUP BY
#                 entity_times.entity_id,
#                 entity_times.analysis_time
#             ORDER BY
#                 entity_times.analysis_time,
#                 entity_times.entity_id
#             LIMIT 1000
#         """.strip()

#         step3_cell = new_code_cell(query_step3)
#         df_step3 = execute_sql_query(query_step3)
#         step3_cell['execution_count'] = 1

#         if df_step3.empty:
#             step3_cell.outputs = [
#                 new_output(
#                     output_type='execute_result',
#                     data={'text/plain': f"No data returned for query:\n{query_step3}"},
#                     execution_count=1
#                 )
#             ]
#         else:
#             df_step3 = df_step3.replace([None, float('inf'), float('-inf')], None)
#             result_json = df_step3.to_dict(orient='records')
#             _stringify_timestamps(result_json)

#             columns = []
#             for col in df_step3.columns:
#                 guessed_type = infer_column_dtype(df_step3[col])
#                 columns.append({"name": col, "type": guessed_type})

#             text_repr = df_step3.head().to_string(index=False)
#             step3_cell.outputs = [
#                 new_output(
#                     output_type='execute_result',
#                     data={
#                         'application/json': {'rows': result_json, 'columns': columns},
#                         'text/plain': text_repr
#                     },
#                     execution_count=1
#                 )
#             ]
#         cells.append(step3_cell)

#         step4_markdown = new_markdown_cell(
#             "### Step 4: Join additional features (1-year lookback before analysis_time)"
#         )
#         cells.append(step4_markdown)

#         if not extra_features:
#             feature_selects = ""
#         else:
#             feature_selects = ",\n    " + ",\n    ".join([f"tbl.{col}" for col in extra_features])

#         step3_sub = query_step3.strip().rstrip(';')
#         query_step4 = f"""
#             WITH core_set AS (
#                 {step3_sub}
#             )
#             SELECT
#                 core_set.entity_id,
#                 core_set.analysis_time,
#                 core_set.target_within_{horizon_label}_after
#                 {feature_selects}
#             FROM core_set
#             INNER JOIN {table_name} AS tbl
#                 ON tbl.{entity_id_column} = core_set.entity_id
#                 AND tbl.{time_column} < core_set.analysis_time
#                 AND tbl.{time_column} >= date_add('year', -1, core_set.analysis_time)
#             ORDER BY
#                 core_set.analysis_time,
#                 core_set.entity_id
#             LIMIT 1000
#         """.strip()

#         step4_cell = new_code_cell(query_step4)
#         df_step4 = execute_sql_query(query_step4)
#         step4_cell['execution_count'] = 1

#         if df_step4.empty:
#             step4_cell.outputs = [
#                 new_output(
#                     output_type='execute_result',
#                     data={'text/plain': f"No data returned for query:\n{query_step4}"},
#                     execution_count=1
#                 )
#             ]
#         else:
#             df_step4 = df_step4.replace([None, float('inf'), float('-inf')], None)
#             result_json = df_step4.to_dict(orient='records')
#             _stringify_timestamps(result_json)

#             columns = []
#             for col in df_step4.columns:
#                 guessed_type = infer_column_dtype(df_step4[col])
#                 columns.append({"name": col, "type": guessed_type})

#             text_repr = df_step4.head().to_string(index=False)
#             step4_cell.outputs = [
#                 new_output(
#                     output_type='execute_result',
#                     data={
#                         'application/json': {'rows': result_json, 'columns': columns},
#                         'text/plain': text_repr
#                     },
#                     execution_count=1
#                 )
#             ]
#         cells.append(step4_cell)

#         nb['cells'] = cells
#         return nb

#     def trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str, file_size_mb: float):
#         print("[DEBUG] Triggering Glue update for table:", table_name)
#         glue = get_glue_client()
#         unique_id = file_key.split('/')[1]
#         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/uploads/{unique_id}/"
#         glue_table_name = self.sanitize_identifier(os.path.splitext(table_name)[0])

#         storage_descriptor = {
#             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
#             'Location': s3_location,
#             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
#             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
#             'SerdeInfo': {
#                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
#                 'Parameters': {
#                     'field.delim': ',',
#                     'skip.header.line.count': '1'
#                 }
#             }
#         }

#         try:
#             glue.update_table(
#                 DatabaseName=ATHENA_SCHEMA_NAME,
#                 TableInput={
#                     'Name': glue_table_name,
#                     'StorageDescriptor': storage_descriptor,
#                     'TableType': 'EXTERNAL_TABLE'
#                 }
#             )
#             print("[DEBUG] Glue table updated successfully:", glue_table_name)
#         except glue.exceptions.EntityNotFoundException:
#             print("[DEBUG] Glue table not found, creating a new one:", glue_table_name)
#             glue.create_table(
#                 DatabaseName=ATHENA_SCHEMA_NAME,
#                 TableInput={
#                     'Name': glue_table_name,
#                     'StorageDescriptor': storage_descriptor,
#                     'TableType': 'EXTERNAL_TABLE'
#                 }
#             )
#             print("[DEBUG] Glue table created successfully:", glue_table_name)

#         user_id = self.get_user_id_from_file_key(file_key)
#         if user_id in user_schemas:
#             user_schemas[user_id][0]["glue_table_name"] = glue_table_name
#             print(f"[DEBUG] Stored Glue table name '{glue_table_name}' for user '{user_id}'.")

#         base_timeout = 80
#         additional_timeout_per_mb = 5
#         dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
#         self.wait_for_table_creation(glue_table_name, timeout=dynamic_timeout)

#     def get_user_id_from_file_key(self, file_key: str) -> str:
#         try:
#             return file_key.split("/")[1]
#         except IndexError:
#             print("[WARNING] Unable to extract user ID, defaulting to 'default_user'.")
#             return "default_user"

#     def sanitize_identifier(self, name):
#         return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())

#     def validate_column_exists(self, column_name, columns_list):
#         if not column_name:
#             return True
#         print("[DEBUG] Validating column existence:", column_name)
#         print("[DEBUG] Available columns:", columns_list)
#         norm_col = normalize_column_name(column_name)
#         norm_list = [normalize_column_name(c) for c in columns_list]
#         if norm_col in norm_list:
#             return True
#         else:
#             print("[DEBUG] Column not found after normalization:", norm_col)
#             return False

#     def sanitize_notebook(self, nb):
#         def sanitize(obj):
#             if isinstance(obj, dict):
#                 for k in obj:
#                     obj[k] = sanitize(obj[k])
#                 return obj
#             elif isinstance(obj, list):
#                 return [sanitize(v) for v in obj]
#             elif isinstance(obj, float):
#                 if np.isnan(obj) or np.isinf(obj):
#                     return None
#                 else:
#                     return obj
#             else:
#                 return obj

#         sanitize(nb)
#         return nb

#     def create_entity_target_notebook(self, entity_id_column, target_column, table_name, columns_list):
#         import nbformat
#         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

#         print("[DEBUG] Creating entity-target notebook...")
#         nb = new_notebook()
#         cells = []
#         cells.append(new_markdown_cell("Core Set"))

#         sanitized_entity_id_column = self.sanitize_identifier(entity_id_column) if entity_id_column else "*"
#         sanitized_target_column = self.sanitize_identifier(target_column) if target_column else "*"
#         sql_query_entity_target = (
#             f"SELECT {sanitized_entity_id_column}, {sanitized_target_column} "
#             f"FROM {table_name} LIMIT 10;"
#         )
#         # Make sure the query is a single string.
#         sql_query_entity_target = sql_query_entity_target.strip()
#         # Potentially add a code cell, etc. if needed
#         # Create a code cell with the query and add it to the notebook cells
#         code_cell = new_code_cell(sql_query_entity_target)
#         cells.append(code_cell)

#         nb['cells'] = cells
#         return nb

#     def create_features_notebook(self, feature_columns, table_name, columns_list):
#         import nbformat
#         from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell, new_output

#         print("[DEBUG] Creating features notebook...")
#         nb = new_notebook()
#         cells = []
#         cells.append(new_markdown_cell("Features or Attributes Test"))

#         sanitized_features = [self.sanitize_identifier(feature) for feature in feature_columns]
#         missing_columns = [feature for feature in sanitized_features if feature not in columns_list]
#         if missing_columns:
#             error_message = f"The following feature columns do not exist in the dataset: {', '.join(missing_columns)}"
#             cells.append(new_markdown_cell(f"**Error:** {error_message}"))
#             nb['cells'] = cells
#             return nb

#         feature_query = (
#             f"SELECT\n    " + ",\n    ".join(sanitized_features) +
#             f"\nFROM {table_name}\nLIMIT 10;"
#         )
#         # You can add code cells, etc.
#         # Add a code cell with the features query

#         # Join and strip in case extra spaces or newline tokens exist.
#         feature_query = feature_query.strip()


#         code_cell = new_code_cell(feature_query)
#         cells.append(code_cell)

#         nb['cells'] = cells
#         return nb

#     def wait_for_table_creation(self, table_name, timeout):
#         import time
#         glue_client = get_glue_client()
#         start_time = time.time()
#         glue_table_ready = False
#         athena_table_ready = False

#         print("[DEBUG] Waiting for Glue table creation:", table_name)
#         while time.time() - start_time < timeout:
#             try:
#                 glue_client.get_table(DatabaseName=ATHENA_SCHEMA_NAME, Name=table_name)
#                 print("[DEBUG] Glue table is now available:", table_name)
#                 glue_table_ready = True
#                 break
#             except glue_client.exceptions.EntityNotFoundException:
#                 time.sleep(5)
#             except Exception as e:
#                 print("[ERROR] Unexpected error while checking Glue table availability:", e)
#                 return False

#         if not glue_table_ready:
#             print(f"[ERROR] Glue table '{table_name}' not available within {timeout} seconds.")
#             return False

#         print("[DEBUG] Checking Athena table availability:", table_name)
#         while time.time() - start_time < timeout:
#             try:
#                 query = f"SELECT 1 FROM {ATHENA_SCHEMA_NAME}.{table_name} LIMIT 1;"
#                 df = execute_sql_query(query)
#                 if df.empty:
#                     print("[DEBUG] Athena recognizes the table (no error), table ready:", table_name)
#                     athena_table_ready = True
#                     break
#                 else:
#                     print("[DEBUG] Athena table ready with data:", table_name)
#                     athena_table_ready = True
#                     break
#             except Exception as e:
#                 error_message = str(e)
#                 if "TableNotFoundException" in error_message or "TABLE_NOT_FOUND" in error_message:
#                     print("[DEBUG] Still waiting for Athena to recognize table:", table_name)
#                     time.sleep(10)
#                 else:
#                     print("[ERROR] Unexpected error while checking Athena table availability:", e)
#                     return False

#         if not athena_table_ready:
#             print(f"[ERROR] Table '{table_name}' not available in Athena within {timeout} seconds.")
#             return False

#         return True


# class ChatHistoryByUserView(APIView):
#     """
#     API to retrieve chat history for a specific user.
#     """
#     def get(self, request):
#         print("DEBUG: ChatHistoryByUserView GET method called")
#         user_id = request.GET.get('user_id')
#         print(f"DEBUG: Received user_id: {user_id}")

#         if not user_id:
#             print("ERROR: user_id is missing in the request")
#             return Response({"error": "user_id is required"}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             print(f"DEBUG: Querying chats for user_id: {user_id}")
#             chats = ChatBackup.objects.filter(user_id=user_id)
#             if not chats.exists():
#                 print(f"WARNING: No chats found for user_id: {user_id}")
#                 return Response(
#                     {"error": f"No chats found for the given user_id: {user_id}"},
#                     status=status.HTTP_404_NOT_FOUND,
#                 )

#             response_data = []
#             for chat in chats:
#                 messages = chat.messages
#                 user_messages = [msg for msg in messages if msg.get("sender") == "user"]
#                 assistant_messages = [msg for msg in messages if msg.get("sender") == "assistant"]

#                 response_data.append({
#                     "chat_id": chat.chat_id,
#                     "title": chat.title,
#                     "user_messages": user_messages,
#                     "assistant_messages": assistant_messages,
#                 })

#             print("DEBUG: Successfully prepared response data")
#             return Response(response_data, status=status.HTTP_200_OK)

#         except ChatBackup.DoesNotExist:
#             print(f"ERROR: No records found for user_id: {user_id}")
#             return Response(
#                 {"error": f"No chat records found for user_id={user_id}"},
#                 status=status.HTTP_404_NOT_FOUND,
#             )
#         except Exception as e:
#             print(f"ERROR: Unexpected error occurred: {e}")
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status

# class NotebookView(APIView):
#     def get(self, request):
#         user_id = request.query_params.get("user_id")
#         chat_id = request.query_params.get("chat_id")

#         if not user_id and not chat_id:
#             return Response(
#                 {"error": "At least one of 'user_id' or 'chat_id' is required."},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )

#         try:
#             filters = {}
#             if user_id:
#                 filters["user_id"] = user_id
#             if chat_id:
#                 filters["chat"] = chat_id

#             notebooks = Notebook.objects.filter(**filters)
#             if not notebooks.exists():
#                 return Response(
#                     {"error": "No notebooks found for the given criteria."},
#                     status=status.HTTP_404_NOT_FOUND,
#                 )

#             notebook_data = [
#                 {
#                     "id": notebook.id,
#                     "user_id": notebook.user_id,
#                     "chat_id": notebook.chat,
#                     "entity_column": notebook.entity_column,
#                     "target_column": notebook.target_column,
#                     "time_column": notebook.time_column,
#                     "time_frame": notebook.time_frame,
#                     "time_frequency": notebook.time_frequency,
#                     "features": notebook.features,
#                     "file_url": notebook.file_url,
#                     "notebook_json": notebook.notebook_json,
#                     "created_at": notebook.created_at,
#                 }
#                 for notebook in notebooks
#             ]

#             return Response(
#                 {
#                     "message": "Notebooks retrieved successfully.",
#                     "user_id": user_id,
#                     "notebooks": notebook_data,
#                 },
#                 status=status.HTTP_200_OK,
#             )

#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)












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
# They track whether we are awaiting a “confirm yes” response,
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
from .models import PredictiveSettings
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


# SYSTEM_INSTRUCTIONS = (
#     "You are a highly intelligent and helpful PACX AI assistant. Your responses should be clear and concise. "
#     "Assist the user in forming the predictive question and any corrections they provide. Reflect the confirmed or "
#     "corrected schema back to the user before proceeding, asking one question at a time.\n\n"
#     "Steps:\n"
#     "1. Discuss the subject they want to predict.\n"
#     "2. Confirm the target value they want to predict.\n"
#     "3. Check if there's a specific time frame for the prediction.\n"
#     "4. Determine whether the prediction should occur on a recurring basis (e.g., daily, weekly, monthly) or after a specific event.\n"
#     "5. Reference the dataset schema if available; if not, ask to upload it.\n"
#     "6. Once all necessary information is confirmed, provide a summary and let the user know they can generate the notebook."
# )


SYSTEM_INSTRUCTIONS = (
    "You are a highly intelligent and helpful PACX AI assistant. Your responses should be clear and concise. "
    "Assist the user in forming the predictive question and any corrections they provide. Reflect the confirmed or "
    "corrected schema back to the user before proceeding, asking one question at a time.\n\n"
    "Steps:\n"
    "1. Discuss the subject they want to predict.\n"
    "2. Confirm the target value they want to predict.\n"
    "3. Check if there's a specific time frame for the prediction (e.g., next month, 6 month, year).\n"
    "4. Reference the dataset schema if available; if not, ask to upload it.\n"
    "5. Once all necessary information is confirmed, provide a summary and let the user know they can generate the notebook."
)

# # Prompt template for AI conversation
# SYSTEM_INSTRUCTIONS = (
#     "You are a highly intelligent and helpful AI assistant. Your responses should be clear and concise. "
#     "Assist the user in forming a predictive question and any corrections they provide. "
# )

chat_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_INSTRUCTIONS),
    HumanMessagePromptTemplate.from_template(
        "Conversation so far:\n{history}\nUser input:\n{user_input}"
    )
])

def get_s3_client():
    print("[DEBUG] Creating S3 client...")
    return boto3.client(
        's3',
        aws_access_key_id=AWS_ACCESS_KEY_ID,
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        region_name=AWS_S3_REGION_NAME
    )


def confirm_value_change(ps, field, new_value, old_value):
    """
    Additional validation step for critical field changes
    """
    if field == 'target_column' and old_value and new_value != old_value:
        logger.warning(f"Critical field change attempted: {field} from {old_value} to {new_value}")
        
        # Add your validation logic here
        # For example, require explicit confirmation message
        if not awaiting_confirmation[ps.id]:
            awaiting_confirmation[ps.id] = True
            pending_column_updates[ps.id][field] = new_value
            return False
            
    return True

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

def infer_column_dtype(series: pd.Series, threshold: float = 0.8) -> str:
    """
    Attempt to infer the most likely data type of a pandas Series.
    This uses 'errors=coerce' for date parsing + threshold-based classification.
    """
    series = series.dropna().astype(str).str.strip()
    if series.empty:
        return "string"

    # Attempt #1: flexible parse toggling dayfirst
    for dayfirst_val in [False, True]:
        dt_series = pd.to_datetime(series, dayfirst=dayfirst_val, errors='coerce')
        valid_count = dt_series.notnull().sum()
        total_count = len(series)
        if total_count > 0 and (valid_count / total_count) >= threshold:
            return "timestamp"

    # Attempt #2: known date formats
    known_formats = ["%Y-%m-%d", "%d-%m-%Y", "%m/%d/%Y", "%d/%m/%Y"]
    for fmt in known_formats:
        dt_series = pd.to_datetime(series, format=fmt, errors='coerce')
        valid_count = dt_series.notnull().sum()
        total_count = len(series)
        if total_count > 0 and (valid_count / total_count) >= threshold:
            return "timestamp"

    # Boolean check
    boolean_values = {'true', 'false', '1', '0', 'yes', 'no', 't', 'f', 'y', 'n'}
    unique_values = set(series.str.lower().unique())
    if unique_values.issubset(boolean_values):
        return "boolean"

    # Integer check
    try:
        int_series = pd.to_numeric(series, errors='raise')
        if (int_series % 1 == 0).all():
            int_min = int_series.min()
            int_max = int_series.max()
            if int_min >= -2147483648 and int_max <= 2147483647:
                return "int"
            else:
                return "bigint"
    except ValueError:
        pass

    # Double check
    try:
        pd.to_numeric(series, errors='raise', downcast='float')
        return "double"
    except ValueError:
        pass

    return "string"

def standardize_datetime_columns(df: pd.DataFrame, schema: list) -> pd.DataFrame:
    """
    For columns recognized as 'timestamp', convert them to a standard ISO datetime string.
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


def update_predictive_settings(ps, parsed_updates, schema_columns):
    """
    Update only fields explicitly mentioned in parsed_updates, with improved validation.
    Also keep predictive_question in sync if the target_column changes.
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
    ]

    def validate_and_match(proposed_value, current_value):
        """Case-insensitive exact + fuzzy matching. If no match found, ask user to correct."""
        if not proposed_value:
            return current_value  # Keep existing if no new value

        # Attempt exact (case-insensitive) match
        for col in schema_columns:
            if col.lower() == proposed_value.lower():
                return col

        # If no exact match, try fuzzy
        matches = difflib.get_close_matches(proposed_value, schema_columns, n=1, cutoff=0.6)
        if matches:
            return matches[0]

        # If no match, log a warning
        logger.warning(f"No column match found for '{proposed_value}'. Keeping old value '{current_value}'.")
        return current_value

    updated_fields = {}
    with transaction.atomic():
        for field in updateable_fields:
            if field in parsed_updates:
                new_value = parsed_updates[field]
                current_value = getattr(ps, field)

                if field in ['target_column', 'entity_column', 'time_column']:
                    # Validate column name in schema
                    validated = validate_and_match(new_value, current_value)
                    if validated != current_value:
                        setattr(ps, field, validated)
                        updated_fields[field] = validated
                else:
                    # For non-column fields (time_frame, time_frequency, predictive_question, etc.)
                    if new_value and new_value != current_value:
                        setattr(ps, field, new_value)
                        updated_fields[field] = new_value

        # ----------------------------
        # If the target_column changed, also auto-update the predictive_question (optional logic)
        # ----------------------------
        if 'target_column' in updated_fields:
            new_target = ps.target_column
            # You can incorporate time_frame or user text, or do a simple question:
            ps.predictive_question = f"How can we predict {new_target}?"
            updated_fields['predictive_question'] = ps.predictive_question

        if updated_fields:
            ps.save(update_fields=list(updated_fields.keys()))

    logger.info(f"[update_predictive_settings] Updated fields: {updated_fields}")
    return updated_fields






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
    

### HANDLE FILE UPLOAD
    def handle_file_upload(self, request, files):
        """
        1) Ingests CSV/XLSX, normalizes columns
        2) Infers schema
        3) Uploads to S3
        4) Creates or updates table in Glue
        5) Returns 'uploaded_files_info' including suggestions
        """
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

            # Normalize column names
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

            # Infer schema and standardize date columns
            raw_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]
            df = standardize_datetime_columns(df, raw_schema)
            final_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]

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
                df[col_name] = df[col_name].astype(str).str.strip().str.lower().replace(replacement_dict)
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
                        # Re-upload as CSV to S3
                        csv_buffer = BytesIO()
                        df.to_csv(csv_buffer, index=False, encoding='utf-8')
                        csv_buffer.seek(0)
                        s3.upload_fileobj(csv_buffer, os.getenv('AWS_STORAGE_BUCKET_NAME'), file_key)
                        print("[DEBUG] S3 upload successful:", file_key)
                        s3.head_object(
                            Bucket=os.getenv('AWS_STORAGE_BUCKET_NAME'),
                            Key=file_key
                        )
                        file_url = f"s3://{os.getenv('AWS_STORAGE_BUCKET_NAME')}/{file_key}"
                        file_instance.file_url = file_url
                        file_instance.save()

                        # Store the schema in DB
                        FileSchema.objects.create(file=file_instance, schema=final_schema)

                        file_size_mb = file.size / (1024 * 1024)
                        self.trigger_glue_update(new_file_name, final_schema, file_key, file_size_mb)

                        # Build suggestions.
                        # (Note: Use the same key names as in PredictiveSettings: target_column and entity_column)
                        # First use a simple suggestion (e.g. last column for target, first for entity)
                        target_suggestion = suggest_target_column(df, [])
                        entity_suggestion = suggest_entity_id_column(df)
                        feature_candidates = [c for c in df.columns if c not in [entity_suggestion, target_suggestion]]

                        # Optionally, run GPT to refine the suggestions
                        full_chat_history = ""
                        conversation_chain = user_conversations.get(f"{user_id}_{request.data.get('chat_id', '')}")
                        if conversation_chain:
                            for msg in conversation_chain.memory.chat_memory.messages:
                                if msg.type == "human":
                                    full_chat_history += f"User: {msg.content}\n"
                                else:
                                    full_chat_history += f"Assistant: {msg.content}\n"
                        gpt_response = gpt_suggest_columns(df.columns.tolist(), full_chat_history)
                        # Override if GPT returns something valid
                        if gpt_response.get("suggested_target_column"):
                            target_suggestion = gpt_response.get("suggested_target_column")
                        if gpt_response.get("suggested_entity_column"):
                            entity_suggestion = gpt_response.get("suggested_entity_column")
                        if not target_suggestion:
                            target_suggestion = df.columns[-1]
                        if not entity_suggestion:
                            entity_suggestion = df.columns[0]

                        uploaded_files_info.append({
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

        # Store the uploaded files info (including suggestions) in memory for this user.
        user_schemas[user_id] = uploaded_files_info

        # (Optionally, initialize the conversation chain if needed.)
        chat_id = request.data.get("chat_id", "")
        memory_key = f"{user_id}_{chat_id}" if chat_id else user_id
        if memory_key not in user_conversations:
            conversation_chain = ConversationChain(
                llm=llm_chatgpt,
                prompt=chat_prompt,
                input_key="user_input",
                memory=ConversationBufferMemory()
            )
            user_conversations[memory_key] = conversation_chain
        else:
            conversation_chain = user_conversations[memory_key]

        # Insert a message about the newly discovered schema
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

    




    # Add this validation function to UnifiedChatGPTAPI class
    def validate_schema_updates(self, parsed_updates: dict, available_columns: list) -> tuple[dict, list[str]]:
        """
        Validates parsed updates against available schema columns.
        Returns (validated_updates, error_messages)
        """
        validated = {}
        errors = []
        
        # Normalize all available columns for case-insensitive comparison
        normalized_columns = {col.lower(): col for col in available_columns}
        
        # Validate target column
        if 'target_column' in parsed_updates:
            target = parsed_updates['target_column'].lower()
            if target in normalized_columns:
                validated['target_column'] = normalized_columns[target]
            else:
                errors.append(f"Target column '{parsed_updates['target_column']}' not found in schema")
        
        # Validate entity column
        if 'entity_column' in parsed_updates:
            entity = parsed_updates['entity_column'].lower()
            if entity in normalized_columns:
                validated['entity_column'] = normalized_columns[entity]
            else:
                errors.append(f"Entity column '{parsed_updates['entity_column']}' not found in schema")
        
        # # Validate time column
        # if 'time_column' in parsed_updates:
        #     time_col = parsed_updates['time_column'].lower()
        #     if time_col in normalized_columns:
        #         validated['time_column'] = normalized_columns[time_col]
        #     else:
        #         errors.append(f"Time column '{parsed_updates['time_column']}' not found in schema")
        
        # # Pass through non-column fields
        # for field in ['time_frame', 'time_frequency', 'predictive_question']:
        #     if field in parsed_updates:
        #         validated[field] = parsed_updates[field]
                
        return validated, errors




    # def handle_chat(self, request):
    #     user_input = request.data.get("message", "").strip()
    #     user_id = request.data.get("user_id", "default_user")
    #     chat_id = request.data.get("chat_id")

    #     if not user_input:
    #         return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

    #     # Validate user existence
    #     try:
    #         user = User.objects.get(id=user_id)
    #     except User.DoesNotExist:
    #         return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

    #     # Create or get a chat backup. If no chat_id is provided, create a new chat.
    #     if not chat_id:
    #         chat_id = str(uuid.uuid4())
    #         chat_title = user_input[:50]
    #         ChatBackup.objects.create(
    #             user=user,
    #             chat_id=chat_id,
    #             title=chat_title,
    #             messages=[]
    #         )

    #     chat_obj, _ = ChatBackup.objects.get_or_create(
    #         user=user,
    #         chat_id=chat_id,
    #         defaults={"title": user_input[:50], "messages": []}
    #     )

    #     # Get schema columns (if available) from the in-memory user_schemas.
    #     schema_columns = []
    #     if user_id in user_schemas and user_schemas[user_id]:
    #         # Use the schema from the first uploaded file
    #         schema_columns = [col["column_name"] for col in user_schemas[user_id][0]["schema"]]

    #     # Parse user input into structured updates.
    #     system_prompt = "Extract user instructions about target/entity/time_frame/time_column/time_frequency/predictive_question."
    #     parsed_updates = parse_nlu_input(
    #         system_prompt=system_prompt,
    #         user_message=user_input,
    #         schema_columns=schema_columns
    #     )

    #     # If the user’s message is a simple confirmation, override parsed_updates with the stored suggestions.
    #     if user_input.strip().lower() in ["yes", "suggestions are correct", "confirm"]:
    #         if user_id in user_schemas and user_schemas[user_id]:
    #             file_info = user_schemas[user_id][0]
    #             suggestions = file_info.get("suggestions", {})
    #             # Use the suggestions from file upload:
    #             if suggestions.get("target_column"):
    #                 parsed_updates["target_column"] = suggestions["target_column"]
    #             if suggestions.get("entity_column"):
    #                 parsed_updates["entity_column"] = suggestions["entity_column"]
    #             # If a date column was detected and only one exists, set it as the time column.
    #             if file_info.get("has_date_column") and file_info.get("date_columns"):
    #                 if len(file_info["date_columns"]) == 1:
    #                     parsed_updates["time_column"] = file_info["date_columns"][0]

    #     # Load existing predictive settings or create new ones.
    #     ps, created = PredictiveSettings.objects.get_or_create(
    #         user=user,
    #         chat_id=chat_id,
    #         defaults={
    #             'target_column': None,
    #             'entity_column': None,
    #             'time_column': None,
    #             'predictive_question': None,
    #             'time_frame': None,
    #             'time_frequency': None,
    #             'machine_learning_type': 'regression'
    #         }
    #     )

    #     # Update the settings using our dedicated helper.
    #     updated_fields = update_predictive_settings(ps, parsed_updates, schema_columns)

    #     # Prepare or restore the conversation chain.
    #     memory_key = f"{user_id}_{chat_id}"
    #     if memory_key not in user_conversations:
    #         restored_memory = ConversationBufferMemory(return_messages=True)
    #         for msg in chat_obj.messages:
    #             if msg["sender"] == "assistant":
    #                 restored_memory.chat_memory.add_message(AIMessage(content=msg["text"]))
    #             else:
    #                 restored_memory.chat_memory.add_message(HumanMessage(content=msg["text"]))
    #         user_conversations[memory_key] = ConversationChain(
    #             llm=llm_chatgpt,
    #             prompt=chat_prompt,
    #             input_key="user_input",
    #             memory=restored_memory
    #         )

    #     conversation_chain = user_conversations[memory_key]
    #     assistant_response = conversation_chain.run(user_input=user_input)

    #     # Save the conversation to the chat backup.
    #     timestamp = datetime.datetime.now().isoformat()
    #     chat_obj.messages.append({"sender": "user", "text": user_input, "timestamp": timestamp})
    #     chat_obj.messages.append({"sender": "assistant", "text": assistant_response, "timestamp": timestamp})
    #     chat_obj.save()

    #     # Determine if the notebook generation should be enabled.
    #     show_generate_notebook = bool(
    #         ps.target_column and 
    #         ps.entity_column and 
    #         (not ps.time_column or ps.time_column in schema_columns)
    #     )

    #     return Response({
    #         "response": assistant_response,
    #         "chat_id": chat_id,
    #         "show_generate_notebook": show_generate_notebook,
    #         "corrected_target_column": ps.target_column,
    #         "corrected_entity_column": ps.entity_column,
    #         "corrected_time_column": ps.time_column,
    #         "settings_updated": bool(updated_fields)  # Indicates if any settings were changed
    #     })


    # def create_new_chat(self, user, initial_text):
    #     """
    #     Creates a new chat backup with no previous conversation,
    #     and returns the new chat object and its unique chat_id.
    #     """
    #     chat_id = str(uuid.uuid4())
    #     chat_title = initial_text[:50]
    #     # Create a new chat backup with empty messages.
    #     chat_obj = ChatBackup.objects.create(
    #         user=user,
    #         chat_id=chat_id,
    #         title=chat_title,
    #         messages=[]
    #     )
    #     # Also clear any existing conversation chain for this chat if present.
    #     memory_key = f"{user.id}_{chat_id}"
    #     if memory_key in user_conversations:
    #         del user_conversations[memory_key]
    #     # Create a new conversation chain with empty memory.
    #     conversation_chain = ConversationChain(
    #         llm=llm_chatgpt,
    #         prompt=chat_prompt,
    #         input_key="user_input",
    #         memory=ConversationBufferMemory(return_messages=True)
    #     )
    #     user_conversations[memory_key] = conversation_chain
    #     return chat_obj, chat_id

    def create_new_chat(self, user, initial_text):
        """
        Creates a new chat backup with no previous conversation,
        and returns the new chat object + unique chat_id.
        """
        chat_id = str(uuid.uuid4())
        chat_title = initial_text[:50]
        chat_obj = ChatBackup.objects.create(
            user=user,
            chat_id=chat_id,
            title=chat_title,
            messages=[]
        )
        # Reset or create a new conversation chain
        memory_key = f"{user.id}_{chat_id}"
        conversation_chain = ConversationChain(
            llm=llm_chatgpt,
            prompt=chat_prompt,
            input_key="user_input",
            memory=ConversationBufferMemory(return_messages=True)
        )
        user_conversations[memory_key] = conversation_chain

        return chat_obj, chat_id
    


    # def handle_chat(self, request):
    #     user_input = request.data.get("message", "").strip()
    #     user_id = request.data.get("user_id", "default_user")
    #     chat_id = request.data.get("chat_id")
    #     new_chat_flag = request.data.get("new_chat", False)

    #     if not user_input:
    #         return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

    #     # Validate user
    #     try:
    #         user = User.objects.get(id=user_id)
    #     except User.DoesNotExist:
    #         return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

    #     # Create new chat if needed
    #     if new_chat_flag or not chat_id:
    #         chat_obj, chat_id = self.create_new_chat(user, user_input)
    #     else:
    #         chat_obj, _ = ChatBackup.objects.get_or_create(
    #             user=user,
    #             chat_id=chat_id,
    #             defaults={"title": user_input[:50], "messages": []}
    #         )
    #         if not chat_obj.messages:
    #             memory_key = f"{user_id}_{chat_id}"
    #             conversation_chain = ConversationChain(
    #                 llm=llm_chatgpt,
    #                 prompt=chat_prompt,
    #                 input_key="user_input",
    #                 memory=ConversationBufferMemory(return_messages=True)
    #             )
    #             user_conversations[memory_key] = conversation_chain

    #     # Gather columns
    #     schema_columns = []
    #     if user_id in user_schemas and user_schemas[user_id]:
    #         schema_columns = [col["column_name"] for col in user_schemas[user_id][0]["schema"]]

    #     # Parse user request
    #     parsed_result = parse_nlu_input(
    #         system_prompt="Extract user instructions about target/entity/time_frame/time_column/time_frequency/predictive_question.",
    #         user_message=user_input,
    #         schema_columns=schema_columns
    #     )
    #     is_confirmation = parsed_result["is_confirmation"]
    #     explicit_updates = parsed_result["updates"]

    #     # Pull or create PredictiveSettings
    #     ps, created = PredictiveSettings.objects.get_or_create(
    #         user=user,
    #         chat_id=chat_id,
    #         defaults={
    #             'target_column': None,
    #             'entity_column': None,
    #             'time_column': None,
    #             'predictive_question': None,
    #             'time_frame': None,
    #             'time_frequency': None,
    #             'machine_learning_type': 'regression'
    #         }
    #     )

    #     # Decide which fields to apply
    #     parsed_updates = {}
    #     if explicit_updates:
    #         parsed_updates = explicit_updates
    #         logger.info(f"[handle_chat] applying explicit updates: {parsed_updates}")
    #     elif is_confirmation:
    #         # <-- If you want to do NOTHING on "yes," just pass
    #         logger.info("[handle_chat] user confirmed; ignoring fallback suggestions.")
    #         pass

    #     logger.info(f"[handle_chat] Before update => {ps.__dict__}")
    #     logger.info(f"[handle_chat] parsed_updates => {parsed_updates}")

    #     updated_fields = update_predictive_settings(ps, parsed_updates, schema_columns)
    #     ps.refresh_from_db()
    #     logger.info(f"[handle_chat] After update => {ps.__dict__}")
    #     logger.info(f"[handle_chat] Updated fields => {updated_fields}")

    #     # Reconstruct or retrieve conversation chain
    #     memory_key = f"{user_id}_{chat_id}"
    #     if memory_key not in user_conversations:
    #         restored_memory = ConversationBufferMemory(return_messages=True)
    #         for msg in chat_obj.messages:
    #             if msg["sender"] == "assistant":
    #                 restored_memory.chat_memory.add_message(AIMessage(content=msg["text"]))
    #             else:
    #                 restored_memory.chat_memory.add_message(HumanMessage(content=msg["text"]))
    #         conversation_chain = ConversationChain(
    #             llm=llm_chatgpt,
    #             prompt=chat_prompt,
    #             input_key="user_input",
    #             memory=restored_memory
    #         )
    #         user_conversations[memory_key] = conversation_chain
    #     else:
    #         conversation_chain = user_conversations[memory_key]

    #     # Get GPT's answer
    #     assistant_response = conversation_chain.run(user_input=user_input)

    #     # Save conversation
    #     timestamp = datetime.datetime.now().isoformat()
    #     chat_obj.messages.append({"sender": "user", "text": user_input, "timestamp": timestamp})
    #     chat_obj.messages.append({"sender": "assistant", "text": assistant_response, "timestamp": timestamp})
    #     chat_obj.save()

    #     # Show 'Generate Notebook' if we have at least target + entity
    #     show_generate_notebook = bool(ps.target_column and ps.entity_column)

    #     return Response({
    #         "response": assistant_response,
    #         "chat_id": chat_id,
    #         "show_generate_notebook": show_generate_notebook,
    #         "corrected_target_column": ps.target_column,
    #         "corrected_entity_column": ps.entity_column,
    #         "corrected_time_column": ps.time_column,
    #         "settings_updated": bool(updated_fields),
    #         "predictive_question": ps.predictive_question
    #     })

    def handle_chat(self, request):
        user_input = request.data.get("message", "").strip()
        user_id = request.data.get("user_id", "default_user")
        chat_id = request.data.get("chat_id")
        new_chat_flag = request.data.get("new_chat", False)

        if not user_input:
            return Response({"error": "No input provided"}, status=status.HTTP_400_BAD_REQUEST)

        # Validate user
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        # Possibly create a new chat
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

        # Gather schema columns from the first uploaded file
        schema_columns = []
        if user_id in user_schemas and user_schemas[user_id]:
            schema_columns = [col["column_name"] for col in user_schemas[user_id][0]["schema"]]

        # Parse user input for changes
        parsed_result = parse_nlu_input(
            system_prompt="Extract user instructions about target/entity/time_frame/time_column/time_frequency/predictive_question.",
            user_message=user_input,
            schema_columns=schema_columns
        )
        is_confirmation = parsed_result["is_confirmation"]
        explicit_updates = parsed_result["updates"]

        # Get/Create PredictiveSettings
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
                'machine_learning_type': 'regression'
            }
        )

        parsed_updates = {}
        if explicit_updates:
            # User explicitly changed something: "Target Column: X" or "Entity Column: Y", etc.
            parsed_updates = explicit_updates
            logger.info(f"[handle_chat] applying explicit updates: {parsed_updates}")
        elif is_confirmation:
            # User said "yes" or "correct." If entity_column is still None, use suggestions.
            # This ensures we default to "customerid" if user didn’t explicitly set an entity column.
            if not ps.entity_column:  
                if user_id in user_schemas and user_schemas[user_id]:
                    file_info = user_schemas[user_id][0]
                    suggestions = file_info.get("suggestions", {})
                    suggested_entity = suggestions.get("entity_column")
                    if suggested_entity:
                        parsed_updates["entity_column"] = suggested_entity
                        logger.info(f"[handle_chat] Setting entity_column to '{suggested_entity}' by fallback suggestion.")
            if not ps.target_column:  
                if user_id in user_schemas and user_schemas[user_id]:
                    file_info = user_schemas[user_id][0]
                    suggestions = file_info.get("suggestions", {})
                    suggested_entity = suggestions.get("target_column")
                    if suggested_entity:
                        parsed_updates["target_column"] = suggested_entity
                        logger.info(f"[handle_chat] Setting entity_column to '{suggested_entity}' by fallback suggestion.")
            # We do NOT overwrite target_column here, to avoid reverting it to e.g. "churn."

        logger.info(f"[handle_chat] Before update => {ps.__dict__}")
        logger.info(f"[handle_chat] parsed_updates => {parsed_updates}")

        # Apply updates (includes fuzzy matching, etc.)
        updated_fields = update_predictive_settings(ps, parsed_updates, schema_columns)
        ps.refresh_from_db()

        logger.info(f"[handle_chat] After update => {ps.__dict__}")
        logger.info(f"[handle_chat] Updated fields => {updated_fields}")

        # Build or retrieve conversation chain
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

        # Ask the LLM to respond
        assistant_response = conversation_chain.run(user_input=user_input)

        # Save messages to DB
        timestamp = datetime.datetime.now().isoformat()
        chat_obj.messages.append({"sender": "user", "text": user_input, "timestamp": timestamp})
        chat_obj.messages.append({"sender": "assistant", "text": assistant_response, "timestamp": timestamp})
        chat_obj.save()

        # Only show "Generate Notebook" if we have a valid target & entity
        show_generate_notebook = bool(ps.target_column and ps.entity_column)

        return Response({
            "response": assistant_response,
            "chat_id": chat_id,
            "show_generate_notebook": show_generate_notebook,
            "corrected_target_column": ps.target_column,
            "corrected_entity_column": ps.entity_column,
            "corrected_time_column": ps.time_column,
            "settings_updated": bool(updated_fields),
            "predictive_question": ps.predictive_question
        })









    def reset_conversation(self, request):
        """
        Clears in-memory conversation. Optionally, remove rows from PredictiveSettings
        or ChatBackup if desired.
        """
        user_id = request.data.get("user_id", "default_user")
        # Remove any keys for the user (either exact or prefixed with user_id + '_')
        keys_to_delete = [key for key in user_conversations if key == str(user_id) or key.startswith(f"{user_id}_")]
        for key in keys_to_delete:
            del user_conversations[key]
        if user_id in user_schemas:
            del user_schemas[user_id]
        if user_id in user_notebooks:
            del user_notebooks[user_id]
        # Optional: Clear from DB as before.
        PredictiveSettings.objects.filter(user_id=user_id).delete()
        ChatBackup.objects.filter(user_id=user_id).delete()

        return Response({"message": "Conversation reset successful."})



    def format_schema_message(self, uploaded_file: Dict[str, Any]) -> str:
        schema = uploaded_file['schema']
        target_column = uploaded_file['suggestions'].get('target_column')
        # Fall back to "entity_column" if "entity_id_column" is not present.
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
            "- Optionally specify 'Time Frame: <X>' (e.g., 'Time Frame: 2 WEEKS') and 'Time Frequency: <daily|weekly|monthly>'.\n\n"
            "(Reply 'yes' to confirm or provide corrections in the format:\n"
            "'Entity ID Column: <column>, Target Column: <column>, Time Column: <column>, Time Frame: <X>, Time Frequency: <Y>')"
        )
        return schema_text

    




    def generate_notebook(self, request):
        """
        Reads final columns from PredictiveSettings. If we have a date column, do time-based approach; otherwise not.
        """
        user_id = request.data.get("user_id")
        chat_id = request.data.get("chat_id")

        if not user_id or not chat_id:
            return Response({"error": "user_id and chat_id are required."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Validate user
        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

        # Validate chat
        try:
            ChatBackup.objects.get(user=user, chat_id=chat_id)
        except ChatBackup.DoesNotExist:
            return Response({"error": "Chat not found."}, status=status.HTTP_404_NOT_FOUND)

        # Pull from PredictiveSettings
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

        # Retrieve the file info from in-memory user_schemas (you could store it in DB if you prefer)
        uploaded_file_info = None
        if user_id in user_schemas and len(user_schemas[user_id]) > 0:
            uploaded_file_info = user_schemas[user_id][0]
        else:
            return Response({"error": "Uploaded file info not found."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Prepare feature columns from suggestions or user input
        feature_columns = []
        if "suggestions" in uploaded_file_info:
            feature_columns = uploaded_file_info["suggestions"].get("feature_columns", [])

        # Basic validations
        columns_list = [col['column_name'] for col in uploaded_file_info['schema']]
        if entity_id_column and entity_id_column not in columns_list:
            return Response({"error": f"Entity ID column '{entity_id_column}' does not exist."},
                            status=status.HTTP_400_BAD_REQUEST)
        if target_column and target_column not in columns_list:
            return Response({"error": f"Target column '{target_column}' does not exist."},
                            status=status.HTTP_400_BAD_REQUEST)

        # Has date column?
        has_date_column = uploaded_file_info.get('has_date_column', False)
        file_url = uploaded_file_info.get('file_url')
        table_name_raw = os.path.splitext(uploaded_file_info['name'])[0]
        sanitized_table_name = self.sanitize_identifier(table_name_raw)

        # Now the actual notebook creation (kept from your original code)...
        if has_date_column and time_column:
            # Time-based approach
            final_time_frame = time_frame if time_frame else "1 WEEK"
            final_notebook = self.create_dynamic_time_based_notebook(
                entity_id_column=entity_id_column,
                time_column=time_column,
                target_column=target_column,
                table_name=sanitized_table_name,
                time_horizon=final_time_frame,
                extra_features=feature_columns,
                time_frequency=time_frequency
            )
            notebook_sanitized = self.sanitize_notebook(final_notebook)
            notebook_json = nbformat.writes(notebook_sanitized, version=4)

            # Save in DB
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

            user_notebooks[user_id] = {'time_based_notebook': notebook_json}
            return Response({
                "message": "Notebook generated and saved successfully.",
                "notebook_id": notebook.id,
                "notebook_data": notebook_json
            }, status=status.HTTP_200_OK)
        else:
            # Non time-based approach
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

            return Response({
                "message": "Notebooks generated and saved successfully (non-time-based).",
                "entity_target_notebook_id": notebook_entity_target_record.id,
                "features_notebook_id": notebook_features_record.id,
                "entity_target_notebook": notebook_entity_target_json,
                "features_notebook": notebook_features_json
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

        step1_markdown = new_markdown_cell(
            "### Step 1: Determine relevant timestamps based on Time Frequency"
        )
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
            LIMIT 1000
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

        step2_markdown = new_markdown_cell(
            "### Step 2: For each entity, gather relevant times after earliest record"
        )
        cells.append(step2_markdown)

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
        # Make sure the query is a single string.
        sql_query_entity_target = sql_query_entity_target.strip()
        # Potentially add a code cell, etc. if needed
        # Create a code cell with the query and add it to the notebook cells
        code_cell = new_code_cell(sql_query_entity_target)
        cells.append(code_cell)

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
        # You can add code cells, etc.
        # Add a code cell with the features query

        # Join and strip in case extra spaces or newline tokens exist.
        feature_query = feature_query.strip()


        code_cell = new_code_cell(feature_query)
        cells.append(code_cell)

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



# views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PredictiveSettings  # Adjust the import if your model is in a different module

# class PredictiveSettingsDetailView(APIView):
#     """
#     GET endpoint to retrieve predictive settings.
#     URL Format: /api/predictive-settings/<user_id>/<chat_id>/
    
#     Returns the predictive settings for a given user and chat.
#     If any field is empty, the response returns the string "Null" for that field.
#     """

#     def get(self, request, user_id, chat_id):
#         try:
#             ps = PredictiveSettings.objects.get(user_id=user_id, chat_id=chat_id)
#         except PredictiveSettings.DoesNotExist:
#             return Response(
#                 {"error": "Predictive settings not found for the given user and chat."},
#                 status=status.HTTP_404_NOT_FOUND
#             )
        
#         # Replace empty fields with "Null"
#         data = {
#             "target_column": ps.target_column if ps.target_column else "Null",
#             "entity_column": ps.entity_column if ps.entity_column else "Null",
#             "time_column": ps.time_column if ps.time_column else "Null",
#             "predictive_question": ps.predictive_question if ps.predictive_question else "Null",
#             "time_frame": ps.time_frame if ps.time_frame else "Null",
#             "time_frequency": ps.time_frequency if ps.time_frequency else "Null",
#             "machine_learning_type": ps.machine_learning_type if ps.machine_learning_type else "Null"
#         }
        
#         return Response(data, status=status.HTTP_200_OK)





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
                "machine_learning_type": ps.machine_learning_type if ps.machine_learning_type else "Null"
                # ... other fields ...
            }
            
            return Response(data, status=status.HTTP_200_OK)
            
        except PredictiveSettings.DoesNotExist:
            return Response(
                {"error": "Settings not found"},
                status=status.HTTP_404_NOT_FOUND
            )