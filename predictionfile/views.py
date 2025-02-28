# # # # import re
# # # # from typing import Dict, List
# # # # import boto3
# # # # from rest_framework.views import APIView
# # # # from rest_framework.response import Response
# # # # from rest_framework import status
# # # # from rest_framework.parsers import MultiPartParser, FormParser
# # # # from django.contrib.auth.models import User
# # # # from django.db import transaction
# # # # from botocore.exceptions import ClientError  # Changed from boto3.exceptions.ClientError
# # # # from chat.models import PredictiveSettings
# # # # from .models import PredictionFileInfo, UploadedFile
# # # # from .utils import get_s3_client, get_glue_client, execute_sql_query, infer_column_dtype, normalize_column_name, parse_dates_with_known_formats, standardize_datetime_columns
# # # # from .prediction_query_generator import PredictionQueryGenerator
# # # # import logging
# # # # import os
# # # # import uuid
# # # # from io import BytesIO
# # # # import pandas as pd
# # # # from django.conf import settings

# # # # logger = logging.getLogger(__name__)

# # # # class PredictionDatasetUploadAPI(APIView):
# # # #     parser_classes = [MultiPartParser, FormParser]

# # # #     def post(self, request):
# # # #         """
# # # #         Handle the upload of a prediction dataset, process it, save to S3 under "Predictions dataset uploads,"
# # # #         infer schema, determine column roles using PredictiveSettings (if available), and generate prediction queries.
# # # #         """
# # # #         user_id = request.data.get("user_id", "default_user")
# # # #         chat_id = request.data.get("chat_id", str(uuid.uuid4()))  # Generate or use provided chat_id
# # # #         files = request.FILES.getlist("file")

# # # #         if not files:
# # # #             return Response({"error": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)

# # # #         try:
# # # #             user = User.objects.get(id=user_id)
# # # #         except User.DoesNotExist:
# # # #             return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

# # # #         s3 = get_s3_client()
# # # #         glue = get_glue_client()
# # # #         uploaded_files_info = []

# # # #         for file in files:
# # # #             logger.info(f"[DEBUG] Processing prediction file: {file.name}")

# # # #             try:
# # # #                 # Read CSV or Excel with strict parsing (no row skipping)
# # # #                 if file.name.lower().endswith('.csv'):
# # # #                     df = pd.read_csv(
# # # #                         file,
# # # #                         low_memory=False,
# # # #                         encoding='utf-8',
# # # #                         delimiter=',',
# # # #                         na_values=['NA', 'N/A', ''],
# # # #                         on_bad_lines='error'
# # # #                     )
# # # #                 else:
# # # #                     df = pd.read_excel(file, engine='openpyxl')

# # # #                 if df.empty or not df.columns.any():
# # # #                     return Response(
# # # #                         {"error": f"Uploaded file {file.name} is empty or has no columns."},
# # # #                         status=status.HTTP_400_BAD_REQUEST
# # # #                     )

# # # #             except pd.errors.ParserError as e:
# # # #                 logger.error(f"[ERROR] CSV parsing error for {file.name}: {e}")
# # # #                 return Response(
# # # #                     {"error": f"CSV parsing error for file {file.name}: {str(e)}"},
# # # #                     status=status.HTTP_400_BAD_REQUEST
# # # #                 )
# # # #             except Exception as e:
# # # #                 logger.error(f"[ERROR] Error reading file {file.name}: {e}")
# # # #                 return Response(
# # # #                     {"error": f"Error reading file {file.name}: {str(e)}"},
# # # #                     status=status.HTTP_400_BAD_REQUEST
# # # #                 )

# # # #             # Normalize column names
# # # #             old_cols = df.columns.tolist()
# # # #             normalized_columns = [normalize_column_name(c) for c in df.columns]
# # # #             if len(normalized_columns) != len(set(normalized_columns)) or any(col == '' for col in normalized_columns):
# # # #                 return Response({"error": "Duplicate or empty columns detected after normalization."},
# # # #                                 status=status.HTTP_400_BAD_REQUEST)
# # # #             df.columns = normalized_columns

# # # #             logger.info("[DEBUG] Old columns -> Normalized columns:")
# # # #             for oc, nc in zip(old_cols, normalized_columns):
# # # #                 logger.info(f"   {oc} -> {nc}")

# # # #             # Infer schema and standardize dates
# # # #             raw_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]
# # # #             df = standardize_datetime_columns(df, raw_schema)
# # # #             final_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]

# # # #             has_date_column = any(c["data_type"] == "timestamp" for c in final_schema)
# # # #             possible_date_cols = [c["column_name"] for c in final_schema if c["data_type"] == "timestamp"]

# # # #             # Build unique file key for S3 under "Predictions dataset uploads"
# # # #             file_name_base, file_extension = os.path.splitext(file.name)
# # # #             file_name_base = file_name_base.lower().replace(' ', '_')
# # # #             unique_id = uuid.uuid4().hex[:8]
# # # #             new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
# # # #             s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
# # # #             s3_path = f"Predictions dataset uploads/{unique_id}/{s3_file_name}"
# # # #             logger.info(f"[DEBUG] Uploading prediction file to S3 at path: {s3_path}")

# # # #             try:
# # # #                 with transaction.atomic():
# # # #                     # Save the file record in Django
# # # #                     file.seek(0)
# # # #                     uploaded_file = UploadedFile.objects.create(name=new_file_name, file_url="")
# # # #                     file_instance = uploaded_file

# # # #                     # Convert DF to CSV in memory and upload to S3
# # # #                     csv_buffer = BytesIO()
# # # #                     df.to_csv(csv_buffer, index=False, encoding='utf-8')
# # # #                     csv_buffer.seek(0)
# # # #                     s3.upload_fileobj(csv_buffer, settings.AWS_STORAGE_BUCKET_NAME, s3_path)
# # # #                     logger.info(f"[DEBUG] S3 upload successful: {s3_path}")
# # # #                     s3.head_object(Bucket=settings.AWS_STORAGE_BUCKET_NAME, Key=s3_path)

# # # #                     file_url = f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{s3_path}"
# # # #                     file_instance.file_url = file_url
# # # #                     file_instance.save()

# # # #                     # Store the schema and metadata in PredictionFileInfo
# # # #                     PredictionFileInfo.objects.create(
# # # #                         user=user,
# # # #                         chat=chat_id,
# # # #                         file=file_instance,
# # # #                         name=file_instance.name,
# # # #                         file_url=file_instance.file_url,
# # # #                         schema=final_schema,
# # # #                         has_date_column=has_date_column,
# # # #                         date_columns=possible_date_cols,
# # # #                     )

# # # #                     file_size_mb = file.size / (1024 * 1024)
# # # #                     self._trigger_glue_update(new_file_name, final_schema, s3_path, file_size_mb)

# # # #                     # Determine column roles using PredictiveSettings (if available) and schema
# # # #                     settings = PredictiveSettings.objects.filter(user_id=user_id, chat_id=chat_id).first()
# # # #                     generator = PredictionQueryGenerator(file_info=PredictionFileInfo.objects.get(file=file_instance))
# # # #                     if settings:
# # # #                         generator.update_with_predictive_settings(settings)

# # # #                     # Generate and execute prediction queries
# # # #                     queries = generator.generate_prediction_queries()
# # # #                     results = generator.execute_queries(
# # # #                         settings.AWS_ACCESS_KEY_ID,
# # # #                         settings.AWS_SECRET_ACCESS_KEY,
# # # #                         settings.AWS_S3_REGION_NAME,
# # # #                         'pa_user_datafiles_db',
# # # #                         settings.AWS_ATHENA_S3_STAGING_DIR
# # # #                     )

# # # #                     file_info = {
# # # #                         'id': file_instance.id,
# # # #                         'name': file_instance.name,
# # # #                         'file_url': file_instance.file_url,
# # # #                         'schema': final_schema,
# # # #                         'file_size_mb': file_size_mb,
# # # #                         'has_date_column': has_date_column,
# # # #                         'date_columns': possible_date_cols,
# # # #                         'prediction_queries': {
# # # #                             'sampling_query': queries["sampling_query"],
# # # #                             'feature_query': queries["feature_query"]
# # # #                         },
# # # #                         'prediction_results': {
# # # #                             'sampling_results': results["sampling_query"].to_dict(orient='records') if not results["sampling_query"].empty else [],
# # # #                             'feature_results': results["feature_query"].to_dict(orient='records') if not results["feature_query"].empty else []
# # # #                         }
# # # #                     }
# # # #                     uploaded_files_info.append(file_info)

# # # #             except boto3.exceptions.ClientError as e:
# # # #                 logger.error(f"[ERROR] AWS ClientError: {e}")
# # # #                 return Response({'error': f'AWS error: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
# # # #             except Exception as e:
# # # #                 logger.error(f"[ERROR] Unexpected error during file processing: {e}")
# # # #                 return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # # #         logger.info("[DEBUG] Prediction dataset processed and queries generated.")
# # # #         return Response({
# # # #             "message": "Prediction dataset uploaded and processed successfully.",
# # # #             "uploaded_files": uploaded_files_info,
# # # #             "chat_id": chat_id
# # # #         }, status=status.HTTP_201_CREATED)

# # # #     def _trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str, file_size_mb: float):
# # # #         """Trigger Glue update for the prediction dataset, ensuring Athena compatibility."""
# # # #         logger.info(f"[DEBUG] Triggering Glue update for prediction table: {table_name}")
# # # #         glue = get_glue_client()
# # # #         s3_location = f"s3://{settings.AWS_STORAGE_BUCKET_NAME}/{file_key}"
# # # #         glue_table_name = self._sanitize_identifier(os.path.splitext(table_name)[0])

# # # #         storage_descriptor = {
# # # #             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
# # # #             'Location': s3_location,
# # # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # # #             'SerdeInfo': {
# # # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # # #                 'Parameters': {
# # # #                     'field.delim': ',',
# # # #                     'skip.header.line.count': '1'
# # # #                 }
# # # #             }
# # # #         }

# # # #         try:
# # # #             glue.update_table(
# # # #                 DatabaseName='pa_user_datafiles_db',
# # # #                 TableInput={
# # # #                     'Name': glue_table_name,
# # # #                     'StorageDescriptor': storage_descriptor,
# # # #                     'TableType': 'EXTERNAL_TABLE'
# # # #                 }
# # # #             )
# # # #             logger.info(f"[DEBUG] Glue table updated successfully: {glue_table_name}")
# # # #         except glue.exceptions.EntityNotFoundException:
# # # #             logger.info(f"[DEBUG] Glue table not found, creating a new one: {glue_table_name}")
# # # #             glue.create_table(
# # # #                 DatabaseName='pa_user_datafiles_db',
# # # #                 TableInput={
# # # #                     'Name': glue_table_name,
# # # #                     'StorageDescriptor': storage_descriptor,
# # # #                     'TableType': 'EXTERNAL_TABLE'
# # # #                 }
# # # #             )
# # # #             logger.info(f"[DEBUG] Glue table created successfully: {glue_table_name}")

# # # #         base_timeout = 80
# # # #         additional_timeout_per_mb = 5
# # # #         dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
# # # #         self._wait_for_table_creation(glue_table_name, dynamic_timeout)

# # # #     def _wait_for_table_creation(self, table_name, timeout):
# # # #         """Wait for Glue and Athena table creation to complete, ensuring compatibility."""
# # # #         import time
# # # #         glue_client = get_glue_client()
# # # #         start_time = time.time()
# # # #         glue_table_ready = False
# # # #         athena_table_ready = False

# # # #         logger.info(f"[DEBUG] Waiting for Glue table creation: {table_name}")
# # # #         while time.time() - start_time < timeout:
# # # #             try:
# # # #                 glue_client.get_table(DatabaseName='pa_user_datafiles_db', Name=table_name)
# # # #                 logger.info(f"[DEBUG] Glue table is now available: {table_name}")
# # # #                 glue_table_ready = True
# # # #                 break
# # # #             except glue_client.exceptions.EntityNotFoundException:
# # # #                 time.sleep(5)
# # # #             except Exception as e:
# # # #                 logger.error(f"[ERROR] Unexpected error while checking Glue table availability: {e}")
# # # #                 return False

# # # #         if not glue_table_ready:
# # # #             logger.error(f"[ERROR] Glue table '{table_name}' not available within {timeout} seconds.")
# # # #             return False

# # # #         logger.info(f"[DEBUG] Checking Athena table availability: {table_name}")
# # # #         while time.time() - start_time < timeout:
# # # #             try:
# # # #                 query = f"SELECT 1 FROM pa_user_datafiles_db.{table_name} LIMIT 1;"
# # # #                 df = execute_sql_query(query)
# # # #                 if df.empty:
# # # #                     logger.info(f"[DEBUG] Athena recognizes the table (no error), table ready: {table_name}")
# # # #                     athena_table_ready = True
# # # #                     break
# # # #                 else:
# # # #                     logger.info(f"[DEBUG] Athena table ready with data: {table_name}")
# # # #                     athena_table_ready = True
# # # #                     break
# # # #             except Exception as e:
# # # #                 error_message = str(e)
# # # #                 if "TableNotFoundException" in error_message or "TABLE_NOT_FOUND" in error_message:
# # # #                     logger.info(f"[DEBUG] Still waiting for Athena to recognize table: {table_name}")
# # # #                     time.sleep(10)
# # # #                 else:
# # # #                     logger.error(f"[ERROR] Unexpected error while checking Athena table availability: {e}")
# # # #                     return False

# # # #         if not athena_table_ready:
# # # #             logger.error(f"[ERROR] Table '{table_name}' not available in Athena within {timeout} seconds.")
# # # #             return False

# # # #         return True

# # # #     def _sanitize_identifier(self, name):
# # # #         """Sanitize identifiers to ensure SQL compatibility."""
# # # #         return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())




# # # import os
# # # import re
# # # import uuid
# # # import logging
# # # import boto3
# # # import openai
# # # import pandas as pd
# # # from io import BytesIO
# # # from typing import Dict, List

# # # from rest_framework.views import APIView
# # # from rest_framework.response import Response
# # # from rest_framework import status
# # # from rest_framework.parsers import MultiPartParser, FormParser
# # # from django.contrib.auth.models import User
# # # from django.db import transaction

# # # from botocore.exceptions import ClientError
# # # from chat.models import PredictiveSettings
# # # from .models import PredictionFileInfo, UploadedFile
# # # from .utils import (
# # #     get_s3_client, get_glue_client, execute_sql_query,
# # #     infer_column_dtype, normalize_column_name,
# # #     parse_dates_with_known_formats, standardize_datetime_columns
# # # )
# # # from .prediction_query_generator import PredictionQueryGenerator

# # # logger = logging.getLogger(__name__)

# # # # ------------------------------------------------------------------------------
# # # # 1. Load Environment Variables
# # # # ------------------------------------------------------------------------------
# # # # Make sure these env variables (AWS_ACCESS_KEY_ID, etc.) are set in your system
# # # # or loaded via something like python-dotenv or django-environ before Django runs.

# # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')


# # # # The Athena schema/database name you want to use:
# # # ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'

# # # # ------------------------------------------------------------------------------
# # # # 2. The Main Class: PredictionDatasetUploadAPI
# # # # ------------------------------------------------------------------------------


# # # class PredictionDatasetUploadAPI(APIView):
# # #     parser_classes = [MultiPartParser, FormParser]

# # #     def post(self, request):
# # #         """
# # #         Handle the upload of a prediction dataset, process it, save to S3 under "Predictions dataset uploads,"
# # #         infer schema, determine column roles using PredictiveSettings (if available), and generate prediction queries.
# # #         """
# # #         user_id = request.data.get("user_id", "default_user")
# # #         chat_id = request.data.get("chat_id", str(uuid.uuid4()))  # Generate or use provided chat_id
# # #         files = request.FILES.getlist("file")

# # #         if not files:
# # #             return Response({"error": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)

# # #         try:
# # #             user = User.objects.get(id=user_id)
# # #         except User.DoesNotExist:
# # #             return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

# # #         # Create AWS clients using your custom utility functions.
# # #         # Inside get_s3_client() / get_glue_client(), you can also read from environment (like below),
# # #         # or you can do the direct approach:
# # #         #
# # #         # s3 = boto3.client(
# # #         #     's3',
# # #         #     region_name=AWS_S3_REGION_NAME,
# # #         #     aws_access_key_id=AWS_ACCESS_KEY_ID,
# # #         #     aws_secret_access_key=AWS_SECRET_ACCESS_KEY
# # #         # )
# # #         #
# # #         # For now, let's assume your get_s3_client / get_glue_client do the same.
# # #         s3 = get_s3_client()
# # #         glue = get_glue_client()

# # #         uploaded_files_info = []

# # #         for file in files:
# # #             logger.info(f"[DEBUG] Processing prediction file: {file.name}")

# # #             # ----------------------------------------------------------
# # #             # 1. Read CSV or Excel with strict parsing
# # #             # ----------------------------------------------------------
# # #             try:
# # #                 if file.name.lower().endswith('.csv'):
# # #                     df = pd.read_csv(
# # #                         file,
# # #                         low_memory=False,
# # #                         encoding='utf-8',
# # #                         delimiter=',',
# # #                         na_values=['NA', 'N/A', ''],
# # #                         on_bad_lines='error'
# # #                     )
# # #                 else:
# # #                     df = pd.read_excel(file, engine='openpyxl')

# # #                 if df.empty or not df.columns.any():
# # #                     return Response(
# # #                         {"error": f"Uploaded file {file.name} is empty or has no columns."},
# # #                         status=status.HTTP_400_BAD_REQUEST
# # #                     )

# # #             except pd.errors.ParserError as e:
# # #                 logger.error(f"[ERROR] CSV parsing error for {file.name}: {e}")
# # #                 return Response(
# # #                     {"error": f"CSV parsing error for file {file.name}: {str(e)}"},
# # #                     status=status.HTTP_400_BAD_REQUEST
# # #                 )
# # #             except Exception as e:
# # #                 logger.error(f"[ERROR] Error reading file {file.name}: {e}")
# # #                 return Response(
# # #                     {"error": f"Error reading file {file.name}: {str(e)}"},
# # #                     status=status.HTTP_400_BAD_REQUEST
# # #                 )

# # #             # ----------------------------------------------------------
# # #             # 2. Normalize column names
# # #             # ----------------------------------------------------------
# # #             old_cols = df.columns.tolist()
# # #             normalized_columns = [normalize_column_name(c) for c in df.columns]
# # #             if len(normalized_columns) != len(set(normalized_columns)) or any(col == '' for col in normalized_columns):
# # #                 return Response(
# # #                     {"error": "Duplicate or empty columns detected after normalization."},
# # #                     status=status.HTTP_400_BAD_REQUEST
# # #                 )
# # #             df.columns = normalized_columns

# # #             logger.info("[DEBUG] Old columns -> Normalized columns:")
# # #             for oc, nc in zip(old_cols, normalized_columns):
# # #                 logger.info(f"   {oc} -> {nc}")

# # #             # ----------------------------------------------------------
# # #             # 3. Infer schema and standardize date columns
# # #             # ----------------------------------------------------------
# # #             raw_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]
# # #             df = standardize_datetime_columns(df, raw_schema)
# # #             final_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]

# # #             has_date_column = any(c["data_type"] == "timestamp" for c in final_schema)
# # #             possible_date_cols = [c["column_name"] for c in final_schema if c["data_type"] == "timestamp"]

# # #             # ----------------------------------------------------------
# # #             # 4. Build unique file key for S3
# # #             # ----------------------------------------------------------
# # #             file_name_base, file_extension = os.path.splitext(file.name)
# # #             file_name_base = file_name_base.lower().replace(' ', '_')
# # #             unique_id = uuid.uuid4().hex[:8]
# # #             new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
# # #             s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
# # #             s3_path = f"Predictions dataset uploads/{unique_id}/{s3_file_name}"
# # #             logger.info(f"[DEBUG] Uploading prediction file to S3 at path: {s3_path}")

# # #             # ----------------------------------------------------------
# # #             # 5. Save file record and upload data to S3
# # #             # ----------------------------------------------------------
# # #             try:
# # #                 with transaction.atomic():
# # #                     # Create a new UploadedFile record
# # #                     file.seek(0)
# # #                     uploaded_file = UploadedFile.objects.create(name=new_file_name, file_url="")
# # #                     file_instance = uploaded_file

# # #                     # Convert DF to CSV in-memory
# # #                     csv_buffer = BytesIO()
# # #                     df.to_csv(csv_buffer, index=False, encoding='utf-8')
# # #                     csv_buffer.seek(0)

# # #                     # Upload to S3 using environment-based bucket
# # #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, s3_path)
# # #                     logger.info(f"[DEBUG] S3 upload successful: {s3_path}")

# # #                     # Double-check that object is actually in S3
# # #                     s3.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=s3_path)

# # #                     # Update the file URL
# # #                     file_url = f"s3://{AWS_STORAGE_BUCKET_NAME}/{s3_path}"
# # #                     file_instance.file_url = file_url
# # #                     file_instance.save()

# # #                     # Create the PredictionFileInfo record
# # #                     PredictionFileInfo.objects.create(
# # #                         user=user,
# # #                         chat=chat_id,
# # #                         file=file_instance,
# # #                         name=file_instance.name,
# # #                         file_url=file_instance.file_url,
# # #                         schema=final_schema,
# # #                         has_date_column=has_date_column,
# # #                         date_columns=possible_date_cols,
# # #                     )

# # #                     file_size_mb = file.size / (1024 * 1024)

# # #                     # Trigger Glue update for Athena
# # #                     self._trigger_glue_update(new_file_name, final_schema, s3_path, file_size_mb)

# # #                     # --------------------------------------------------
# # #                     # 6. Handle PredictiveSettings (column roles, etc.)
# # #                     # --------------------------------------------------
# # #                     predictive_settings = PredictiveSettings.objects.filter(
# # #                         user_id=user_id, chat_id=chat_id
# # #                     ).first()

# # #                     generator = PredictionQueryGenerator(
# # #                         file_info=PredictionFileInfo.objects.get(file=file_instance)
# # #                     )
# # #                     if predictive_settings:
# # #                         generator.update_with_predictive_settings(predictive_settings)

# # #                     # --------------------------------------------------
# # #                     # 7. Generate and execute prediction queries
# # #                     # --------------------------------------------------
# # #                     # We assume predictive_settings has the AWS creds; if it's None, this may fail.
# # #                     queries = generator.generate_prediction_queries()
# # #                     results = generator.execute_queries(
# # #                         predictive_settings.AWS_ACCESS_KEY_ID,
# # #                         predictive_settings.AWS_SECRET_ACCESS_KEY,
# # #                         predictive_settings.AWS_S3_REGION_NAME,
# # #                         ATHENA_SCHEMA_NAME,  # e.g. 'pa_user_datafiles_db'
# # #                         predictive_settings.AWS_ATHENA_S3_STAGING_DIR
# # #                     )

# # #                     file_info = {
# # #                         'id': file_instance.id,
# # #                         'name': file_instance.name,
# # #                         'file_url': file_instance.file_url,
# # #                         'schema': final_schema,
# # #                         'file_size_mb': file_size_mb,
# # #                         'has_date_column': has_date_column,
# # #                         'date_columns': possible_date_cols,
# # #                         'prediction_queries': {
# # #                             'sampling_query': queries["sampling_query"],
# # #                             'feature_query': queries["feature_query"]
# # #                         },
# # #                         'prediction_results': {
# # #                             'sampling_results': (
# # #                                 results["sampling_query"].to_dict(orient='records')
# # #                                 if not results["sampling_query"].empty else []
# # #                             ),
# # #                             'feature_results': (
# # #                                 results["feature_query"].to_dict(orient='records')
# # #                                 if not results["feature_query"].empty else []
# # #                             )
# # #                         }
# # #                     }
# # #                     uploaded_files_info.append(file_info)

# # #             except ClientError as e:
# # #                 logger.error(f"[ERROR] AWS ClientError: {e}")
# # #                 return Response(
# # #                     {'error': f'AWS error: {str(e)}'},
# # #                     status=status.HTTP_500_INTERNAL_SERVER_ERROR
# # #                 )
# # #             except Exception as e:
# # #                 logger.error(f"[ERROR] Unexpected error during file processing: {e}")
# # #                 return Response(
# # #                     {'error': f'File processing failed: {str(e)}'},
# # #                     status=status.HTTP_500_INTERNAL_SERVER_ERROR
# # #                 )

# # #         logger.info("[DEBUG] Prediction dataset processed and queries generated.")
# # #         return Response(
# # #             {
# # #                 "message": "Prediction dataset uploaded and processed successfully.",
# # #                 "uploaded_files": uploaded_files_info,
# # #                 "chat_id": chat_id
# # #             },
# # #             status=status.HTTP_201_CREATED
# # #         )

# # #     # --------------------------------------------------------------------------
# # #     #  _trigger_glue_update
# # #     # --------------------------------------------------------------------------
# # #     def _trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str, file_size_mb: float):
# # #         """
# # #         Trigger Glue update for the prediction dataset, ensuring Athena compatibility.
# # #         """
# # #         logger.info(f"[DEBUG] Triggering Glue update for prediction table: {table_name}")
# # #         glue = get_glue_client()

# # #         # Build S3 location from env-based bucket
# # #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/{file_key}"
# # #         glue_table_name = self._sanitize_identifier(os.path.splitext(table_name)[0])

# # #         storage_descriptor = {
# # #             'Columns': [
# # #                 {"Name": col['column_name'], "Type": col['data_type']}
# # #                 for col in schema
# # #             ],
# # #             'Location': s3_location,
# # #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# # #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# # #             'SerdeInfo': {
# # #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# # #                 'Parameters': {
# # #                     'field.delim': ',',
# # #                     'skip.header.line.count': '1'
# # #                 }
# # #             }
# # #         }

# # #         try:
# # #             glue.update_table(
# # #                 DatabaseName=ATHENA_SCHEMA_NAME,
# # #                 TableInput={
# # #                     'Name': glue_table_name,
# # #                     'StorageDescriptor': storage_descriptor,
# # #                     'TableType': 'EXTERNAL_TABLE'
# # #                 }
# # #             )
# # #             logger.info(f"[DEBUG] Glue table updated successfully: {glue_table_name}")
# # #         except glue.exceptions.EntityNotFoundException:
# # #             logger.info(f"[DEBUG] Glue table not found, creating a new one: {glue_table_name}")
# # #             glue.create_table(
# # #                 DatabaseName=ATHENA_SCHEMA_NAME,
# # #                 TableInput={
# # #                     'Name': glue_table_name,
# # #                     'StorageDescriptor': storage_descriptor,
# # #                     'TableType': 'EXTERNAL_TABLE'
# # #                 }
# # #             )
# # #             logger.info(f"[DEBUG] Glue table created successfully: {glue_table_name}")

# # #         # Dynamically adjust the wait time based on file size
# # #         base_timeout = 80
# # #         additional_timeout_per_mb = 5
# # #         dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
# # #         self._wait_for_table_creation(glue_table_name, dynamic_timeout)

# # #     # --------------------------------------------------------------------------
# # #     #  _wait_for_table_creation
# # #     # --------------------------------------------------------------------------
# # #     def _wait_for_table_creation(self, table_name, timeout):
# # #         """
# # #         Wait for Glue and Athena table creation to complete, ensuring compatibility.
# # #         """
# # #         import time
# # #         glue_client = get_glue_client()
# # #         start_time = time.time()
# # #         glue_table_ready = False
# # #         athena_table_ready = False

# # #         logger.info(f"[DEBUG] Waiting for Glue table creation: {table_name}")
# # #         while time.time() - start_time < timeout:
# # #             try:
# # #                 glue_client.get_table(DatabaseName=ATHENA_SCHEMA_NAME, Name=table_name)
# # #                 logger.info(f"[DEBUG] Glue table is now available: {table_name}")
# # #                 glue_table_ready = True
# # #                 break
# # #             except glue_client.exceptions.EntityNotFoundException:
# # #                 time.sleep(5)
# # #             except Exception as e:
# # #                 logger.error(f"[ERROR] Unexpected error while checking Glue table availability: {e}")
# # #                 return False

# # #         if not glue_table_ready:
# # #             logger.error(f"[ERROR] Glue table '{table_name}' not available within {timeout} seconds.")
# # #             return False

# # #         logger.info(f"[DEBUG] Checking Athena table availability: {table_name}")
# # #         while time.time() - start_time < timeout:
# # #             try:
# # #                 query = f"SELECT 1 FROM {ATHENA_SCHEMA_NAME}.{table_name} LIMIT 1;"
# # #                 df = execute_sql_query(query)
# # #                 if df.empty:
# # #                     logger.info(f"[DEBUG] Athena recognizes the table (no error), table ready: {table_name}")
# # #                     athena_table_ready = True
# # #                     break
# # #                 else:
# # #                     logger.info(f"[DEBUG] Athena table ready with data: {table_name}")
# # #                     athena_table_ready = True
# # #                     break
# # #             except Exception as e:
# # #                 error_message = str(e)
# # #                 if "TableNotFoundException" in error_message or "TABLE_NOT_FOUND" in error_message:
# # #                     logger.info(f"[DEBUG] Still waiting for Athena to recognize table: {table_name}")
# # #                     time.sleep(10)
# # #                 else:
# # #                     logger.error(f"[ERROR] Unexpected error while checking Athena table availability: {e}")
# # #                     return False

# # #         if not athena_table_ready:
# # #             logger.error(f"[ERROR] Table '{table_name}' not available in Athena within {timeout} seconds.")
# # #             return False

# # #         return True

# # #     # --------------------------------------------------------------------------
# # #     #  _sanitize_identifier
# # #     # --------------------------------------------------------------------------
# # #     def _sanitize_identifier(self, name):
# # #         """
# # #         Sanitize identifiers to ensure SQL compatibility.
# # #         """
# # #         return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())





# # import os
# # import re
# # import uuid
# # import logging
# # import boto3
# # import openai
# # import pandas as pd
# # from io import BytesIO
# # from typing import Dict, List

# # from rest_framework.views import APIView
# # from rest_framework.response import Response
# # from rest_framework import status
# # from rest_framework.parsers import MultiPartParser, FormParser
# # from django.contrib.auth.models import User
# # from django.db import transaction

# # from botocore.exceptions import ClientError
# # from chat.models import PredictiveSettings
# # from .models import PredictionFileInfo, UploadedFile
# # from .utils import (
# #     get_s3_client, get_glue_client, execute_sql_query,
# #     infer_column_dtype, normalize_column_name,
# #     parse_dates_with_known_formats, standardize_datetime_columns
# # )
# # from .prediction_query_generator import PredictionQueryGenerator

# # logger = logging.getLogger(__name__)

# # # ------------------------------------------------------------------------------
# # # 1. Load Environment Variables
# # # ------------------------------------------------------------------------------
# # # These environment variables should be set in your system or via a .env file
# # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')

# # # The Athena schema/database name you want to use:
# # ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'

# # # ------------------------------------------------------------------------------
# # # 2. The Main Class: PredictionDatasetUploadAPI
# # # ------------------------------------------------------------------------------

# # class PredictionDatasetUploadAPI(APIView):
# #     parser_classes = [MultiPartParser, FormParser]

# #     def post(self, request):
# #         """
# #         Handle the upload of a prediction dataset, process it, save to S3 under "Predictions dataset uploads,"
# #         infer schema, determine column roles using PredictiveSettings (if available), and generate prediction queries.
# #         """
# #         user_id = request.data.get("user_id", "default_user")
# #         chat_id = request.data.get("chat_id", str(uuid.uuid4()))  # Generate or use provided chat_id
# #         files = request.FILES.getlist("file")

# #         if not files:
# #             return Response({"error": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)

# #         try:
# #             user = User.objects.get(id=user_id)
# #         except User.DoesNotExist:
# #             return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

# #         # Create AWS clients using your custom utility functions.
# #         s3 = get_s3_client()
# #         glue = get_glue_client()

# #         uploaded_files_info = []

# #         for file in files:
# #             logger.info(f"[DEBUG] Processing prediction file: {file.name}")

# #             # ----------------------------------------------------------
# #             # 1. Read CSV or Excel with strict parsing
# #             # ----------------------------------------------------------
# #             try:
# #                 if file.name.lower().endswith('.csv'):
# #                     df = pd.read_csv(
# #                         file,
# #                         low_memory=False,
# #                         encoding='utf-8',
# #                         delimiter=',',
# #                         na_values=['NA', 'N/A', ''],
# #                         on_bad_lines='error'
# #                     )
# #                 else:
# #                     df = pd.read_excel(file, engine='openpyxl')

# #                 if df.empty or not df.columns.any():
# #                     return Response(
# #                         {"error": f"Uploaded file {file.name} is empty or has no columns."},
# #                         status=status.HTTP_400_BAD_REQUEST
# #                     )

# #             except pd.errors.ParserError as e:
# #                 logger.error(f"[ERROR] CSV parsing error for {file.name}: {e}")
# #                 return Response(
# #                     {"error": f"CSV parsing error for file {file.name}: {str(e)}"},
# #                     status=status.HTTP_400_BAD_REQUEST
# #                 )
# #             except Exception as e:
# #                 logger.error(f"[ERROR] Error reading file {file.name}: {e}")
# #                 return Response(
# #                     {"error": f"Error reading file {file.name}: {str(e)}"},
# #                     status=status.HTTP_400_BAD_REQUEST
# #                 )

# #             # ----------------------------------------------------------
# #             # 2. Normalize column names
# #             # ----------------------------------------------------------
# #             old_cols = df.columns.tolist()
# #             normalized_columns = [normalize_column_name(c) for c in df.columns]
# #             if len(normalized_columns) != len(set(normalized_columns)) or any(col == '' for col in normalized_columns):
# #                 return Response(
# #                     {"error": "Duplicate or empty columns detected after normalization."},
# #                     status=status.HTTP_400_BAD_REQUEST
# #                 )
# #             df.columns = normalized_columns

# #             logger.info("[DEBUG] Old columns -> Normalized columns:")
# #             for oc, nc in zip(old_cols, normalized_columns):
# #                 logger.info(f"   {oc} -> {nc}")

# #             # ----------------------------------------------------------
# #             # 3. Infer schema and standardize date columns
# #             # ----------------------------------------------------------
# #             raw_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]
# #             df = standardize_datetime_columns(df, raw_schema)
# #             final_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]

# #             has_date_column = any(c["data_type"] == "timestamp" for c in final_schema)
# #             possible_date_cols = [c["column_name"] for c in final_schema if c["data_type"] == "timestamp"]

# #             # ----------------------------------------------------------
# #             # 4. Build unique file key for S3
# #             # ----------------------------------------------------------
# #             file_name_base, file_extension = os.path.splitext(file.name)
# #             file_name_base = file_name_base.lower().replace(' ', '_')
# #             unique_id = uuid.uuid4().hex[:8]
# #             new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
# #             s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
# #             s3_path = f"Predictions dataset uploads/{unique_id}/{s3_file_name}"
# #             logger.info(f"[DEBUG] Uploading prediction file to S3 at path: {s3_path}")

# #             # ----------------------------------------------------------
# #             # 5. Save file record and upload data to S3
# #             # ----------------------------------------------------------
# #             try:
# #                 with transaction.atomic():
# #                     # Create a new UploadedFile record
# #                     file.seek(0)
# #                     uploaded_file = UploadedFile.objects.create(name=new_file_name, file_url="")
# #                     file_instance = uploaded_file

# #                     # Convert DF to CSV in-memory
# #                     csv_buffer = BytesIO()
# #                     df.to_csv(csv_buffer, index=False, encoding='utf-8')
# #                     csv_buffer.seek(0)

# #                     # Upload to S3 using environment-based bucket
# #                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, s3_path)
# #                     logger.info(f"[DEBUG] S3 upload successful: {s3_path}")

# #                     # Double-check that object is actually in S3
# #                     s3.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=s3_path)

# #                     # Update the file URL
# #                     file_url = f"s3://{AWS_STORAGE_BUCKET_NAME}/{s3_path}"
# #                     file_instance.file_url = file_url
# #                     file_instance.save()

# #                     # Create the PredictionFileInfo record
# #                     PredictionFileInfo.objects.create(
# #                         user=user,
# #                         chat=chat_id,
# #                         file=file_instance,
# #                         name=file_instance.name,
# #                         file_url=file_instance.file_url,
# #                         schema=final_schema,
# #                         has_date_column=has_date_column,
# #                         date_columns=possible_date_cols,
# #                     )

# #                     file_size_mb = file.size / (1024 * 1024)

# #                     # Trigger Glue update for Athena
# #                     self._trigger_glue_update(new_file_name, final_schema, s3_path, file_size_mb)

# #                     # --------------------------------------------------
# #                     # 6. Handle PredictiveSettings (column roles, etc.)
# #                     # --------------------------------------------------
# #                     predictive_settings = PredictiveSettings.objects.filter(
# #                         user_id=user_id, chat_id=chat_id
# #                     ).first()

# #                     generator = PredictionQueryGenerator(
# #                         file_info=PredictionFileInfo.objects.get(file=file_instance)
# #                     )
# #                     if predictive_settings:
# #                         generator.update_with_predictive_settings(predictive_settings)

# #                     # --------------------------------------------------
# #                     # 7. Generate and execute prediction queries
# #                     # --------------------------------------------------
# #                     queries = generator.generate_prediction_queries()
# #                     # Use AWS credentials loaded from the environment variables
# #                     results = generator.execute_queries(
# #                         AWS_ACCESS_KEY_ID,
# #                         AWS_SECRET_ACCESS_KEY,
# #                         AWS_S3_REGION_NAME,
# #                         ATHENA_SCHEMA_NAME,  # e.g. 'pa_user_datafiles_db'
# #                         AWS_ATHENA_S3_STAGING_DIR
# #                     )

# #                     file_info = {
# #                         'id': file_instance.id,
# #                         'name': file_instance.name,
# #                         'file_url': file_instance.file_url,
# #                         'schema': final_schema,
# #                         'file_size_mb': file_size_mb,
# #                         'has_date_column': has_date_column,
# #                         'date_columns': possible_date_cols,
# #                         'prediction_queries': {
# #                             'sampling_query': queries["sampling_query"],
# #                             'feature_query': queries["feature_query"]
# #                         },
# #                         'prediction_results': {
# #                             'sampling_results': (
# #                                 results["sampling_query"].to_dict(orient='records')
# #                                 if not results["sampling_query"].empty else []
# #                             ),
# #                             'feature_results': (
# #                                 results["feature_query"].to_dict(orient='records')
# #                                 if not results["feature_query"].empty else []
# #                             )
# #                         }
# #                     }
# #                     uploaded_files_info.append(file_info)

# #             except ClientError as e:
# #                 logger.error(f"[ERROR] AWS ClientError: {e}")
# #                 return Response(
# #                     {'error': f'AWS error: {str(e)}'},
# #                     status=status.HTTP_500_INTERNAL_SERVER_ERROR
# #                 )
# #             except Exception as e:
# #                 logger.error(f"[ERROR] Unexpected error during file processing: {e}")
# #                 return Response(
# #                     {'error': f'File processing failed: {str(e)}'},
# #                     status=status.HTTP_500_INTERNAL_SERVER_ERROR
# #                 )

# #         logger.info("[DEBUG] Prediction dataset processed and queries generated.")
# #         return Response(
# #             {
# #                 "message": "Prediction dataset uploaded and processed successfully.",
# #                 "uploaded_files": uploaded_files_info,
# #                 "chat_id": chat_id
# #             },
# #             status=status.HTTP_201_CREATED
# #         )

# #     # --------------------------------------------------------------------------
# #     #  _trigger_glue_update
# #     # --------------------------------------------------------------------------
# #     def _trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_key: str, file_size_mb: float):
# #         """
# #         Trigger Glue update for the prediction dataset, ensuring Athena compatibility.
# #         """
# #         logger.info(f"[DEBUG] Triggering Glue update for prediction table: {table_name}")
# #         glue = get_glue_client()

# #         # Build S3 location from env-based bucket
# #         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/{file_key}"
# #         glue_table_name = self._sanitize_identifier(os.path.splitext(table_name)[0])

# #         storage_descriptor = {
# #             'Columns': [
# #                 {"Name": col['column_name'], "Type": col['data_type']}
# #                 for col in schema
# #             ],
# #             'Location': s3_location,
# #             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
# #             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
# #             'SerdeInfo': {
# #                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
# #                 'Parameters': {
# #                     'field.delim': ',',
# #                     'skip.header.line.count': '1'
# #                 }
# #             }
# #         }

# #         try:
# #             glue.update_table(
# #                 DatabaseName=ATHENA_SCHEMA_NAME,
# #                 TableInput={
# #                     'Name': glue_table_name,
# #                     'StorageDescriptor': storage_descriptor,
# #                     'TableType': 'EXTERNAL_TABLE'
# #                 }
# #             )
# #             logger.info(f"[DEBUG] Glue table updated successfully: {glue_table_name}")
# #         except glue.exceptions.EntityNotFoundException:
# #             logger.info(f"[DEBUG] Glue table not found, creating a new one: {glue_table_name}")
# #             glue.create_table(
# #                 DatabaseName=ATHENA_SCHEMA_NAME,
# #                 TableInput={
# #                     'Name': glue_table_name,
# #                     'StorageDescriptor': storage_descriptor,
# #                     'TableType': 'EXTERNAL_TABLE'
# #                 }
# #             )
# #             logger.info(f"[DEBUG] Glue table created successfully: {glue_table_name}")

# #         # Dynamically adjust the wait time based on file size
# #         base_timeout = 80
# #         additional_timeout_per_mb = 5
# #         dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
# #         self._wait_for_table_creation(glue_table_name, dynamic_timeout)

# #     # --------------------------------------------------------------------------
# #     #  _wait_for_table_creation
# #     # --------------------------------------------------------------------------
# #     def _wait_for_table_creation(self, table_name, timeout):
# #         """
# #         Wait for Glue and Athena table creation to complete, ensuring compatibility.
# #         """
# #         import time
# #         glue_client = get_glue_client()
# #         start_time = time.time()
# #         glue_table_ready = False
# #         athena_table_ready = False

# #         logger.info(f"[DEBUG] Waiting for Glue table creation: {table_name}")
# #         while time.time() - start_time < timeout:
# #             try:
# #                 glue_client.get_table(DatabaseName=ATHENA_SCHEMA_NAME, Name=table_name)
# #                 logger.info(f"[DEBUG] Glue table is now available: {table_name}")
# #                 glue_table_ready = True
# #                 break
# #             except glue_client.exceptions.EntityNotFoundException:
# #                 time.sleep(5)
# #             except Exception as e:
# #                 logger.error(f"[ERROR] Unexpected error while checking Glue table availability: {e}")
# #                 return False

# #         if not glue_table_ready:
# #             logger.error(f"[ERROR] Glue table '{table_name}' not available within {timeout} seconds.")
# #             return False

# #         logger.info(f"[DEBUG] Checking Athena table availability: {table_name}")
# #         while time.time() - start_time < timeout:
# #             try:
# #                 query = f"SELECT 1 FROM {ATHENA_SCHEMA_NAME}.{table_name} LIMIT 1;"
# #                 df = execute_sql_query(query)
# #                 if df.empty:
# #                     logger.info(f"[DEBUG] Athena recognizes the table (no error), table ready: {table_name}")
# #                     athena_table_ready = True
# #                     break
# #                 else:
# #                     logger.info(f"[DEBUG] Athena table ready with data: {table_name}")
# #                     athena_table_ready = True
# #                     break
# #             except Exception as e:
# #                 error_message = str(e)
# #                 if "TableNotFoundException" in error_message or "TABLE_NOT_FOUND" in error_message:
# #                     logger.info(f"[DEBUG] Still waiting for Athena to recognize table: {table_name}")
# #                     time.sleep(10)
# #                 else:
# #                     logger.error(f"[ERROR] Unexpected error while checking Athena table availability: {e}")
# #                     return False

# #         if not athena_table_ready:
# #             logger.error(f"[ERROR] Table '{table_name}' not available in Athena within {timeout} seconds.")
# #             return False

# #         return True

# #     # --------------------------------------------------------------------------
# #     #  _sanitize_identifier
# #     # --------------------------------------------------------------------------
# #     def _sanitize_identifier(self, name):
# #         """
# #         Sanitize identifiers to ensure SQL compatibility.
# #         """
# #         return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())






# import os
# import re
# import uuid
# import logging
# import boto3
# import pandas as pd
# from io import BytesIO
# from typing import Dict, List

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.parsers import MultiPartParser, FormParser
# from django.contrib.auth.models import User
# from django.db import transaction

# from botocore.exceptions import ClientError
# from chat.models import PredictiveSettings
# from .models import PredictionFileInfo, UploadedFile
# from .utils import (
#     get_s3_client, get_glue_client, execute_sql_query,
#     infer_column_dtype, normalize_column_name,
#     parse_dates_with_known_formats, standardize_datetime_columns
# )
# from .prediction_query_generator import PredictionQueryGenerator

# logger = logging.getLogger(__name__)

# # Load Environment Variables (ensure these are set correctly)
# AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')
# ATHENA_SCHEMA_NAME = os.getenv('ATHENA_SCHEMA_NAME', 'pa_user_datafiles_db')

# class PredictionDatasetUploadAPI(APIView):
#     parser_classes = [MultiPartParser, FormParser]

#     def post(self, request):
#         """
#         Handle the upload of a prediction dataset, process it, save to S3 under "Predictions dataset uploads,"
#         infer schema, determine column roles using PredictiveSettings (if available), and generate prediction queries.
#         """
#         user_id = request.data.get("user_id", "default_user")
#         chat_id = request.data.get("chat_id", str(uuid.uuid4()))  # Generate or use provided chat_id
#         files = request.FILES.getlist("file")

#         if not files:
#             logger.error("No files provided in the request")
#             return Response({"error": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)

#         try:
#             user = User.objects.get(id=user_id)
#         except User.DoesNotExist:
#             logger.error(f"User not found for user_id: {user_id}")
#             return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

#         s3 = get_s3_client()
#         glue = get_glue_client()
#         uploaded_files_info = []

#         for file in files:
#             logger.info(f"[DEBUG] Processing prediction file: {file.name}")

#             try:
#                 # Read CSV with strict parsing, explicitly handle date format
#                 if file.name.lower().endswith('.csv'):
#                     df = pd.read_csv(
#                         file,
#                         low_memory=False,
#                         encoding='utf-8',
#                         delimiter=',',
#                         na_values=['NA', 'N/A', ''],
#                         on_bad_lines='error',
#                         parse_dates=['date'],  # Explicitly parse the 'date' column
#                         date_parser=lambda x: pd.to_datetime(x, format='%d-%m-%Y %H:%M:%S', errors='coerce')
#                     )
#                 else:
#                     df = pd.read_excel(file, engine='openpyxl')

#                 logger.info(f"[DEBUG] Shape after reading file: {df.shape}")
#                 logger.info("[DEBUG] Head of DataFrame:\n", df.head(5))

#                 if df.empty:
#                     logger.error(f"Uploaded file {file.name} is empty")
#                     return Response(
#                         {"error": f"Uploaded file {file.name} is empty."},
#                         status=status.HTTP_400_BAD_REQUEST
#                     )
#                 if not df.columns.any():
#                     logger.error(f"Uploaded file {file.name} has no columns")
#                     return Response(
#                         {"error": f"Uploaded file {file.name} has no columns."},
#                         status=status.HTTP_400_BAD_REQUEST
#                     )

#             except pd.errors.ParserError as e:
#                 logger.error(f"[ERROR] CSV parsing error for {file.name}: {e}")
#                 return Response(
#                     {"error": f"CSV parsing error for file {file.name}: {str(e)}"},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )
#             except Exception as e:
#                 logger.error(f"[ERROR] Error reading file {file.name}: {e}")
#                 return Response(
#                     {"error": f"Error reading file {file.name}: {str(e)}"},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )

#             # Normalize column names
#             old_cols = df.columns.tolist()
#             normalized_columns = [normalize_column_name(c) for c in df.columns]
#             if len(normalized_columns) != len(set(normalized_columns)):
#                 logger.error("Duplicate columns detected after normalization")
#                 return Response(
#                     {"error": "Duplicate columns detected after normalization."},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )
#             if any(col == '' for col in normalized_columns):
#                 logger.error("Empty column names detected after normalization")
#                 return Response(
#                     {"error": "Some columns have empty names after normalization."},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )
#             df.columns = normalized_columns

#             logger.info("[DEBUG] Old columns -> Normalized columns:")
#             for oc, nc in zip(old_cols, normalized_columns):
#                 logger.info(f"   {oc} -> {nc}")

#             # Infer schema
#             raw_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]
#             logger.info("[DEBUG] Raw schema inferred:", raw_schema)

#             # Standardize date columns
#             rows_before_std = df.shape[0]
#             df = standardize_datetime_columns(df, raw_schema)
#             rows_after_std = df.shape[0]
#             logger.info(f"[DEBUG] Rows before date standardization: {rows_before_std}, after: {rows_after_std}")

#             # Re-check final schema
#             final_schema = []
#             for col in df.columns:
#                 final_schema.append({
#                     "column_name": col,
#                     "data_type": infer_column_dtype(df[col])
#                 })
#             logger.info("[DEBUG] Final schema after standardizing date columns:", final_schema)

#             # Fix boolean columns (e.g., holiday_flag)
#             boolean_columns = [c['column_name'] for c in final_schema if c['data_type'] == 'boolean']
#             replacement_dict = {
#                 '1': 'true', '0': 'false',
#                 'yes': 'true', 'no': 'false',
#                 't': 'true', 'f': 'false',
#                 'y': 'true', 'n': 'false',
#                 'true': 'true', 'false': 'false',
#             }
#             for col_name in boolean_columns:
#                 df[col_name] = (
#                     df[col_name].astype(str)
#                     .str.strip()
#                     .str.lower()
#                     .replace(replacement_dict)
#                 )
#                 unexpected_values = [v for v in df[col_name].unique() if v not in ['true', 'false']]
#                 if unexpected_values:
#                     logger.error(f"[ERROR] Unexpected boolean values in column {col_name}: {unexpected_values}")
#                     return Response(
#                         {"error": f"Unexpected boolean values in column {col_name}: {unexpected_values}"},
#                         status=status.HTTP_400_BAD_REQUEST
#                     )

#             has_date_column = any(c["data_type"] == "timestamp" for c in final_schema)
#             possible_date_cols = [c["column_name"] for c in final_schema if c["data_type"] == "timestamp"]

#             # Build unique file key for S3 under "Predictions dataset uploads"
#             file_name_base, file_extension = os.path.splitext(file.name)
#             file_name_base = file_name_base.lower().replace(' ', '_')
#             unique_id = uuid.uuid4().hex[:8]
#             new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
#             s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
#             s3_path = f"Predictions dataset uploads/{unique_id}/{s3_file_name}"
#             logger.info(f"[DEBUG] Uploading prediction file to S3 at path: {s3_path}")

#             try:
#                 with transaction.atomic():
#                     # Save the file record in Django
#                     file.seek(0)
#                     uploaded_file = UploadedFile.objects.create(name=new_file_name, file_url="")
#                     file_instance = uploaded_file

#                     # Convert DF to CSV in memory and upload to S3
#                     csv_buffer = BytesIO()
#                     df.to_csv(csv_buffer, index=False, encoding='utf-8')
#                     csv_buffer.seek(0)
#                     logger.info(f"[DEBUG] CSV buffer content length: {csv_buffer.getbuffer().nbytes}")
#                     s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, s3_path)
#                     logger.info(f"[DEBUG] S3 upload successful: {s3_path}")
#                     s3.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=s3_path)

#                     file_url = f"s3://{AWS_STORAGE_BUCKET_NAME}/{s3_path}"
#                     file_instance.file_url = file_url
#                     file_instance.save()

#                     # Store the schema and metadata in PredictionFileInfo
#                     PredictionFileInfo.objects.create(
#                         user=user,
#                         chat=chat_id,
#                         file=file_instance,
#                         name=file_instance.name,
#                         file_url=file_instance.file_url,
#                         schema=final_schema,
#                         has_date_column=has_date_column,
#                         date_columns=possible_date_cols,
#                     )

#                     file_size_mb = file.size / (1024 * 1024)

#                     # Trigger Glue update for Athena
#                     self._trigger_glue_update(new_file_name, final_schema, s3_path, file_size_mb)

#                     # Determine column roles using PredictiveSettings (if available) and schema
#                     predictive_settings = PredictiveSettings.objects.filter(user_id=user_id, chat_id=chat_id).first()
#                     generator = PredictionQueryGenerator(file_info=PredictionFileInfo.objects.get(file=file_instance))
#                     if predictive_settings:
#                         generator.update_with_predictive_settings(predictive_settings)

#                     # Generate and execute prediction queries
#                     queries = generator.generate_prediction_queries()
#                     results = generator.execute_queries(
#                         AWS_ACCESS_KEY_ID,
#                         AWS_SECRET_ACCESS_KEY,
#                         AWS_S3_REGION_NAME,
#                         ATHENA_SCHEMA_NAME,
#                         AWS_ATHENA_S3_STAGING_DIR
#                     )

#                     # Build file info without suggestions
#                     file_info = {
#                         'id': file_instance.id,
#                         'name': file_instance.name,
#                         'file_url': file_instance.file_url,
#                         'schema': final_schema,
#                         'file_size_mb': file_size_mb,
#                         'has_date_column': has_date_column,
#                         'date_columns': possible_date_cols,
#                         'prediction_queries': {
#                             'sampling_query': queries["sampling_query"],
#                             'feature_query': queries["feature_query"]
#                         },
#                         'prediction_results': {
#                             'sampling_results': results["sampling_query"].to_dict(orient='records') if not results["sampling_query"].empty else [],
#                             'feature_results': results["feature_query"].to_dict(orient='records') if not results["feature_query"].empty else []
#                         }
#                     }
#                     uploaded_files_info.append(file_info)

#                     # Additional debug: Run a simple Athena query to verify data
#                     debug_query = f"SELECT * FROM {ATHENA_SCHEMA_NAME}.{self._sanitize_identifier(os.path.splitext(new_file_name)[0])} LIMIT 10;"
#                     debug_results = execute_sql_query(debug_query)
#                     logger.info(f"[DEBUG] Athena debug query results: {debug_results.to_dict() if not debug_results.empty else 'No rows returned'}")

#             except ClientError as e:
#                 logger.error(f"[ERROR] AWS ClientError: {e}")
#                 return Response(
#                     {'error': f'AWS error: {str(e)}'},
#                     status=status.HTTP_500_INTERNAL_SERVER_ERROR
#                 )
#             except Exception as e:
#                 logger.error(f"[ERROR] Unexpected error during file processing: {e}")
#                 return Response(
#                     {'error': f'File processing failed: {str(e)}'},
#                     status=status.HTTP_500_INTERNAL_SERVER_ERROR
#                 )

#         logger.info("[DEBUG] Prediction dataset processed and queries generated.")
#         return Response(
#             {
#                 "message": "Prediction dataset uploaded and processed successfully.",
#                 "uploaded_files": uploaded_files_info,
#                 "chat_id": chat_id
#             },
#             status=status.HTTP_201_CREATED
#         )

#     def _trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], file_url: str, file_size_mb: float):
#         """
#         Trigger Glue update for the prediction dataset, ensuring Athena compatibility.
#         """
#         logger.info(f"[DEBUG] Triggering Glue update for prediction table: {table_name}")
#         glue = get_glue_client()

#         s3_location = f"s3://{AWS_STORAGE_BUCKET_NAME}/{file_url}"

#         glue_table_name = self._sanitize_identifier(os.path.splitext(table_name)[0])

#         storage_descriptor = {
#             'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
#             'Location': s3_location,
#             'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
#             'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
#             'SerdeInfo': {
#                 'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
#                 'Parameters': {
#                     'field.delim': ',',
#                     'skip.header.line.count': '1',
#                     'serialization.format': ','
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
#             logger.info(f"[DEBUG] Glue table updated successfully: {glue_table_name}")
#         except glue.exceptions.EntityNotFoundException:
#             logger.info(f"[DEBUG] Glue table not found, creating a new one: {glue_table_name}")
#             glue.create_table(
#                 DatabaseName=ATHENA_SCHEMA_NAME,
#                 TableInput={
#                     'Name': glue_table_name,
#                     'StorageDescriptor': storage_descriptor,
#                     'TableType': 'EXTERNAL_TABLE'
#                 }
#             )
#             logger.info(f"[DEBUG] Glue table created successfully: {glue_table_name}")
#             logger.info(f"[DEBUG] table created on this: {s3_location}")

#         base_timeout = 80
#         additional_timeout_per_mb = 5
#         dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
#         self._wait_for_table_creation(glue_table_name, dynamic_timeout)

#     def _wait_for_table_creation(self, table_name, timeout):
#         """
#         Wait for Glue and Athena table creation to complete, ensuring compatibility.
#         """
#         import time
#         glue_client = get_glue_client()
#         start_time = time.time()
#         glue_table_ready = False
#         athena_table_ready = False

#         logger.info(f"[DEBUG] Waiting for Glue table creation: {table_name}")
#         while time.time() - start_time < timeout:
#             try:
#                 glue_client.get_table(DatabaseName=ATHENA_SCHEMA_NAME, Name=table_name)
#                 logger.info(f"[DEBUG] Glue table is now available: {table_name}")
#                 glue_table_ready = True
#                 break
#             except glue_client.exceptions.EntityNotFoundException:
#                 time.sleep(5)
#             except Exception as e:
#                 logger.error(f"[ERROR] Unexpected error while checking Glue table availability: {e}")
#                 return False

#         if not glue_table_ready:
#             logger.error(f"[ERROR] Glue table '{table_name}' not available within {timeout} seconds.")
#             return False

#         logger.info(f"[DEBUG] Checking Athena table availability: {table_name}")
#         while time.time() - start_time < timeout:
#             try:
#                 query = f"SELECT 1 FROM {ATHENA_SCHEMA_NAME}.{table_name} LIMIT 1;"
#                 df = execute_sql_query(query)
#                 if df.empty:
#                     logger.info(f"[DEBUG] Athena recognizes the table (no error), table ready: {table_name}")
#                     athena_table_ready = True
#                     break
#                 else:
#                     logger.info(f"[DEBUG] Athena table ready with data: {table_name}")
#                     athena_table_ready = True
#                     break
#             except Exception as e:
#                 error_message = str(e)
#                 if "TableNotFoundException" in error_message or "TABLE_NOT_FOUND" in error_message:
#                     logger.info(f"[DEBUG] Still waiting for Athena to recognize table: {table_name}")
#                     time.sleep(10)
#                 else:
#                     logger.error(f"[ERROR] Unexpected error while checking Athena table availability: {e}")
#                     return False

#         if not athena_table_ready:
#             logger.error(f"[ERROR] Table '{table_name}' not available in Athena within {timeout} seconds.")
#             return False

#         return True

#     def _sanitize_identifier(self, name):
#         """
#         Sanitize identifiers to ensure SQL compatibility.
#         """
#         return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())



import os
import re
import uuid
import logging
import boto3
import pandas as pd
from io import BytesIO
from typing import Dict, List

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from django.contrib.auth.models import User
from django.db import transaction

from botocore.exceptions import ClientError
from chat.models import PredictiveSettings
from .models import PredictionFileInfo, UploadedFile
from .utils import (
    get_s3_client, get_glue_client, execute_sql_query,
    infer_column_dtype, normalize_column_name,
    parse_dates_with_known_formats, standardize_datetime_columns
)
from .prediction_query_generator import PredictionQueryGenerator

logger = logging.getLogger(__name__)

# Load Environment Variables (ensure these are set correctly)
AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME', 'pa-documents-storage-bucket')  # Default to provided bucket
AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')
ATHENA_SCHEMA_NAME = os.getenv('ATHENA_SCHEMA_NAME', 'pa_user_datafiles_db')

class PredictionDatasetUploadAPI(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request):
        """
        Handle the upload of a prediction dataset, process it, save to S3 under "Predictions dataset uploads,"
        infer schema, determine column roles using PredictiveSettings (if available), and generate prediction queries.
        """
        user_id = request.data.get("user_id", "default_user")
        chat_id = request.data.get("chat_id", str(uuid.uuid4()))  # Generate or use provided chat_id
        files = request.FILES.getlist("file")

        if not files:
            logger.error("No files provided in the request")
            return Response({"error": "No files provided"}, status=status.HTTP_400_BAD_REQUEST)

        try:
            user = User.objects.get(id=user_id)
        except User.DoesNotExist:
            logger.error(f"User not found for user_id: {user_id}")
            return Response({"error": "User not found"}, status=status.HTTP_404_NOT_FOUND)

        s3 = get_s3_client()
        glue = get_glue_client()
        uploaded_files_info = []

        for file in files:
            logger.info(f"[DEBUG] Processing prediction file: {file.name}")

            try:
                # Read CSV with strict parsing, explicitly handle date format
                if file.name.lower().endswith('.csv'):
                    df = pd.read_csv(
                        file,
                        low_memory=False,
                        encoding='utf-8',
                        delimiter=',',
                        na_values=['NA', 'N/A', ''],
                        on_bad_lines='error',
                        parse_dates=['date'],  # Explicitly parse the 'date' column
                        date_parser=lambda x: pd.to_datetime(x, format='%d-%m-%Y %H:%M:%S', errors='coerce')
                    )
                else:
                    df = pd.read_excel(file, engine='openpyxl')

                logger.info(f"[DEBUG] Shape after reading file: {df.shape}")
                logger.info(f"[DEBUG] Head of DataFrame:\n{df.head(5).to_string()}")

                if df.empty:
                    logger.error(f"Uploaded file {file.name} is empty")
                    return Response(
                        {"error": f"Uploaded file {file.name} is empty."},
                        status=status.HTTP_400_BAD_REQUEST
                    )
                if not df.columns.any():
                    logger.error(f"Uploaded file {file.name} has no columns")
                    return Response(
                        {"error": f"Uploaded file {file.name} has no columns."},
                        status=status.HTTP_400_BAD_REQUEST
                    )

            except pd.errors.ParserError as e:
                logger.error(f"[ERROR] CSV parsing error for {file.name}: {e}")
                return Response(
                    {"error": f"CSV parsing error for file {file.name}: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )
            except Exception as e:
                logger.error(f"[ERROR] Error reading file {file.name}: {e}")
                return Response(
                    {"error": f"Error reading file {file.name}: {str(e)}"},
                    status=status.HTTP_400_BAD_REQUEST
                )

            # Normalize column names
            old_cols = df.columns.tolist()
            normalized_columns = [normalize_column_name(c) for c in df.columns]
            if len(normalized_columns) != len(set(normalized_columns)):
                logger.error("Duplicate columns detected after normalization")
                return Response(
                    {"error": "Duplicate columns detected after normalization."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            if any(col == '' for col in normalized_columns):
                logger.error("Empty column names detected after normalization")
                return Response(
                    {"error": "Some columns have empty names after normalization."},
                    status=status.HTTP_400_BAD_REQUEST
                )
            df.columns = normalized_columns

            logger.info("[DEBUG] Old columns -> Normalized columns:")
            for oc, nc in zip(old_cols, normalized_columns):
                logger.info(f"   {oc} -> {nc}")

            # Infer schema
            raw_schema = [{"column_name": col, "data_type": infer_column_dtype(df[col])} for col in df.columns]
            logger.info(f"[DEBUG] Raw schema inferred: {raw_schema}")

            # Standardize date columns
            rows_before_std = df.shape[0]
            df = standardize_datetime_columns(df, raw_schema)
            rows_after_std = df.shape[0]
            logger.info(f"[DEBUG] Rows before date standardization: {rows_before_std}, after: {rows_after_std}")

            # Re-check final schema
            final_schema = []
            for col in df.columns:
                final_schema.append({
                    "column_name": col,
                    "data_type": infer_column_dtype(df[col])
                })
            logger.info(f"[DEBUG] Final schema after standardizing date columns: {final_schema}")

            # Fix boolean columns (e.g., holiday_flag)
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
                    logger.error(f"[ERROR] Unexpected boolean values in column {col_name}: {unexpected_values}")
                    return Response(
                        {"error": f"Unexpected boolean values in column {col_name}: {unexpected_values}"},
                        status=status.HTTP_400_BAD_REQUEST
                    )

            has_date_column = any(c["data_type"] == "timestamp" for c in final_schema)
            possible_date_cols = [c["column_name"] for c in final_schema if c["data_type"] == "timestamp"]

            # Build unique file path for S3 under "Predictions dataset uploads"
            file_name_base, file_extension = os.path.splitext(file.name)
            file_name_base = file_name_base.lower().replace(' ', '_')
            unique_id = uuid.uuid4().hex[:8]
            new_file_name = f"{file_name_base}_{unique_id}{file_extension}"
            s3_file_name = os.path.splitext(new_file_name)[0] + '.csv'
            s3_path = f"Predictions dataset uploads/{unique_id}/{s3_file_name}"
            logger.info(f"[DEBUG] Uploading prediction file to S3 at path: {s3_path}")

            try:
                with transaction.atomic():
                    # Save the file record in Django
                    file.seek(0)
                    uploaded_file = UploadedFile.objects.create(name=new_file_name, file_url="")
                    file_instance = uploaded_file

                    # Convert DF to CSV in memory and upload to S3
                    csv_buffer = BytesIO()
                    df.to_csv(csv_buffer, index=False, encoding='utf-8')
                    csv_buffer.seek(0)
                    logger.info(f"[DEBUG] CSV buffer content length: {csv_buffer.getbuffer().nbytes}")
                    s3.upload_fileobj(csv_buffer, AWS_STORAGE_BUCKET_NAME, s3_path)
                    logger.info(f"[DEBUG] S3 upload successful: {s3_path}")

                    # Verify S3 upload by listing the object
                    s3_objects = s3.list_objects_v2(Bucket=AWS_STORAGE_BUCKET_NAME, Prefix=s3_path)
                    if not s3_objects.get('Contents'):
                        logger.error(f"[ERROR] File not found in S3 at {s3_path}")
                        return Response(
                            {"error": f"File upload to S3 failed or file not found at {s3_path}"},
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR
                        )
                    logger.info(f"[DEBUG] S3 file verified at {s3_path}")
                    s3.head_object(Bucket=AWS_STORAGE_BUCKET_NAME, Key=s3_path)

                    # Construct the full S3 URL as file_url
                    file_url = f"s3://{AWS_STORAGE_BUCKET_NAME}/{s3_path}"
                    file_instance.file_url = file_url
                    file_instance.save()

                    # Store the schema and metadata in PredictionFileInfo
                    PredictionFileInfo.objects.create(
                        user=user,
                        chat=chat_id,
                        file=file_instance,
                        name=file_instance.name,
                        file_url=file_url,
                        schema=final_schema,
                        has_date_column=has_date_column,
                        date_columns=possible_date_cols,
                    )

                    file_size_mb = file.size / (1024 * 1024)

                    # Trigger Glue update for Athena using the directory path
                    s3_directory = f"s3://{AWS_STORAGE_BUCKET_NAME}/Predictions dataset uploads/{unique_id}/"
                    self._trigger_glue_update(new_file_name, final_schema, s3_directory, file_size_mb)

                    # Determine column roles using PredictiveSettings (if available) and schema
                    predictive_settings = PredictiveSettings.objects.filter(user_id=user_id, chat_id=chat_id).first()
                    generator = PredictionQueryGenerator(file_info=PredictionFileInfo.objects.get(file=file_instance))
                    if predictive_settings:
                        generator.update_with_predictive_settings(predictive_settings)

                    # Generate and execute prediction queries
                    queries = generator.generate_prediction_queries()
                    logger.info(f"[DEBUG] Generated sampling query: {queries['sampling_query']}")
                    logger.info(f"[DEBUG] Generated feature query: {queries['feature_query']}")

                    results = generator.execute_queries(
                        AWS_ACCESS_KEY_ID,
                        AWS_SECRET_ACCESS_KEY,
                        AWS_S3_REGION_NAME,
                        ATHENA_SCHEMA_NAME,
                        AWS_ATHENA_S3_STAGING_DIR
                    )

                    # Build file info
                    file_info = {
                        'id': file_instance.id,
                        'name': file_instance.name,
                        'file_url': file_url,
                        'schema': final_schema,
                        'file_size_mb': file_size_mb,
                        'has_date_column': has_date_column,
                        'date_columns': possible_date_cols,
                        'prediction_queries': {
                            'sampling_query': queries["sampling_query"],
                            'feature_query': queries["feature_query"]
                        },
                        'prediction_results': {
                            'sampling_results': results["sampling_query"].to_dict(orient='records') if not results["sampling_query"].empty else [],
                            'feature_results': results["feature_query"].to_dict(orient='records') if not results["feature_query"].empty else []
                        }
                    }
                    uploaded_files_info.append(file_info)

                    # Additional debug: Run a simple Athena query to verify data
                    glue_table_name = self._sanitize_identifier(os.path.splitext(new_file_name)[0])
                    debug_query = f"SELECT * FROM {ATHENA_SCHEMA_NAME}.{glue_table_name} LIMIT 10;"
                    debug_results = execute_sql_query(debug_query)
                    logger.info(f"[DEBUG] Athena debug query results for table {glue_table_name}: {debug_results.to_dict() if not debug_results.empty else 'No rows returned'}")

            except ClientError as e:
                logger.error(f"[ERROR] AWS ClientError: {e}")
                return Response(
                    {'error': f'AWS error: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )
            except Exception as e:
                logger.error(f"[ERROR] Unexpected error during file processing: {e}")
                return Response(
                    {'error': f'File processing failed: {str(e)}'},
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR
                )

        logger.info("[DEBUG] Prediction dataset processed and queries generated.")
        return Response(
            {
                "message": "Prediction dataset uploaded and processed successfully.",
                "uploaded_files": uploaded_files_info,
                "chat_id": chat_id
            },
            status=status.HTTP_201_CREATED
        )

    def _trigger_glue_update(self, table_name: str, schema: List[Dict[str, str]], s3_directory: str, file_size_mb: float):
        """
        Trigger Glue update for the prediction dataset, ensuring Athena compatibility.
        Uses the S3 directory path (up to the unique_id) instead of the full file URL.
        """
        logger.info(f"[DEBUG] Triggering Glue update for prediction table: {table_name}")
        glue = get_glue_client()

        s3_location = s3_directory  # Use the directory path (e.g., s3://pa-documents-storage-bucket/Predictions dataset uploads/389a100d/)
        logger.info(f"[DEBUG] Table created on this: {s3_location}")
        glue_table_name = self._sanitize_identifier(os.path.splitext(table_name)[0])

        storage_descriptor = {
            'Columns': [{"Name": col['column_name'], "Type": col['data_type']} for col in schema],
            'Location': s3_location,  # Point to the directory
            'InputFormat': 'org.apache.hadoop.mapred.TextInputFormat',
            'OutputFormat': 'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat',
            'SerdeInfo': {
                'SerializationLibrary': 'org.apache.hadoop.hive.serde2.lazy.LazySimpleSerDe',
                'Parameters': {
                    'field.delim': ',',
                    'skip.header.line.count': '1',
                    'serialization.format': ','
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
            logger.info(f"[DEBUG] Glue table updated successfully: {glue_table_name}")
        except glue.exceptions.EntityNotFoundException:
            logger.info(f"[DEBUG] Glue table not found, creating a new one: {glue_table_name}")
            glue.create_table(
                DatabaseName=ATHENA_SCHEMA_NAME,
                TableInput={
                    'Name': glue_table_name,
                    'StorageDescriptor': storage_descriptor,
                    'TableType': 'EXTERNAL_TABLE'
                }
            )
            logger.info(f"[DEBUG] Glue table created successfully: {glue_table_name}")

        base_timeout = 80
        additional_timeout_per_mb = 5
        dynamic_timeout = base_timeout + (file_size_mb * additional_timeout_per_mb)
        self._wait_for_table_creation(glue_table_name, dynamic_timeout)

    def _wait_for_table_creation(self, table_name, timeout):
        """
        Wait for Glue and Athena table creation to complete, ensuring compatibility.
        """
        import time
        glue_client = get_glue_client()
        start_time = time.time()
        glue_table_ready = False
        athena_table_ready = False

        logger.info(f"[DEBUG] Waiting for Glue table creation: {table_name}")
        while time.time() - start_time < timeout:
            try:
                glue_client.get_table(DatabaseName=ATHENA_SCHEMA_NAME, Name=table_name)
                logger.info(f"[DEBUG] Glue table is now available: {table_name}")
                glue_table_ready = True
                break
            except glue_client.exceptions.EntityNotFoundException:
                time.sleep(5)
            except Exception as e:
                logger.error(f"[ERROR] Unexpected error while checking Glue table availability: {e}")
                return False

        if not glue_table_ready:
            logger.error(f"[ERROR] Glue table '{table_name}' not available within {timeout} seconds.")
            return False

        logger.info(f"[DEBUG] Checking Athena table availability: {table_name}")
        while time.time() - start_time < timeout:
            try:
                query = f"SELECT 1 FROM {ATHENA_SCHEMA_NAME}.{table_name} LIMIT 1;"
                df = execute_sql_query(query)
                if df.empty:
                    logger.info(f"[DEBUG] Athena recognizes the table (no error), table ready: {table_name}")
                    athena_table_ready = True
                    break
                else:
                    logger.info(f"[DEBUG] Athena table ready with data: {table_name}")
                    athena_table_ready = True
                    break
            except Exception as e:
                error_message = str(e)
                if "TableNotFoundException" in error_message or "TABLE_NOT_FOUND" in error_message:
                    logger.info(f"[DEBUG] Still waiting for Athena to recognize table: {table_name}")
                    time.sleep(10)
                else:
                    logger.error(f"[ERROR] Unexpected error while checking Athena table availability: {e}")
                    return False

        if not athena_table_ready:
            logger.error(f"[ERROR] Table '{table_name}' not available in Athena within {timeout} seconds.")
            return False

        return True

    def _sanitize_identifier(self, name):
        """
        Sanitize identifiers to ensure SQL compatibility.
        """
        return re.sub(r'[^A-Za-z0-9_]+', '_', name.lower())
    







######## For Saving Prediction data prepared results#####
  

# import os
# import io
# import uuid
# import boto3
# import pandas as pd
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status, permissions, authentication
# from django.conf import settings
# from .models import PredictionFileInfo
# from django.contrib.auth.models import User

# class SavePredictionResultsView(APIView):
#     """
#     Accepts prediction query results, saves them as CSV files in S3, and updates PredictionFileInfo.
#     """
#     authentication_classes = [authentication.TokenAuthentication]
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request):
#         user_id = request.data.get('user_id')
#         chat_id = request.data.get('chat_id')
#         file_id = request.data.get('file_id')
#         cells = request.data.get('cells', [])

#         if not all([user_id, chat_id, file_id]):
#             return Response({"error": "Missing user_id, chat_id, or file_id"}, status=status.HTTP_400_BAD_REQUEST)

#         # S3 client setup
#         s3_client = boto3.client(
#             's3',
#             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#             region_name=os.getenv('AWS_S3_REGION_NAME')
#         )
#         bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')

#         saved_files = {}
#         errors = []

#         # Process each cell
#         for index, cell_info in enumerate(cells, start=1):
#             try:
#                 provided_id = cell_info.get('cellId')
#                 cell_id = str(provided_id) if provided_id is not None and f"cell{provided_id}" not in saved_files else str(index)

#                 columns_info = cell_info.get('columns', [])
#                 rows_data = cell_info.get('rows', [])
#                 query = cell_info.get('query', 'no_query')

#                 if not rows_data:
#                     rows_data = []

#                 col_order = [c['name'] for c in columns_info]
#                 df = pd.DataFrame(rows_data)
#                 if not df.empty:
#                     intersection = [c for c in col_order if c in df.columns]
#                     df = df[intersection]
#                 else:
#                     df = pd.DataFrame(columns=col_order)

#                 file_key = f"{user_id}/{chat_id}/prediction_cell_{cell_id}_{uuid.uuid4().hex[:6]}.csv"
#                 csv_buffer = io.StringIO()
#                 df.to_csv(csv_buffer, index=False)
#                 csv_buffer.seek(0)

#                 s3_client.put_object(
#                     Bucket=bucket_name,
#                     Key=f"prediction_saves/{file_key}",
#                     Body=csv_buffer.getvalue(),
#                     ContentType='text/csv'
#                 )

#                 s3_url = f"s3://{bucket_name}/prediction_saves/{file_key}"
#                 saved_files[f"cell{cell_id}"] = s3_url
#             except Exception as e:
#                 error_msg = f"Error processing cell {cell_info.get('cellId') or index}: {str(e)}"
#                 print("[SavePredictionResultsView] Exception:", error_msg)
#                 errors.append(error_msg)

#         # Update PredictionFileInfo
#         try:
#             user_instance = User.objects.get(pk=user_id)
#             prediction_file = PredictionFileInfo.objects.get(
#                 user=user_instance,
#                 chat=chat_id,
#                 file_id=file_id
#             )
#             prediction_file.prediction_s3_links = saved_files
#             prediction_file.save()
#         except User.DoesNotExist:
#             return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)
#         except PredictionFileInfo.DoesNotExist:
#             return Response({"error": "Prediction file not found."}, status=status.HTTP_404_NOT_FOUND)

#         response_data = {"message": "Prediction results saved successfully.", "files": saved_files}
#         if errors:
#             response_data["errors"] = errors

#         return Response(response_data, status=status.HTTP_200_OK)




import os
import io
import uuid
import boto3
import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions, authentication
from django.conf import settings
from .models import PredictionFileInfo
from django.contrib.auth.models import User

class SavePredictionResultsView(APIView):
    """
    Accepts prediction query results, saves them as CSV files in S3, and updates PredictionFileInfo.
    """
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        user_id = request.data.get('user_id')
        chat_id = request.data.get('chat_id')
        file_id = request.data.get('file_id')
        cells = request.data.get('cells', [])

        if not all([user_id, chat_id, file_id]):
            return Response({"error": "Missing user_id, chat_id, or file_id"}, status=status.HTTP_400_BAD_REQUEST)

        # S3 client setup
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_S3_REGION_NAME')
        )
        bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')

        saved_files = {}
        errors = []

        # Process each cell
        for index, cell_info in enumerate(cells, start=1):
            try:
                provided_id = cell_info.get('cellId')
                # Use provided_id if available and not already used; otherwise, fallback to loop index
                cell_id = str(provided_id) if provided_id is not None and f"cell{provided_id}" not in saved_files else str(index)

                columns_info = cell_info.get('columns', [])
                rows_data = cell_info.get('rows', [])
                query = cell_info.get('query', 'no_query')

                # Even if rows_data is empty, create an empty list.
                if not rows_data:
                    rows_data = []

                # Build expected column order from the provided columns info
                col_order = [c['name'] for c in columns_info]

                # Convert the cell’s rows data into a pandas DataFrame.
                df = pd.DataFrame(rows_data)
                if not df.empty and col_order:
                    # [FIXED]: Only filter if there is a non-empty intersection; otherwise, keep all columns.
                    intersection = [c for c in col_order if c in df.columns]
                    if intersection:
                        df = df[intersection]
                    else:
                        logger.warning(f"No matching columns found when filtering. Using all DataFrame columns: {df.columns.tolist()}")
                else:
                    # Create an empty DataFrame with the expected columns if rows_data is empty
                    df = pd.DataFrame(columns=col_order)

                # Generate a unique file key for this cell.
                file_key = f"{user_id}/{chat_id}/prediction_cell_{cell_id}_{uuid.uuid4().hex[:6]}.csv"

                # Write the DataFrame as CSV into an in-memory string buffer.
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                # Upload the CSV file to S3 under the "prediction_saves/" prefix.
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=f"prediction_saves/{file_key}",
                    Body=csv_buffer.getvalue(),
                    ContentType='text/csv'
                )

                # Build the full S3 URL.
                s3_url = f"s3://{bucket_name}/prediction_saves/{file_key}"
                saved_files[f"cell{cell_id}"] = s3_url
            except Exception as e:
                error_msg = f"Error processing cell {cell_info.get('cellId') or index}: {str(e)}"
                print("[SavePredictionResultsView] Exception:", error_msg)
                errors.append(error_msg)

        # Update PredictionFileInfo
        try:
            user_instance = User.objects.get(pk=user_id)
            prediction_file = PredictionFileInfo.objects.get(
                user=user_instance,
                chat=chat_id,
                file_id=file_id
            )
            prediction_file.prediction_s3_links = saved_files
            prediction_file.save()
        except User.DoesNotExist:
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)
        except PredictionFileInfo.DoesNotExist:
            return Response({"error": "Prediction file not found."}, status=status.HTTP_404_NOT_FOUND)

        response_data = {"message": "Prediction results saved successfully.", "files": saved_files}
        if errors:
            response_data["errors"] = errors

        return Response(response_data, status=status.HTTP_200_OK)
