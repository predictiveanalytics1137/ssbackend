

# # # #Sql_Notebook/views.py

# # # import os
# # # import time
# # # from rest_framework.views import APIView
# # # from rest_framework.response import Response
# # # from rest_framework import status, permissions, authentication
# # # import boto3
# # # from botocore.exceptions import ClientError
# # # from .serializers import SQLQuerySerializer

# # # class ExecuteSQLView(APIView):
# # #     """
# # #     API view to execute SQL SELECT queries using AWS Athena and return results with data types.
# # #     """
# # #     authentication_classes = [authentication.TokenAuthentication]
# # #     permission_classes = [permissions.IsAuthenticated]

# # #     def post(self, request):
# # #         serializer = SQLQuerySerializer(data=request.data)
# # #         if serializer.is_valid():
# # #             query = serializer.validated_data['query'].strip()
            
# # #             # Ensure only SELECT queries are allowed
# # #             if not query.lower().startswith('select'):
# # #                 print("Attempted non-SELECT query.")
# # #                 return Response({'error': 'Only SELECT queries are allowed.'}, status=status.HTTP_400_BAD_REQUEST)

# # #             # Set up Athena client using environment variables for security
# # #             athena = boto3.client(
# # #                 'athena',
# # #                 aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
# # #                 aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
# # #                 region_name=os.getenv('AWS_S3_REGION_NAME')
# # #             )

# # #             # Athena configuration
# # #             DATABASE = 'pa_user_datafiles_db'  # Your Glue database name
# # #             OUTPUT_LOCATION = 's3://pa-documents-storage-bucket/query-results/'  # Your S3 output location

# # #             try:
# # #                 # Start query execution
# # #                 response = athena.start_query_execution(
# # #                     QueryString=query,
# # #                     QueryExecutionContext={'Database': DATABASE},
# # #                     ResultConfiguration={'OutputLocation': OUTPUT_LOCATION}
# # #                 )
# # #                 query_execution_id = response['QueryExecutionId']
# # #                 # print(f"Query Execution ID: {query_execution_id}")  # Debug log for query execution ID

# # #                 # Polling to check query status
# # #                 max_execution = 30  # Maximum number of polling attempts
# # #                 execution_count = 0

# # #                 while True:
# # #                     execution_response = athena.get_query_execution(QueryExecutionId=query_execution_id)
# # #                     query_status = execution_response['QueryExecution']['Status']['State']
# # #                     # print(f"Query Status: {query_status}")  # Debug log for query status

# # #                     if query_status == 'SUCCEEDED':
# # #                         print("Query succeeded.")
# # #                         break
# # #                     elif query_status in ['FAILED', 'CANCELLED']:
# # #                         reason = execution_response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
# # #                         print(f"Query Failed: {reason}")  # Debug log for query failure reason
# # #                         return Response({'error': f'Query failed or was cancelled: {reason}'}, status=status.HTTP_400_BAD_REQUEST)
# # #                     else:
# # #                         time.sleep(1)  # Wait for 1 second before next poll
# # #                         execution_count += 1
# # #                         if execution_count >= max_execution:
# # #                             # print("Query timed out.")  # Debug log for timeout
# # #                             return Response({'error': 'Query timed out'}, status=status.HTTP_408_REQUEST_TIMEOUT)

# # #                 # Retrieve query results with pagination
# # #                 result_paginator = athena.get_paginator('get_query_results')
# # #                 result_iter = result_paginator.paginate(QueryExecutionId=query_execution_id)

# # #                 columns = []  # List to hold column metadata
# # #                 rows = []     # List to hold row data
# # #                 first_page = True

# # #                 for results_page in result_iter:
# # #                     if first_page:
# # #                         # Extract column names and data types from the first page
# # #                         column_info = results_page['ResultSet']['ResultSetMetadata']['ColumnInfo']
# # #                         columns = [{"name": col['Name'], "type": self.map_data_type(col['Type'])} for col in column_info]
# # #                         print(f"Columns: {columns}")  # Debug log for columns
# # #                         # Skip header row which contains column names
# # #                         result_rows = results_page['ResultSet']['Rows'][1:]
# # #                         first_page = False
# # #                     else:
# # #                         # Subsequent pages contain only data rows
# # #                         result_rows = results_page['ResultSet']['Rows']

# # #                     for row in result_rows:
# # #                         data_dict = {}
# # #                         data = row.get('Data', [])
# # #                         for idx, column in enumerate(columns):
# # #                             if idx < len(data):
# # #                                 datum = data[idx]
# # #                                 value = datum.get('VarCharValue', None)
# # #                             else:
# # #                                 value = None
# # #                             data_dict[column['name']] = value
# # #                         rows.append(data_dict)

# # #                 print(f"Number of rows fetched: {len(rows)}")  # Debug log for row count

# # #                 # Prepare the response data including data types
# # #                 data = {
# # #                     'columns': columns,  # List of dictionaries with 'name' and 'type'
# # #                     'rows': rows,        # List of row data as dictionaries
# # #                 }
# # #                 return Response(data, status=status.HTTP_200_OK)

# # #             except ClientError as e:
# # #                 print(f"Athena ClientError: {e}")  # Debug log for Athena client errors
# # #                 return Response({'error': f'Athena error: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
# # #             except Exception as e:
# # #                 print(f"Unexpected error: {e}")  # Debug log for unexpected errors
# # #                 return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# # #         else:
# # #             print(f"Invalid query payload: {serializer.errors}")  # Debug log for invalid payload
# # #             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # #     def map_data_type(self, athena_type):
# # #         """
# # #         Maps Athena data types to desired display types.
# # #         Specifically maps 'Varchar' to 'String'.
# # #         """
# # #         type_mapping = {
# # #             'varchar': 'String',
# # #             'boolean': 'Boolean',
# # #             'bigint': 'BigInt',
# # #             'integer': 'Integer',
# # #             'double': 'Double',
# # #             'float': 'Float',
# # #             'date': 'Date',
# # #             'timestamp': 'Timestamp',
# # #             # Add more mappings as needed
# # #         }
# # #         # Normalize the athena_type to lowercase for mapping
# # #         return type_mapping.get(athena_type.lower(), athena_type)




# # import os
# # import time
# # import re
# # import boto3
# # from botocore.exceptions import ClientError
# # from rest_framework.views import APIView
# # from rest_framework.response import Response
# # from rest_framework import status, permissions, authentication
# # from .serializers import SQLQuerySerializer

# # class ExecuteSQLView(APIView):
# #     """
# #     API view to execute SQL SELECT queries (including ones that begin with 'WITH')
# #     using AWS Athena and return results with data types.
# #     """
# #     authentication_classes = [authentication.TokenAuthentication]
# #     permission_classes = [permissions.IsAuthenticated]

# #     def post(self, request):
# #         serializer = SQLQuerySerializer(data=request.data)
# #         if serializer.is_valid():
# #             # Retrieve the user query and strip leading/trailing whitespace
# #             query = serializer.validated_data['query'].strip()
# #             query_lower = query.lower()

# #             # -----------------------------------------------
# #             # ALLOW queries that begin with 'SELECT' or 'WITH'
# #             # -----------------------------------------------
# #             # Also, we want to disallow destructive statements (INSERT, DROP, etc.).
# #             # We'll do a quick check for that as well.
# #             allowed_starts = ("select", "with")
# #             # Remove leading whitespace (in case of newlines etc.)
# #             stripped_query_lower = query_lower.lstrip()

# #             if not any(stripped_query_lower.startswith(start) for start in allowed_starts):
# #                 print("Attempted query that doesn't start with SELECT or WITH.")
# #                 return Response(
# #                     {'error': 'Only read-only queries (SELECT or WITH) are allowed.'},
# #                     status=status.HTTP_400_BAD_REQUEST
# #                 )

# #             # Optionally, block DDL/DML keywords to ensure read-only:
# #             disallowed_keywords = ("insert", "update", "delete", "alter", "drop", "create", "truncate")
# #             if any(k in stripped_query_lower for k in disallowed_keywords):
# #                 print("Query contains a disallowed keyword for this read-only endpoint.")
# #                 return Response(
# #                     {'error': 'Only read-only queries (SELECT/WITH) are allowed.'},
# #                     status=status.HTTP_400_BAD_REQUEST
# #                 )

# #             # -----------------------------------------------
# #             # Continue with the original Athena logic
# #             # -----------------------------------------------
# #             athena = boto3.client(
# #                 'athena',
# #                 aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
# #                 aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
# #                 region_name=os.getenv('AWS_S3_REGION_NAME')
# #             )

# #             DATABASE = 'pa_user_datafiles_db'  # Your Glue database name
# #             OUTPUT_LOCATION = 's3://pa-documents-storage-bucket/query-results/'  # Your S3 output location

# #             try:
# #                 # Start query execution
# #                 response = athena.start_query_execution(
# #                     QueryString=query,
# #                     QueryExecutionContext={'Database': DATABASE},
# #                     ResultConfiguration={'OutputLocation': OUTPUT_LOCATION}
# #                 )
# #                 query_execution_id = response['QueryExecutionId']

# #                 # Polling to check query status
# #                 max_execution = 30  # Maximum number of polling attempts
# #                 execution_count = 0

# #                 while True:
# #                     execution_response = athena.get_query_execution(QueryExecutionId=query_execution_id)
# #                     query_status = execution_response['QueryExecution']['Status']['State']

# #                     if query_status == 'SUCCEEDED':
# #                         print("Query succeeded.")
# #                         break
# #                     elif query_status in ['FAILED', 'CANCELLED']:
# #                         reason = execution_response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
# #                         print(f"Query Failed: {reason}")
# #                         return Response(
# #                             {'error': f'Query failed or was cancelled: {reason}'},
# #                             status=status.HTTP_400_BAD_REQUEST
# #                         )
# #                     else:
# #                         time.sleep(1)
# #                         execution_count += 1
# #                         if execution_count >= max_execution:
# #                             return Response({'error': 'Query timed out'}, status=status.HTTP_408_REQUEST_TIMEOUT)

# #                 # Retrieve query results with pagination
# #                 result_paginator = athena.get_paginator('get_query_results')
# #                 result_iter = result_paginator.paginate(QueryExecutionId=query_execution_id)

# #                 columns = []
# #                 rows = []
# #                 first_page = True

# #                 for results_page in result_iter:
# #                     if first_page:
# #                         # Extract column metadata from the first page
# #                         column_info = results_page['ResultSet']['ResultSetMetadata']['ColumnInfo']
# #                         columns = [
# #                             {"name": col['Name'], "type": self.map_data_type(col['Type'])}
# #                             for col in column_info
# #                         ]
# #                         print(f"Columns: {columns}")
# #                         # Skip the header row in the first page (contains column names)
# #                         result_rows = results_page['ResultSet']['Rows'][1:]
# #                         first_page = False
# #                     else:
# #                         # Subsequent pages have only data rows
# #                         result_rows = results_page['ResultSet']['Rows']

# #                     for row in result_rows:
# #                         data_dict = {}
# #                         data = row.get('Data', [])
# #                         for idx, column in enumerate(columns):
# #                             if idx < len(data):
# #                                 datum = data[idx]
# #                                 value = datum.get('VarCharValue', None)
# #                             else:
# #                                 value = None
# #                             data_dict[column['name']] = value
# #                         rows.append(data_dict)

# #                 print(f"Number of rows fetched: {len(rows)}")

# #                 # Prepare the final response
# #                 data = {
# #                     'columns': columns,  # each item has 'name', 'type'
# #                     'rows': rows,        # list of row dicts
# #                 }
# #                 return Response(data, status=status.HTTP_200_OK)

# #             except ClientError as e:
# #                 print(f"Athena ClientError: {e}")
# #                 return Response({'error': f'Athena error: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
# #             except Exception as e:
# #                 print(f"Unexpected error: {e}")
# #                 return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

# #         else:
# #             # If serializer is invalid, log and return the validation errors
# #             print(f"Invalid query payload: {serializer.errors}")
# #             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# #     def map_data_type(self, athena_type):
# #         """
# #         Maps Athena data types to desired display types.
# #         Specifically maps 'Varchar' to 'String'.
# #         """
# #         type_mapping = {
# #             'varchar': 'String',
# #             'boolean': 'Boolean',
# #             'bigint': 'BigInt',
# #             'integer': 'Integer',
# #             'double': 'Double',
# #             'float': 'Float',
# #             'date': 'Date',
# #             'timestamp': 'Timestamp',
# #         }
# #         # Normalize the athena_type to lowercase for mapping
# #         return type_mapping.get(athena_type.lower(), athena_type)




# import os
# import time
# import re
# import boto3
# from botocore.exceptions import ClientError
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status, permissions, authentication
# from .serializers import SQLQuerySerializer

# class ExecuteSQLView(APIView):
#     """
#     API view to execute SQL SELECT queries (including ones that begin with 'WITH')
#     using AWS Athena and return results with data types.
#     """
#     authentication_classes = [authentication.TokenAuthentication]
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request):
#         serializer = SQLQuerySerializer(data=request.data)
#         if serializer.is_valid():
#             # 1) Get the raw query and break it into lines
#             raw_query = serializer.validated_data['query']
#             query_lines = raw_query.split('\n')

#             # 2) Remove leading lines that are either empty or comments (like "-- comment")
#             cleaned_lines = []
#             found_non_comment = False
#             for line in query_lines:
#                 stripped_line = line.strip()
#                 # If we haven't found a non-comment line yet, skip blank or comment lines
#                 if not found_non_comment:
#                     if (stripped_line.startswith('--')  # single-line comment
#                         or stripped_line.startswith('#')  # another style
#                         or stripped_line == ''):
#                         # skip this line
#                         continue
#                     else:
#                         found_non_comment = True
#                         cleaned_lines.append(line)
#                 else:
#                     # Already found non-comment lines, so keep the rest
#                     cleaned_lines.append(line)

#             # 3) Rebuild the query from cleaned lines and lowercase for checking
#             cleaned_query = '\n'.join(cleaned_lines).strip()
#             query_lower = cleaned_query.lower()

#             # 4) Check whether the cleaned query starts with SELECT or WITH
#             allowed_starts = ("select", "with")
#             if not any(query_lower.startswith(start) for start in allowed_starts):
#                 print("Attempted query that doesn't start with SELECT or WITH.")
#                 return Response(
#                     {'error': 'Only read-only queries (SELECT or WITH) are allowed.'},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )

#             # 5) Optionally disallow DDL/DML keywords (insert, update, etc.)
#             disallowed_keywords = ("insert", "update", "delete", "alter", "drop", "create", "truncate")
#             if any(k in query_lower for k in disallowed_keywords):
#                 print("Query contains a disallowed keyword for this read-only endpoint.")
#                 return Response(
#                     {'error': 'Only read-only queries (SELECT/WITH) are allowed.'},
#                     status=status.HTTP_400_BAD_REQUEST
#                 )

#             # 6) Now proceed with Athena logic using 'cleaned_query'
#             athena = boto3.client(
#                 'athena',
#                 aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#                 aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#                 region_name=os.getenv('AWS_S3_REGION_NAME')
#             )

#             DATABASE = 'pa_user_datafiles_db'
#             OUTPUT_LOCATION = 's3://pa-documents-storage-bucket/query-results/'

#             try:
#                 # Start query execution
#                 response = athena.start_query_execution(
#                     QueryString=cleaned_query,
#                     QueryExecutionContext={'Database': DATABASE},
#                     ResultConfiguration={'OutputLocation': OUTPUT_LOCATION}
#                 )
#                 query_execution_id = response['QueryExecutionId']

#                 # Poll for completion
#                 max_execution = 30
#                 execution_count = 0
#                 while True:
#                     execution_response = athena.get_query_execution(QueryExecutionId=query_execution_id)
#                     query_status = execution_response['QueryExecution']['Status']['State']

#                     if query_status == 'SUCCEEDED':
#                         print("Query succeeded.")
#                         break
#                     elif query_status in ['FAILED', 'CANCELLED']:
#                         reason = execution_response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
#                         print(f"Query Failed: {reason}")
#                         return Response(
#                             {'error': f'Query failed or was cancelled: {reason}'},
#                             status=status.HTTP_400_BAD_REQUEST
#                         )
#                     else:
#                         time.sleep(1)
#                         execution_count += 1
#                         if execution_count >= max_execution:
#                             return Response({'error': 'Query timed out'}, status=status.HTTP_408_REQUEST_TIMEOUT)

#                 # Paginate results
#                 result_paginator = athena.get_paginator('get_query_results')
#                 result_iter = result_paginator.paginate(QueryExecutionId=query_execution_id)

#                 columns = []
#                 rows = []
#                 first_page = True

#                 for results_page in result_iter:
#                     if first_page:
#                         # Column metadata
#                         column_info = results_page['ResultSet']['ResultSetMetadata']['ColumnInfo']
#                         columns = [
#                             {"name": col['Name'], "type": self.map_data_type(col['Type'])}
#                             for col in column_info
#                         ]
#                         print(f"Columns: {columns}")
#                         # Skip header row in first page
#                         result_rows = results_page['ResultSet']['Rows'][1:]
#                         first_page = False
#                     else:
#                         # Only data rows
#                         result_rows = results_page['ResultSet']['Rows']

#                     for row in result_rows:
#                         data_dict = {}
#                         data = row.get('Data', [])
#                         for idx, column in enumerate(columns):
#                             value = None
#                             if idx < len(data):
#                                 datum = data[idx]
#                                 value = datum.get('VarCharValue', None)
#                             data_dict[column['name']] = value
#                         rows.append(data_dict)

#                 print(f"Number of rows fetched: {len(rows)}")

#                 return Response({'columns': columns, 'rows': rows}, status=status.HTTP_200_OK)

#             except ClientError as e:
#                 print(f"Athena ClientError: {e}")
#                 return Response({'error': f'Athena error: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
#             except Exception as e:
#                 print(f"Unexpected error: {e}")
#                 return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

#         else:
#             # If serializer is invalid
#             print(f"Invalid query payload: {serializer.errors}")
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#     def map_data_type(self, athena_type):
#         """
#         Maps Athena data types to more user-friendly strings.
#         """
#         type_mapping = {
#             'varchar': 'String',
#             'boolean': 'Boolean',
#             'bigint': 'BigInt',
#             'integer': 'Integer',
#             'double': 'Double',
#             'float': 'Float',
#             'date': 'Date',
#             'timestamp': 'Timestamp',
#         }
#         return type_mapping.get(athena_type.lower(), athena_type)






import os
import io
import time
import re
import uuid
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions, authentication
from .serializers import SQLQuerySerializer
from django.conf import settings

class ExecuteSQLView(APIView):
    """
    API view to execute SQL SELECT queries (including ones that begin with 'WITH')
    using AWS Athena and return results with data types.
    """
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = SQLQuerySerializer(data=request.data)
        if serializer.is_valid():
            raw_query = serializer.validated_data['query']
            query_lines = raw_query.split('\n')

            # Remove leading blank or comment lines
            cleaned_lines = []
            found_non_comment = False
            for line in query_lines:
                stripped_line = line.strip()
                if not found_non_comment:
                    if (stripped_line.startswith('--')
                        or stripped_line.startswith('#')
                        or stripped_line == ''):
                        continue
                    else:
                        found_non_comment = True
                        cleaned_lines.append(line)
                else:
                    cleaned_lines.append(line)

            cleaned_query = '\n'.join(cleaned_lines).strip()
            query_lower = cleaned_query.lower()

            allowed_starts = ("select", "with")
            if not any(query_lower.startswith(start) for start in allowed_starts):
                print("Attempted query that doesn't start with SELECT or WITH.")
                return Response(
                    {'error': 'Only read-only queries (SELECT or WITH) are allowed.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            disallowed_keywords = ("insert", "update", "delete", "alter", "drop", "create", "truncate")
            if any(k in query_lower for k in disallowed_keywords):
                print("Query contains a disallowed keyword for this read-only endpoint.")
                return Response(
                    {'error': 'Only read-only queries (SELECT/WITH) are allowed.'},
                    status=status.HTTP_400_BAD_REQUEST
                )

            athena = boto3.client(
                'athena',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_S3_REGION_NAME')
            )

            DATABASE = 'pa_user_datafiles_db'
            OUTPUT_LOCATION = 's3://pa-documents-storage-bucket/query-results/'

            try:
                response = athena.start_query_execution(
                    QueryString=cleaned_query,
                    QueryExecutionContext={'Database': DATABASE},
                    ResultConfiguration={'OutputLocation': OUTPUT_LOCATION}
                )
                query_execution_id = response['QueryExecutionId']

                max_execution = 30
                execution_count = 0
                while True:
                    execution_response = athena.get_query_execution(QueryExecutionId=query_execution_id)
                    query_status = execution_response['QueryExecution']['Status']['State']

                    if query_status == 'SUCCEEDED':
                        print("Query succeeded.")
                        break
                    elif query_status in ['FAILED', 'CANCELLED']:
                        reason = execution_response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                        print(f"Query Failed: {reason}")
                        return Response(
                            {'error': f'Query failed or was cancelled: {reason}'},
                            status=status.HTTP_400_BAD_REQUEST
                        )
                    else:
                        time.sleep(1)
                        execution_count += 1
                        if execution_count >= max_execution:
                            return Response({'error': 'Query timed out'}, status=status.HTTP_408_REQUEST_TIMEOUT)

                # Retrieve results
                result_paginator = athena.get_paginator('get_query_results')
                result_iter = result_paginator.paginate(QueryExecutionId=query_execution_id)

                columns = []
                rows = []
                first_page = True

                for results_page in result_iter:
                    if first_page:
                        column_info = results_page['ResultSet']['ResultSetMetadata']['ColumnInfo']
                        columns = [
                            {"name": col['Name'], "type": self.map_data_type(col['Type'])}
                            for col in column_info
                        ]
                        print(f"Columns: {columns}")
                        result_rows = results_page['ResultSet']['Rows'][1:]
                        first_page = False
                    else:
                        result_rows = results_page['ResultSet']['Rows']

                    for row in result_rows:
                        data_dict = {}
                        data = row.get('Data', [])
                        for idx, column in enumerate(columns):
                            value = None
                            if idx < len(data):
                                datum = data[idx]
                                value = datum.get('VarCharValue', None)
                            data_dict[column['name']] = value
                        rows.append(data_dict)

                print(f"Number of rows fetched: {len(rows)}")

                return Response({'columns': columns, 'rows': rows}, status=status.HTTP_200_OK)

            except ClientError as e:
                print(f"Athena ClientError: {e}")
                return Response({'error': f'Athena error: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                print(f"Unexpected error: {e}")
                return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

        else:
            print(f"Invalid query payload: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def map_data_type(self, athena_type):
        type_mapping = {
            'varchar': 'String',
            'boolean': 'Boolean',
            'bigint': 'BigInt',
            'integer': 'Integer',
            'double': 'Double',
            'float': 'Float',
            'date': 'Date',
            'timestamp': 'Timestamp',
        }
        return type_mapping.get(athena_type.lower(), athena_type)


# class SaveNotebooksView(APIView):
#     """
#     Accepts aggregated query results from the front end, merges (or stores) them as CSV, and uploads to S3.
#     """
#     authentication_classes = [authentication.TokenAuthentication]
#     permission_classes = [permissions.IsAuthenticated]

#     def post(self, request):
#         """
#         Expects JSON in the form:
#         {
#           "user_id": "<user>",
#           "chat_id": "<chat>",
#           "cells": [
#             {
#               "cellId": 1,
#               "query": "...",
#               "columns": [...],
#               "rows": [...]
#             },
#             ...
#           ]
#         }
#         """
#         user_id = request.data.get('user_id')
#         chat_id = request.data.get('chat_id')
#         cells = request.data.get('cells', [])

#         if not user_id or not chat_id:
#             return Response({"error": "Missing user_id or chat_id"}, status=status.HTTP_400_BAD_REQUEST)

#         # # Create an S3 client
#         # s3_client = boto3.client(
#         #     's3',
#         #     aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
#         #     aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
#         #     region_name=settings.AWS_S3_REGION_NAME
#         # )
#         # bucket_name = settings.AWS_STORAGE_BUCKET_NAME

#         s3_client = boto3.client(
#                 's3',
#                 aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#                 aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#                 region_name=os.getenv('AWS_S3_REGION_NAME')
#             )
#         bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')

#         saved_files = []
#         try:
#             # You can store each cell's result in its own CSV,
#             # or combine them. We'll store each cell separately here.
#             for cell_info in cells:
#                 cell_id = cell_info.get('cellId')
#                 columns_info = cell_info.get('columns', [])
#                 rows_data = cell_info.get('rows', [])
#                 query = cell_info.get('query', 'no_query')

#                 # Convert rows_data to DataFrame
#                 # columns_info is like [{'name': 'colA', 'type': 'String'}, ...]
#                 col_order = [c['name'] for c in columns_info]
#                 df = pd.DataFrame(rows_data)
#                 # We only keep the columns that actually exist
#                 intersection = [c for c in col_order if c in df.columns]
#                 df = df[intersection]

#                 # Make a unique CSV name
#                 # file_key = f"{user_id}_{chat_id}_cell_{cell_id}_{uuid.uuid4().hex[:6]}.csv"

#                 file_key = f"{user_id}/{chat_id}/cell_{cell_id}_{uuid.uuid4().hex[:6]}.csv"

#                 # Save to CSV in memory
#                 csv_buffer = io.StringIO()
#                 df.to_csv(csv_buffer, index=False)
#                 csv_buffer.seek(0)

#                 s3_client.put_object(
#                     Bucket=bucket_name,
#                     Key=f"notebook_saves/{file_key}",
#                     Body=csv_buffer.getvalue(),
#                     ContentType='text/csv'
#                 )

#                 saved_files.append(file_key)

#             return Response({"message": "Notebooks saved successfully.", "files": saved_files}, status=status.HTTP_200_OK)

#         except Exception as e:
#             print("[SaveNotebooksView] Exception:", e)
#             return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)




import os
import io
import uuid
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions, authentication
from django.conf import settings

# Import the Notebook model from your chat app.
from chat.models import Notebook
from django.contrib.auth.models import User

class SaveNotebooksView(APIView):
    """
    This endpoint accepts aggregated query results for each notebook cell from the front end,
    converts each cell’s data to a CSV file, uploads it to S3, and saves a mapping of cell identifiers
    to their S3 URLs in the Notebook model.
    
    Expected JSON payload:
    {
      "user_id": "<user>",
      "chat_id": "<chat>",
      "cells": [
        {
          "cellId": 1,        // optional; if missing or duplicate, a unique index is used
          "query": "...",
          "columns": [{"name": "colA", "type": "String"}, ...],
          "rows": [{...}, {...}, ...]
        },
        {
          "cellId": 2,
          "query": "...",
          "columns": [...],
          "rows": [...]
        },
        ...
      ]
    }
    
    This implementation dynamically processes every cell in the payload.
    """
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        user_id = request.data.get('user_id')
        chat_id = request.data.get('chat_id')
        cells = request.data.get('cells', [])

        if not user_id or not chat_id:
            return Response({"error": "Missing user_id or chat_id"}, status=status.HTTP_400_BAD_REQUEST)

        # Create an S3 client (using environment variables).
        s3_client = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_S3_REGION_NAME')
        )
        bucket_name = os.getenv('AWS_STORAGE_BUCKET_NAME')

        saved_files = {}  # This dictionary will hold mapping like {"cell1": "s3://bucket/...csv", ...}
        errors = []       # Collect errors for individual cells if needed.

        # Process every cell in the payload.
        for index, cell_info in enumerate(cells, start=1):
            try:
                # Get the provided cellId if available.
                provided_id = cell_info.get('cellId')
                # Use provided_id if it exists and if it hasn’t been used already;
                # otherwise, fall back to the current loop index.
                if provided_id is not None and f"cell{provided_id}" not in saved_files:
                    cell_id = str(provided_id)
                else:
                    cell_id = str(index)

                columns_info = cell_info.get('columns', [])
                rows_data = cell_info.get('rows', [])
                query = cell_info.get('query', 'no_query')

                # Even if rows_data is empty, create an empty list.
                if not rows_data:
                    rows_data = []

                # Convert the cell’s rows data into a pandas DataFrame.
                col_order = [c['name'] for c in columns_info]
                df = pd.DataFrame(rows_data)
                if not df.empty:
                    # Keep only the columns that exist in the DataFrame.
                    intersection = [c for c in col_order if c in df.columns]
                    df = df[intersection]
                else:
                    # Create an empty DataFrame with the expected columns.
                    df = pd.DataFrame(columns=col_order)

                # Generate a unique file key for this cell.
                file_key = f"{user_id}/{chat_id}/cell_{cell_id}_{uuid.uuid4().hex[:6]}.csv"

                # Write the DataFrame as CSV into an in-memory string buffer.
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                csv_buffer.seek(0)

                # Upload the CSV file to S3 under the "notebook_saves/" prefix.
                s3_client.put_object(
                    Bucket=bucket_name,
                    Key=f"notebook_saves/{file_key}",
                    Body=csv_buffer.getvalue(),
                    ContentType='text/csv'
                )

                # Build the full S3 URL.
                s3_url = f"s3://{bucket_name}/notebook_saves/{file_key}"
                # Save the URL using a key that is guaranteed to be unique.
                saved_files[f"cell{cell_id}"] = s3_url

            except Exception as e:
                error_msg = f"Error processing cell {cell_info.get('cellId') or index}: {str(e)}"
                print("[SaveNotebooksView] Exception:", error_msg)
                errors.append(error_msg)
                # Continue processing remaining cells.

        # Get the user instance.
        try:
            user_instance = User.objects.get(pk=user_id)
        except User.DoesNotExist:
            return Response({"error": "User not found."}, status=status.HTTP_404_NOT_FOUND)

        # Retrieve Notebook records for this user and chat.
        notebooks = Notebook.objects.filter(user=user_instance, chat=chat_id)
        if notebooks.exists():
            # If multiple exist, choose the most recent one.
            notebook = notebooks.order_by('-created_at').first()
            notebook.cell_s3_links = saved_files
            notebook.save()
        else:
            # Create a new Notebook record with the S3 mapping.
            notebook = Notebook.objects.create(
                user=user_instance,
                chat=chat_id,
                entity_column='',
                target_column='',
                features={},
                file_url='',
                notebook_json={},
                cell_s3_links=saved_files
            )

        response_data = {"message": "Notebooks saved successfully.", "files": saved_files}
        if errors:
            response_data["errors"] = errors

        return Response(response_data, status=status.HTTP_200_OK)
