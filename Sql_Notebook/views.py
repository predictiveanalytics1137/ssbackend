

#Sql_Notebook/views.py

import os
import time
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions, authentication
import boto3
from botocore.exceptions import ClientError
from .serializers import SQLQuerySerializer

class ExecuteSQLView(APIView):
    """
    API view to execute SQL SELECT queries using AWS Athena and return results with data types.
    """
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = SQLQuerySerializer(data=request.data)
        if serializer.is_valid():
            query = serializer.validated_data['query'].strip()
            
            # Ensure only SELECT queries are allowed
            if not query.lower().startswith('select'):
                print("Attempted non-SELECT query.")
                return Response({'error': 'Only SELECT queries are allowed.'}, status=status.HTTP_400_BAD_REQUEST)

            # Set up Athena client using environment variables for security
            athena = boto3.client(
                'athena',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
                region_name=os.getenv('AWS_S3_REGION_NAME')
            )

            # Athena configuration
            DATABASE = 'pa_user_datafiles_db'  # Your Glue database name
            OUTPUT_LOCATION = 's3://pa-documents-storage-bucket/query-results/'  # Your S3 output location

            try:
                # Start query execution
                response = athena.start_query_execution(
                    QueryString=query,
                    QueryExecutionContext={'Database': DATABASE},
                    ResultConfiguration={'OutputLocation': OUTPUT_LOCATION}
                )
                query_execution_id = response['QueryExecutionId']
                # print(f"Query Execution ID: {query_execution_id}")  # Debug log for query execution ID

                # Polling to check query status
                max_execution = 30  # Maximum number of polling attempts
                execution_count = 0

                while True:
                    execution_response = athena.get_query_execution(QueryExecutionId=query_execution_id)
                    query_status = execution_response['QueryExecution']['Status']['State']
                    # print(f"Query Status: {query_status}")  # Debug log for query status

                    if query_status == 'SUCCEEDED':
                        print("Query succeeded.")
                        break
                    elif query_status in ['FAILED', 'CANCELLED']:
                        reason = execution_response['QueryExecution']['Status'].get('StateChangeReason', 'Unknown error')
                        print(f"Query Failed: {reason}")  # Debug log for query failure reason
                        return Response({'error': f'Query failed or was cancelled: {reason}'}, status=status.HTTP_400_BAD_REQUEST)
                    else:
                        time.sleep(1)  # Wait for 1 second before next poll
                        execution_count += 1
                        if execution_count >= max_execution:
                            # print("Query timed out.")  # Debug log for timeout
                            return Response({'error': 'Query timed out'}, status=status.HTTP_408_REQUEST_TIMEOUT)

                # Retrieve query results with pagination
                result_paginator = athena.get_paginator('get_query_results')
                result_iter = result_paginator.paginate(QueryExecutionId=query_execution_id)

                columns = []  # List to hold column metadata
                rows = []     # List to hold row data
                first_page = True

                for results_page in result_iter:
                    if first_page:
                        # Extract column names and data types from the first page
                        column_info = results_page['ResultSet']['ResultSetMetadata']['ColumnInfo']
                        columns = [{"name": col['Name'], "type": self.map_data_type(col['Type'])} for col in column_info]
                        print(f"Columns: {columns}")  # Debug log for columns
                        # Skip header row which contains column names
                        result_rows = results_page['ResultSet']['Rows'][1:]
                        first_page = False
                    else:
                        # Subsequent pages contain only data rows
                        result_rows = results_page['ResultSet']['Rows']

                    for row in result_rows:
                        data_dict = {}
                        data = row.get('Data', [])
                        for idx, column in enumerate(columns):
                            if idx < len(data):
                                datum = data[idx]
                                value = datum.get('VarCharValue', None)
                            else:
                                value = None
                            data_dict[column['name']] = value
                        rows.append(data_dict)

                print(f"Number of rows fetched: {len(rows)}")  # Debug log for row count

                # Prepare the response data including data types
                data = {
                    'columns': columns,  # List of dictionaries with 'name' and 'type'
                    'rows': rows,        # List of row data as dictionaries
                }
                return Response(data, status=status.HTTP_200_OK)

            except ClientError as e:
                print(f"Athena ClientError: {e}")  # Debug log for Athena client errors
                return Response({'error': f'Athena error: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)
            except Exception as e:
                print(f"Unexpected error: {e}")  # Debug log for unexpected errors
                return Response({'error': f'File processing failed: {str(e)}'}, status=status.HTTP_400_BAD_REQUEST)

        else:
            print(f"Invalid query payload: {serializer.errors}")  # Debug log for invalid payload
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

    def map_data_type(self, athena_type):
        """
        Maps Athena data types to desired display types.
        Specifically maps 'Varchar' to 'String'.
        """
        type_mapping = {
            'varchar': 'String',
            'boolean': 'Boolean',
            'bigint': 'BigInt',
            'integer': 'Integer',
            'double': 'Double',
            'float': 'Float',
            'date': 'Date',
            'timestamp': 'Timestamp',
            # Add more mappings as needed
        }
        # Normalize the athena_type to lowercase for mapping
        return type_mapping.get(athena_type.lower(), athena_type)
