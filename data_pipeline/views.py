


# # import os
# # from rest_framework.views import APIView
# # from rest_framework.response import Response
# # from rest_framework import status
# # from sqlalchemy import create_engine, text  # Import text for raw SQL queries
# # import boto3

# # # AWS and Athena Configuration
# # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')
# # AWS_REGION_NAME = AWS_S3_REGION_NAME
# # ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Athena database name

# # class DataForAutomationAPI(APIView):
# #     """
# #     Endpoint: /api/data-for-automation/
# #     Fetch data dynamically based on user-confirmed schema and pass it to the automation pipeline.
# #     """

# #     def post(self, request):
# #         """
# #         Handles POST requests to fetch data for automation based on user-confirmed schema.
# #         """
# #         print("[DEBUG] Received POST request at DataForAutomationAPI.")
# #         print("[DEBUG] Request data:", request.data)

# #         # Extract data from the request
# #         entity_column = request.data.get("entity_column")
# #         target_column = request.data.get("target_column")
# #         feature_columns = request.data.get("features", [])
# #         glue_table_name = request.data.get("glue_table_name")

# #         # Validate request data
# #         if not all([entity_column, target_column, feature_columns, glue_table_name]):
# #             print("[ERROR] Missing required data:", {
# #                 "entity_column": entity_column,
# #                 "target_column": target_column,
# #                 "features": feature_columns,
# #                 "glue_table_name": glue_table_name,
# #             })
# #             return Response(
# #                 {"error": "Missing required data: entity_column, target_column, features, glue_table_name"},
# #                 status=status.HTTP_400_BAD_REQUEST,
# #             )

# #         # SQL queries with LIMIT 10 for presentation purposes
# #         entity_target_query = (
# #             f"SELECT {entity_column}, {target_column} FROM {ATHENA_SCHEMA_NAME}.{glue_table_name} LIMIT 12;"
# #         )
# #         features_query = (
# #             f"SELECT {', '.join(feature_columns)} FROM {ATHENA_SCHEMA_NAME}.{glue_table_name} LIMIT 12;"
# #         )

# #         try:
# #             # Debugging: Log SQL queries
# #             print("[DEBUG] Entity & Target Query:", entity_target_query)
# #             print("[DEBUG] Features Query:", features_query)

# #             # Execute Athena queries
# #             entity_target_data = self.execute_athena_query(entity_target_query)
# #             features_data = self.execute_athena_query(features_query)

# #             # Combine results into a response object
# #             response_data = {
# #                 "entity_target_data": entity_target_data,
# #                 "features_data": features_data,
# #             }

# #             print("[DEBUG] Data successfully fetched for automation.")
# #             return Response(
# #                 {
# #                     "message": "Data fetched successfully.",
# #                     "data": response_data,
# #                 },
# #                 status=status.HTTP_200_OK,
# #             )
# #         except Exception as e:
# #             # Handle any errors during data retrieval or processing
# #             print(f"[ERROR] Failed to fetch data for automation: {str(e)}")
# #             return Response({"error": f"Failed to fetch data for automation: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# #     def execute_athena_query(self, query):
# #         """
# #         Execute a SQL query using Athena and return the results.

# #         Args:
# #             query (str): The SQL query to execute.

# #         Returns:
# #             List[Dict[str, Any]]: The query results as a list of dictionaries.
# #         """
# #         print("[DEBUG] Executing Athena query:", query)

# #         if not AWS_ATHENA_S3_STAGING_DIR:
# #             raise ValueError("AWS_ATHENA_S3_STAGING_DIR is not set.")

# #         # Create Athena connection string
# #         connection_string = (
# #             f"awsathena+rest://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}"
# #             f"@athena.{AWS_REGION_NAME}.amazonaws.com:443/{ATHENA_SCHEMA_NAME}?"
# #             f"s3_staging_dir={AWS_ATHENA_S3_STAGING_DIR}&catalog_name=AwsDataCatalog"
# #         )

# #         print("[DEBUG] Athena connection string created.")

# #         # Execute the query
# #         engine = create_engine(connection_string)
# #         with engine.connect() as connection:
# #             try:
# #                 # Use the `text` object for raw SQL queries
# #                 result = connection.execute(text(query))
# #                 data = []

# #                 # Map each row to its corresponding column name
# #                 columns = result.keys()  # Retrieve column names
# #                 for row in result:
# #                     data.append(dict(zip(columns, row)))

# #                 print("[DEBUG] Query executed successfully. Rows returned:", len(data))
# #                 return data

# #             except Exception as e:
# #                 print(f"[ERROR] Failed to execute query: {query}. Error: {str(e)}")
# #                 raise


# #     def verify_glue_table(self, database_name, table_name):
# #         """
# #         Ensure the Glue table exists before querying Athena.

# #         Args:
# #             database_name (str): The Athena database name (e.g., defined in settings).
# #             table_name (str): The Glue table name to check for existence.

# #         Raises:
# #             ValueError: If the Glue table is not found.
# #         """
# #         glue_client = boto3.client("glue", region_name=AWS_REGION_NAME)
# #         try:
# #             glue_client.get_table(DatabaseName=database_name, Name=table_name)
# #             print(f"[DEBUG] Glue table '{table_name}' verified.")
# #         except glue_client.exceptions.EntityNotFoundException:
# #             raise ValueError(f"Glue table '{table_name}' not found in database '{database_name}'.")

# #     def pass_to_automation(self, user_id, data):
# #         """
# #         Placeholder for integration with the automation pipeline.

# #         Args:
# #             user_id (str): The user identifier (default_user in this case).
# #             data (Dict[str, Any]): The raw data fetched from Athena queries.
# #         """
# #         print(f"[DEBUG] Passing data for user '{user_id}' to automation pipeline...")
# #         # Example: requests.post(AUTOMATION_PIPELINE_URL, json=data)
# #         pass




# import os
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status

# class DataForAutomationAPI(APIView):
#     """
#     Endpoint: /api/data-for-automation/
#     Modified to return the S3 file URL and confirmed schema columns without querying Athena.
#     """

#     def post(self, request):
#         """
#         Handles POST requests to return the S3 file URL and confirmed schema details.
#         This version does not perform any SQL queries against Athena.
#         """
#         print("[DEBUG] Received POST request at DataForAutomationAPI.")
#         print("[DEBUG] Request data:", request.data)

#         # Extract data from the request
#         # Expecting the client to provide these details now that we are not querying Athena
#         entity_column = request.data.get("entity_column")
#         target_column = request.data.get("target_column")
#         feature_columns = request.data.get("features", [])
#         file_url = request.data.get("file_url")

#         # Validate request data
#         # We must have file_url, entity_column, target_column, and features
#         if not (file_url and entity_column and target_column and feature_columns):
#             print("[ERROR] Missing required data:", {
#                 "file_url": file_url,
#                 "entity_column": entity_column,
#                 "target_column": target_column,
#                 "features": feature_columns,
#             })
#             return Response(
#                 {"error": "Missing required data: file_url, entity_column, target_column, features"},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )

#         # At this point, we've got everything we need. Just return them.
#         print("[DEBUG] All required data received. Preparing response.")
#         print(f"[DEBUG] File URL: {file_url}")
#         print(f"[DEBUG] Entity Column: {entity_column}")
#         print(f"[DEBUG] Target Column: {target_column}")
#         print(f"[DEBUG] Features: {feature_columns}")

#         response_data = {
#             "file_url": file_url,
#             "entity_column": entity_column,
#             "target_column": target_column,
#             "features": feature_columns
#         }

#         print("[DEBUG] Response data prepared successfully. Returning response.")
#         return Response(
#             {
#                 "message": "Data prepared successfully.",
#                 "data": response_data,
#             },
#             status=status.HTTP_200_OK,
#         )



import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

class DataForAutomationAPI(APIView):
    """
    Endpoint: /api/data-for-automation/
    Modified to return the S3 file URL and confirmed schema columns without querying Athena.
    """

    def post(self, request):
        print("[DEBUG] DataForAutomationAPI: Received POST request.")
        print("[DEBUG] Request data:", request.data)

        # Extract data from the request
        # Expecting the client to provide these details now that we are not querying Athena
        entity_column = request.data.get("entity_column")
        target_column = request.data.get("target_column")
        feature_columns = request.data.get("features", [])
        file_url = request.data.get("file_url")

        # Validate request data
        # We must have file_url, entity_column, target_column, and features
        if not (file_url and entity_column and target_column and feature_columns):
            print("[ERROR] Missing required data:", {
                "file_url": file_url,
                "entity_column": entity_column,
                "target_column": target_column,
                "features": feature_columns,
            })
            return Response(
                {"error": "Missing required data: file_url, entity_column, target_column, features"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        # At this point, we've got everything we need. Just return them.
        print("[DEBUG] All required data received. Preparing response.")
        print(f"[DEBUG] File URL: {file_url}")
        print(f"[DEBUG] Entity Column: {entity_column}")
        print(f"[DEBUG] Target Column: {target_column}")
        print(f"[DEBUG] Features: {feature_columns}")

        response_data = {
            "file_url": file_url,
            "entity_column": entity_column,
            "target_column": target_column,
            "features": feature_columns
        }

        print("[DEBUG] Response data prepared successfully. Returning response.")
        return Response(
            {
                "message": "Data prepared successfully.",
                "data": response_data,
            },
            status=status.HTTP_200_OK,
        )
