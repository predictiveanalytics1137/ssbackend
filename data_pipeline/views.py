


# # # # import os
# # # # from rest_framework.views import APIView
# # # # from rest_framework.response import Response
# # # # from rest_framework import status
# # # # from sqlalchemy import create_engine, text  # Import text for raw SQL queries
# # # # import boto3

# # # # # AWS and Athena Configuration
# # # # AWS_ACCESS_KEY_ID = os.getenv('AWS_ACCESS_KEY_ID')
# # # # AWS_SECRET_ACCESS_KEY = os.getenv('AWS_SECRET_ACCESS_KEY')
# # # # AWS_S3_REGION_NAME = os.getenv('AWS_S3_REGION_NAME')
# # # # AWS_STORAGE_BUCKET_NAME = os.getenv('AWS_STORAGE_BUCKET_NAME')
# # # # AWS_ATHENA_S3_STAGING_DIR = os.getenv('AWS_ATHENA_S3_STAGING_DIR')
# # # # AWS_REGION_NAME = AWS_S3_REGION_NAME
# # # # ATHENA_SCHEMA_NAME = 'pa_user_datafiles_db'  # Athena database name

# # # # class DataForAutomationAPI(APIView):
# # # #     """
# # # #     Endpoint: /api/data-for-automation/
# # # #     Fetch data dynamically based on user-confirmed schema and pass it to the automation pipeline.
# # # #     """

# # # #     def post(self, request):
# # # #         """
# # # #         Handles POST requests to fetch data for automation based on user-confirmed schema.
# # # #         """
# # # #         print("[DEBUG] Received POST request at DataForAutomationAPI.")
# # # #         print("[DEBUG] Request data:", request.data)

# # # #         # Extract data from the request
# # # #         entity_column = request.data.get("entity_column")
# # # #         target_column = request.data.get("target_column")
# # # #         feature_columns = request.data.get("features", [])
# # # #         glue_table_name = request.data.get("glue_table_name")

# # # #         # Validate request data
# # # #         if not all([entity_column, target_column, feature_columns, glue_table_name]):
# # # #             print("[ERROR] Missing required data:", {
# # # #                 "entity_column": entity_column,
# # # #                 "target_column": target_column,
# # # #                 "features": feature_columns,
# # # #                 "glue_table_name": glue_table_name,
# # # #             })
# # # #             return Response(
# # # #                 {"error": "Missing required data: entity_column, target_column, features, glue_table_name"},
# # # #                 status=status.HTTP_400_BAD_REQUEST,
# # # #             )

# # # #         # SQL queries with LIMIT 10 for presentation purposes
# # # #         entity_target_query = (
# # # #             f"SELECT {entity_column}, {target_column} FROM {ATHENA_SCHEMA_NAME}.{glue_table_name} LIMIT 12;"
# # # #         )
# # # #         features_query = (
# # # #             f"SELECT {', '.join(feature_columns)} FROM {ATHENA_SCHEMA_NAME}.{glue_table_name} LIMIT 12;"
# # # #         )

# # # #         try:
# # # #             # Debugging: Log SQL queries
# # # #             print("[DEBUG] Entity & Target Query:", entity_target_query)
# # # #             print("[DEBUG] Features Query:", features_query)

# # # #             # Execute Athena queries
# # # #             entity_target_data = self.execute_athena_query(entity_target_query)
# # # #             features_data = self.execute_athena_query(features_query)

# # # #             # Combine results into a response object
# # # #             response_data = {
# # # #                 "entity_target_data": entity_target_data,
# # # #                 "features_data": features_data,
# # # #             }

# # # #             print("[DEBUG] Data successfully fetched for automation.")
# # # #             return Response(
# # # #                 {
# # # #                     "message": "Data fetched successfully.",
# # # #                     "data": response_data,
# # # #                 },
# # # #                 status=status.HTTP_200_OK,
# # # #             )
# # # #         except Exception as e:
# # # #             # Handle any errors during data retrieval or processing
# # # #             print(f"[ERROR] Failed to fetch data for automation: {str(e)}")
# # # #             return Response({"error": f"Failed to fetch data for automation: {str(e)}"}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# # # #     def execute_athena_query(self, query):
# # # #         """
# # # #         Execute a SQL query using Athena and return the results.

# # # #         Args:
# # # #             query (str): The SQL query to execute.

# # # #         Returns:
# # # #             List[Dict[str, Any]]: The query results as a list of dictionaries.
# # # #         """
# # # #         print("[DEBUG] Executing Athena query:", query)

# # # #         if not AWS_ATHENA_S3_STAGING_DIR:
# # # #             raise ValueError("AWS_ATHENA_S3_STAGING_DIR is not set.")

# # # #         # Create Athena connection string
# # # #         connection_string = (
# # # #             f"awsathena+rest://{AWS_ACCESS_KEY_ID}:{AWS_SECRET_ACCESS_KEY}"
# # # #             f"@athena.{AWS_REGION_NAME}.amazonaws.com:443/{ATHENA_SCHEMA_NAME}?"
# # # #             f"s3_staging_dir={AWS_ATHENA_S3_STAGING_DIR}&catalog_name=AwsDataCatalog"
# # # #         )

# # # #         print("[DEBUG] Athena connection string created.")

# # # #         # Execute the query
# # # #         engine = create_engine(connection_string)
# # # #         with engine.connect() as connection:
# # # #             try:
# # # #                 # Use the `text` object for raw SQL queries
# # # #                 result = connection.execute(text(query))
# # # #                 data = []

# # # #                 # Map each row to its corresponding column name
# # # #                 columns = result.keys()  # Retrieve column names
# # # #                 for row in result:
# # # #                     data.append(dict(zip(columns, row)))

# # # #                 print("[DEBUG] Query executed successfully. Rows returned:", len(data))
# # # #                 return data

# # # #             except Exception as e:
# # # #                 print(f"[ERROR] Failed to execute query: {query}. Error: {str(e)}")
# # # #                 raise


# # # #     def verify_glue_table(self, database_name, table_name):
# # # #         """
# # # #         Ensure the Glue table exists before querying Athena.

# # # #         Args:
# # # #             database_name (str): The Athena database name (e.g., defined in settings).
# # # #             table_name (str): The Glue table name to check for existence.

# # # #         Raises:
# # # #             ValueError: If the Glue table is not found.
# # # #         """
# # # #         glue_client = boto3.client("glue", region_name=AWS_REGION_NAME)
# # # #         try:
# # # #             glue_client.get_table(DatabaseName=database_name, Name=table_name)
# # # #             print(f"[DEBUG] Glue table '{table_name}' verified.")
# # # #         except glue_client.exceptions.EntityNotFoundException:
# # # #             raise ValueError(f"Glue table '{table_name}' not found in database '{database_name}'.")

# # # #     def pass_to_automation(self, user_id, data):
# # # #         """
# # # #         Placeholder for integration with the automation pipeline.

# # # #         Args:
# # # #             user_id (str): The user identifier (default_user in this case).
# # # #             data (Dict[str, Any]): The raw data fetched from Athena queries.
# # # #         """
# # # #         print(f"[DEBUG] Passing data for user '{user_id}' to automation pipeline...")
# # # #         # Example: requests.post(AUTOMATION_PIPELINE_URL, json=data)
# # # #         pass




# # # import os
# # # from rest_framework.views import APIView
# # # from rest_framework.response import Response
# # # from rest_framework import status

# # # class DataForAutomationAPI(APIView):
# # #     """
# # #     Endpoint: /api/data-for-automation/
# # #     Modified to return the S3 file URL and confirmed schema columns without querying Athena.
# # #     """

# # #     def post(self, request):
# # #         """
# # #         Handles POST requests to return the S3 file URL and confirmed schema details.
# # #         This version does not perform any SQL queries against Athena.
# # #         """
# # #         print("[DEBUG] Received POST request at DataForAutomationAPI.")
# # #         print("[DEBUG] Request data:", request.data)

# # #         # Extract data from the request
# # #         # Expecting the client to provide these details now that we are not querying Athena
# # #         entity_column = request.data.get("entity_column")
# # #         target_column = request.data.get("target_column")
# # #         feature_columns = request.data.get("features", [])
# # #         file_url = request.data.get("file_url")

# # #         # Validate request data
# # #         # We must have file_url, entity_column, target_column, and features
# # #         if not (file_url and entity_column and target_column and feature_columns):
# # #             print("[ERROR] Missing required data:", {
# # #                 "file_url": file_url,
# # #                 "entity_column": entity_column,
# # #                 "target_column": target_column,
# # #                 "features": feature_columns,
# # #             })
# # #             return Response(
# # #                 {"error": "Missing required data: file_url, entity_column, target_column, features"},
# # #                 status=status.HTTP_400_BAD_REQUEST,
# # #             )

# # #         # At this point, we've got everything we need. Just return them.
# # #         print("[DEBUG] All required data received. Preparing response.")
# # #         print(f"[DEBUG] File URL: {file_url}")
# # #         print(f"[DEBUG] Entity Column: {entity_column}")
# # #         print(f"[DEBUG] Target Column: {target_column}")
# # #         print(f"[DEBUG] Features: {feature_columns}")

# # #         response_data = {
# # #             "file_url": file_url,
# # #             "entity_column": entity_column,
# # #             "target_column": target_column,
# # #             "features": feature_columns
# # #         }

# # #         print("[DEBUG] Response data prepared successfully. Returning response.")
# # #         return Response(
# # #             {
# # #                 "message": "Data prepared successfully.",
# # #                 "data": response_data,
# # #             },
# # #             status=status.HTTP_200_OK,
# # #         )



# # import os
# # from rest_framework.views import APIView
# # from rest_framework.response import Response
# # from rest_framework import status

# # # class DataForAutomationAPI(APIView):
# # #     """
# # #     Endpoint: /api/data-for-automation/
# # #     Modified to return the S3 file URL and confirmed schema columns without querying Athena.
# # #     """

# # #     def post(self, request):
# # #         print("[DEBUG] DataForAutomationAPI: Received POST request.")
# # #         print("[DEBUG] Request data:", request.data)

# # #         # Extract data from the request
# # #         # Expecting the client to provide these details now that we are not querying Athena
# # #         entity_column = request.data.get("entity_column")
# # #         target_column = request.data.get("target_column")
# # #         feature_columns = request.data.get("features", [])
# # #         file_url = request.data.get("file_url")

# # #         # Validate request data
# # #         # We must have file_url, entity_column, target_column, and features
# # #         if not (file_url and entity_column and target_column and feature_columns):
# # #             print("[ERROR] Missing required data:", {
# # #                 "file_url": file_url,
# # #                 "entity_column": entity_column,
# # #                 "target_column": target_column,
# # #                 "features": feature_columns,
# # #             })
# # #             return Response(
# # #                 {"error": "Missing required data: file_url, entity_column, target_column, features"},
# # #                 status=status.HTTP_400_BAD_REQUEST,
# # #             )

# # #         # At this point, we've got everything we need. Just return them.
# # #         print("[DEBUG] All required data received. Preparing response.")
# # #         print(f"[DEBUG] File URL: {file_url}")
# # #         print(f"[DEBUG] Entity Column: {entity_column}")
# # #         print(f"[DEBUG] Target Column: {target_column}")
# # #         print(f"[DEBUG] Features: {feature_columns}")

# # #         response_data = {
# # #             "file_url": file_url,
# # #             "entity_column": entity_column,
# # #             "target_column": target_column,
# # #             "features": feature_columns
# # #         }

# # #         print("[DEBUG] Response data prepared successfully. Returning response.")
# # #         return Response(
# # #             {
# # #                 "message": "Data prepared successfully.",
# # #                 "data": response_data,
# # #             },
# # #             status=status.HTTP_200_OK,
# # #         )



# # import pandas as pd
# # import requests
# # from io import StringIO
# # from rest_framework.views import APIView
# # from rest_framework.response import Response
# # from rest_framework import status
# # # from src.pipeline import train_pipeline  # Assuming train_pipeline is properly defined in src.pipeline
# # from automation.scripts import train_pipeline

# # class DataForAutomationAPI(APIView):
# #     """
# #     Endpoint: /api/data-for-automation/
# #     Modified to start the train_pipeline with file_url and target_column.
# #     """

# #     def post(self, request):
# #         print("[DEBUG] DataForAutomationAPI: Received POST request.")
# #         print("[DEBUG] Request data:", request.data)

# #         # Extract data from the request
# #         file_url = request.data.get("file_url")
# #         target_column = request.data.get("target_column")

# #         # Validate request data
# #         if not file_url or not target_column:
# #             print("[ERROR] Missing required data:", {
# #                 "file_url": file_url,
# #                 "target_column": target_column,
# #             })
# #             return Response(
# #                 {"error": "Missing required data: file_url, target_column"},
# #                 status=status.HTTP_400_BAD_REQUEST,
# #             )

# #         try:
# #             # Fetch the CSV file from the file_url
# #             print("[DEBUG] Fetching CSV file from URL:", file_url)
# #             response = requests.get(file_url)
# #             response.raise_for_status()  # Raise an error for HTTP issues
            
# #             # Read the CSV content into a Pandas DataFrame
# #             csv_content = response.content.decode('utf-8')
# #             data = pd.read_csv(StringIO(csv_content))
# #             print("[DEBUG] CSV file loaded successfully. Shape:", data.shape)

# #             # Start the training pipeline
# #             print("[DEBUG] Starting train_pipeline...")
# #             best_model, best_params = train_pipeline(data, target_column)
# #             print("[DEBUG] train_pipeline completed successfully.")

# #             return Response(
# #                 {
# #                     "message": "Training completed successfully.",
# #                     "best_model": str(best_model),  # Serialize model details as needed
# #                     "best_params": best_params,
# #                 },
# #                 status=status.HTTP_200_OK,
# #             )

# #         except requests.exceptions.RequestException as e:
# #             print("[ERROR] Failed to fetch CSV file from URL:", str(e))
# #             return Response(
# #                 {"error": f"Failed to fetch CSV file from URL: {str(e)}"},
# #                 status=status.HTTP_400_BAD_REQUEST,
# #             )
# #         except Exception as e:
# #             print("[ERROR] Error in train_pipeline:", str(e))
# #             return Response(
# #                 {"error": f"Error in train_pipeline: {str(e)}"},
# #                 status=status.HTTP_500_INTERNAL_SERVER_ERROR,
# #             )



# from io import StringIO
# import subprocess
# import os
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# import re

# # from automation.scripts import predict

# class DataForAutomationAPI(APIView):
#     """
#     Endpoint: /api/data-for-automation/
#     Modified to start the train_pipeline.py script with file_url and target_column.
#     """

#     def post(self, request):
#         print("[DEBUG] DataForAutomationAPI: Received POST request.")
#         print("[DEBUG] Request data:", request.data)

#         # Extract data from the request
#         file_url = request.data.get("file_url")
#         target_column = request.data.get("target_column")

#         # Validate request data
#         if not file_url or not target_column:
#             print("[ERROR] Missing required data:", {
#                 "file_url": file_url,
#                 "target_column": target_column,
#             })
#             return Response(
#                 {"error": "Missing required data: file_url, target_column"},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )

#         try:
#             # Full path to the train_pipeline.py script
#             script_path = r"C:\Predictive Analysis\New Backend\ssbackend\automation\scripts\train_pipeline.py"

#             # Construct the command to run train_pipeline.py
#             command = [
#                 "python", 
#                 script_path, 
#                 "--file_url", file_url, 
#                 "--target_column", target_column
#             ]
            
#             print("[DEBUG] Running command:", " ".join(command))
            
#             # Run the command and capture the output
#             process = subprocess.run(
#                 command, 
#                 text=True, 
#                 stdout=subprocess.PIPE, 
#                 stderr=subprocess.PIPE
#             )

#             if process.returncode != 0:
#                 # Log and return error response if the script fails
#                 print("[ERROR] train_pipeline.py failed:", process.stderr)
#                 return Response(
#                     {
#                         "error": "Failed to execute train_pipeline.py",
#                         "details": process.stderr,
#                     },
#                     status=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 )

#             # Log and return success response
#             print("[DEBUG] train_pipeline.py output:", process.stdout)
#             return Response(
#                 {
#                     "message": "Training completed successfully.",
#                     "details": process.stdout,
#                 },
#                 status=status.HTTP_200_OK,
#             )

#         except Exception as e:
#             print("[ERROR] Exception occurred while running train_pipeline.py:", str(e))
#             return Response(
#                 {"error": f"An error occurred: {str(e)}"},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             )




# import pandas as pd
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# import subprocess
# import os
# import requests
# from io import BytesIO

# class PredictDataAPI(APIView):
#     """
#     Endpoint: /api/predict-data/
#     Modified to start the predict.py script with a file URL.
#     """

#     def post(self, request):
#         print("[DEBUG] PredictDataAPI: Received POST request.")
#         print("[DEBUG] Request data:", request.data)

#         # Extract data from the request
#         # file_url = request.data.get("file_url")
#         # import pdb, pdb.set_trace()
#         # import pdb; pdb.set_trace()
#         file_url = request.data.get("file_url")
#         bucket_name = request.data.get("bucket_name")
        

#         # Validate request data
#         if not file_url:
#             print("[ERROR] Missing file_url in request data.")
#             return Response(
#                 {"error": "Missing required data: file_url"},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )

#         try:
#             # Download the file from the file_url
#             print("[DEBUG] Downloading file from URL:", file_url)
#             # response = requests.get(file_url)
#             # response.raise_for_status()

#             # Save the CSV content to a temporary file
#             # temp_csv_path = "temp_predict_data.csv"
#             # with open(temp_csv_path, "wb") as f:
#             #     f.write(response.content)
#             # print("[DEBUG] File downloaded and saved to:", temp_csv_path)

#             # Full path to the predict.py script
#             script_path = r"C:\Predictive Analysis\New Backend\ssbackend\automation\scripts\predict.py"
            

#             # Construct the command to run predict.py
#             # command = [
#             #     "python",
#             #     script_path,
#             #     temp_csv_path,
#             #     bucket_name
#             # ]

#             command = [
#                 "python", 
#                 script_path, 
#                 "--file_url", file_url, 
#                 "--bucket_name", bucket_name
#             ]

#             print("[DEBUG] Running command:", " ".join(command))

#             # Run the command and capture the output
#             process = subprocess.run(
#                 command,
#                 text=True,
#                 stdout=subprocess.PIPE,
#                 stderr=subprocess.PIPE
#             )

#             # Remove the temporary file after use
#             # os.remove(temp_csv_path)

#             if process.returncode != 0:
#                 # Log and return error response if the script fails
#                 print("[ERROR] predict.py failed:", process.stderr)
#                 return Response(
#                     {
#                         "error": "Failed to execute predict.py",
#                         "details": process.stderr,
#                     },
#                     status=status.HTTP_500_INTERNAL_SERVER_ERROR,
#                 )

#             # Parse and return the predictions
#             print("[DEBUG] predict.py output:", process.stdout)
#             predictions = process.stdout.strip()
#             return Response(
#                 {
#                     "message": "Prediction completed successfully.",
#                     "predictions": predictions,
#                 },
#                 status=status.HTTP_200_OK,
#             )

#         except requests.exceptions.RequestException as e:
#             print("[ERROR] Failed to download file:", str(e))
#             return Response(
#                 {"error": f"Failed to download file: {str(e)}"},
#                 status=status.HTTP_400_BAD_REQUEST,
#             )
#         except Exception as e:
#             print("[ERROR] Exception occurred while running predict.py:", str(e))
#             return Response(
#                 {"error": f"An error occurred: {str(e)}"},
#                 status=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             )





from io import StringIO
import subprocess
import os
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import re


class DataForAutomationAPI(APIView):
    """
    Endpoint: /api/data-for-automation/
    Modified to start the train_pipeline.py script with file_url and target_column.
    """

    def post(self, request):
        print("[DEBUG] DataForAutomationAPI: Received POST request.")
        print("[DEBUG] Request data:", request.data)

        # Extract data from the request
        file_url = request.data.get("file_url")
        target_column = request.data.get("target_column")
        print(f"[DEBUG] Extracted file_url: {file_url}, target_column: {target_column}")

        # Validate request data
        if not file_url or not target_column:
            print("[ERROR] Missing required data:", {
                "file_url": file_url,
                "target_column": target_column,
            })
            return Response(
                {"error": "Missing required data: file_url, target_column"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Full path to the train_pipeline.py script
            script_path = r"C:\Predictive Analysis\New Backend\ssbackend\automation\scripts\train_pipeline.py"
            print(f"[DEBUG] Script path: {script_path}")

            # Construct the command to run train_pipeline.py
            command = [
                "python", 
                script_path, 
                "--file_url", file_url, 
                "--target_column", target_column
            ]
            print(f"[DEBUG] Constructed command: {' '.join(command)}")

            # Run the command and capture the output
            print("[DEBUG] Starting subprocess to run train_pipeline.py...")
            process = subprocess.run(
                command, 
                text=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.PIPE
            )
            print(f"[DEBUG] Subprocess finished with return code: {process.returncode}")

            if process.returncode != 0:
                print("[ERROR] train_pipeline.py failed:")
                print(process.stderr)
                return Response(
                    {
                        "error": "Failed to execute train_pipeline.py",
                        "details": process.stderr,
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            print("[DEBUG] train_pipeline.py succeeded. Output:")
            print(process.stdout)
            return Response(
                {
                    "message": "Training completed successfully.",
                    "details": process.stdout,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            print("[ERROR] Exception occurred while running train_pipeline.py:")
            print(str(e))
            return Response(
                {"error": f"An error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )


import pandas as pd
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import subprocess
import os
import requests
from io import BytesIO


class PredictDataAPI(APIView):
    """
    Endpoint: /api/predict-data/
    Modified to start the predict.py script with a file URL.
    """

    def post(self, request):
        print("[DEBUG] PredictDataAPI: Received POST request.")
        print("[DEBUG] Request data:", request.data)

        # Extract data from the request
        file_url = request.data.get("file_url")
        bucket_name = request.data.get("bucket_name")
        print(f"[DEBUG] Extracted file_url: {file_url}, bucket_name: {bucket_name}")

        # Validate request data
        if not file_url:
            print("[ERROR] Missing file_url in request data.")
            return Response(
                {"error": "Missing required data: file_url"},
                status=status.HTTP_400_BAD_REQUEST,
            )

        try:
            # Full path to the predict.py script
            script_path = r"C:\Predictive Analysis\New Backend\ssbackend\automation\scripts\predict.py"
            print(f"[DEBUG] Script path: {script_path}")

            # Construct the command to run predict.py
            command = [
                "python", 
                script_path, 
                "--file_url", file_url, 
                "--bucket_name", bucket_name
            ]
            print(f"[DEBUG] Constructed command: {' '.join(command)}")

            # Run the command and capture the output
            print("[DEBUG] Starting subprocess to run predict.py...")
            process = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print(f"[DEBUG] Subprocess finished with return code: {process.returncode}")

            if process.returncode != 0:
                print("[ERROR] predict.py failed:")
                print(process.stderr)
                return Response(
                    {
                        "error": "Failed to execute predict.py",
                        "details": process.stderr,
                    },
                    status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                )

            print("[DEBUG] predict.py succeeded. Output:")
            print(process.stdout)
            predictions = process.stdout.strip()
            return Response(
                {
                    "message": "Prediction completed successfully.",
                    "predictions": predictions,
                },
                status=status.HTTP_200_OK,
            )

        except requests.exceptions.RequestException as e:
            print("[ERROR] Failed to download file:")
            print(str(e))
            return Response(
                {"error": f"Failed to download file: {str(e)}"},
                status=status.HTTP_400_BAD_REQUEST,
            )
        except Exception as e:
            print("[ERROR] Exception occurred while running predict.py:")
            print(str(e))
            return Response(
                {"error": f"An error occurred: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
