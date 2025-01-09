
import argparse
import sys
import os
import boto3
import pandas as pd
from io import StringIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import train_pipeline


# def fetch_csv_from_s3(s3_path):
#     print("[DEBUG] fetch_csv_from_s3: Starting to fetch data from S3...")
#     print(f"[DEBUG] Received S3 path: {s3_path}")

#     # Parse S3 path
#     if not s3_path.startswith("s3://"):
#         print("[ERROR] Invalid S3 path. Ensure it starts with 's3://'.")
#         raise ValueError("Invalid S3 path. Make sure it starts with 's3://'.")

#     s3_parts = s3_path.replace("s3://", "").split("/")
#     bucket_name = s3_parts[0]
#     object_key = "/".join(s3_parts[1:])
#     print(f"[DEBUG] Parsed bucket_name: {bucket_name}, object_key: {object_key}")

#     # Initialize boto3 S3 client with hardcoded credentials
#     print("[DEBUG] Initializing S3 client...")
#     s3 = boto3.client(
#         's3',
#         aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#         aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#         region_name=os.getenv('AWS_S3_REGION_NAME'),
#     )
#     print("[DEBUG] S3 client initialized successfully.")

#     # Fetch the object
#     try:
#         print(f"[DEBUG] Fetching object from S3 bucket '{bucket_name}' with key '{object_key}'...")
#         obj = s3.get_object(Bucket=bucket_name, Key=object_key)
#         print("[DEBUG] S3 object fetched successfully.")
#     except Exception as e:
#         print(f"[ERROR] Failed to fetch object from S3: {str(e)}")
#         raise

#     # Read the CSV file content
#     try:
#         print("[DEBUG] Reading CSV content from S3 object...")
#         csv_content = obj['Body'].read().decode('utf-8')
#         df = pd.read_csv(StringIO(csv_content))
#         print(f"[DEBUG] CSV content successfully loaded. DataFrame shape: {df.shape}")
#         return df
#     except Exception as e:
#         print(f"[ERROR] Failed to load CSV content: {str(e)}")
#         raise




# if __name__ == "__main__":
#     print("[DEBUG] Starting script execution...")

#     # Parse arguments
#     parser = argparse.ArgumentParser(description="Run the train pipeline.")
#     parser.add_argument("--file_url", required=True, help="S3 file URL for the CSV data.")
#     parser.add_argument("--target_column", required=True, help="Target column for training.")
#     parser.add_argument("--user_id", required=True, help="User ID for this training session.")
#     parser.add_argument("--chat_id", required=True, help="Chat ID for this training session.")
#     parser.add_argument("--column_id", required=True, help="Column ID for this training session.")
#     args = parser.parse_args()
#     print(f"[DEBUG] Arguments received - file_url: {args.file_url}, target_column: {args.target_column}, user_id: {args.user_id}, chat_id: {args.chat_id}, column_id: {args.column_id}")

#     try:
#         # Fetch data from S3
#         print("[DEBUG] Fetching data from S3...")
#         data = fetch_csv_from_s3(args.file_url)
#         print("[DEBUG] Data fetched successfully from S3. Here's a preview:")
#         # print(data.head())

#         # Train the pipeline with user_id and chat_id
#         print("[DEBUG] Starting training pipeline...")
#         best_model, best_params = train_pipeline(data, args.target_column, args.user_id, args.chat_id, args.column_id)
        
#         print("------------------------------------------")
#         print(args.column_id)
#         print("------------------------------------------")
#         print("[DEBUG] Training complete.")
#         print(f"[DEBUG] Best Model: {best_model}")
#         print(f"[DEBUG] Best Parameters: {best_params}")
#     except Exception as e:
#         print(f"[ERROR] An error occurred during execution: {str(e)}")
#         sys.exit(1)

#     print("[DEBUG] Script execution completed successfully.")


import argparse
import os
import sys
import pandas as pd
import boto3
from io import StringIO
from src.pipeline import train_pipeline
from src.logging_config import get_logger

logger = get_logger(__name__)

def fetch_csv_from_s3(s3_path):
    """
    Fetches a CSV file from S3 with error handling for empty data.
    """
    try:
        if not s3_path.startswith("s3://"):
            raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'.")
        
        s3_parts = s3_path.replace("s3://", "").split("/")
        bucket_name = s3_parts[0]
        object_key = "/".join(s3_parts[1:])
        
        logger.info(f"Fetching CSV from S3 bucket: {bucket_name}, key: {object_key}")
        
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_S3_REGION_NAME'),
        )

        obj = s3.get_object(Bucket=bucket_name, Key=object_key)
        csv_content = obj['Body'].read().decode('utf-8')
        df = pd.read_csv(StringIO(csv_content))

        if df.shape[0] == 0:
            raise ValueError("Fetched CSV is empty. Cannot proceed with training.")

        logger.info(f"CSV fetched successfully. DataFrame shape: {df.shape}")
        return df

    except Exception as e:
        logger.error(f"Error fetching CSV from S3: {e}")
        raise

if __name__ == "__main__":
    
    data = "s3://pa-documents-storage-bucket/uploads/4a87aeba/Updated_Dataset_with_Entity_Column.csv"
    data = fetch_csv_from_s3(data)

            #best_model, best_params = train_pipeline(data, args.target_column, args.user_id, args.chat_id, args.column_id)
    best_model, best_params = train_pipeline(
        data,
        "math_score",       # target_column
        "9938938HHDU",      # user_id
        "IDSH938749",       # chat_id
        "entity"            # column_id
    )
    
    
    
    
    
    
    
    '''
    parser = argparse.ArgumentParser(description="Run the train pipeline.")
    parser.add_argument("--file_url", required=True, help="S3 file URL for the CSV data.")
    parser.add_argument("--target_column", required=True, help="Target column for training.")
    parser.add_argument("--user_id", required=True, help="User ID for this training session.")
    parser.add_argument("--chat_id", required=True, help="Chat ID for this training session.")
    parser.add_argument("--column_id", required=True, help="Column ID (unique entity) for this session.")
    args = parser.parse_args()

    logger.info(f"Training pipeline starting with args: {args}")

    try:
        data = fetch_csv_from_s3(args.file_url)
        logger.info("Data fetched from S3. Proceeding with pipeline...")
        best_model, best_params = train_pipeline(
            df=data,
            target_column=args.target_column,
            user_id=args.user_id,
            chat_id=args.chat_id,
            column_id=args.column_id
        )
        logger.info(f"Training complete. Best model: {best_model}, Params: {best_params}")

    except Exception as e:
        logger.error(f"Error during train pipeline execution: {e}")
        sys.exit(1)
        '''
        
        