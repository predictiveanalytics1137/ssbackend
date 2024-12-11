import argparse
import sys
import os
import boto3
import pandas as pd
from io import StringIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import predict_new_data


def fetch_csv_from_s3(s3_path):
    """
    Fetches a CSV file from an S3 bucket using boto3.
    """
    # Parse S3 path
    if not s3_path.startswith("s3://"):
        raise ValueError(f"Invalid S3 path: {s3_path}. Make sure it starts with 's3://'.")
    
    s3_parts = s3_path.replace("s3://", "").split("/")
    bucket_name = s3_parts[0]
    object_key = "/".join(s3_parts[1:])
    
    print(f"[INFO] S3 Bucket: {bucket_name}")
    print(f"[INFO] S3 Key: {object_key}")

    # Initialize boto3 S3 client with hardcoded credentials
    s3 = boto3.client(
        's3',
        aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_S3_REGION_NAME'),
    )
    
    # Fetch the object from S3
    try:
        obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    except Exception as e:
        raise ValueError(f"Failed to fetch the file from S3. Error: {e}")
    
    # Read the content of the file and load it into a DataFrame
    csv_content = obj['Body'].read().decode('utf-8')
    print("[INFO] File content fetched successfully from S3.")
    df = pd.read_csv(StringIO(csv_content))
    print("[INFO] DataFrame created successfully.")
    return df

if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Run the prediction pipeline.")
    parser.add_argument("--file_url", required=True, help="S3 file URL for the CSV data.")
    parser.add_argument("--bucket_name", default="artifacts1137", help="S3 bucket name where artifacts are stored.")
    args = parser.parse_args()

    # Debugging log for inputs
    print(f"[INFO] Received file_url: {args.file_url}")
    print(f"[INFO] Received bucket_name: {args.bucket_name}")

    try:
        # Fetch data from S3
        print("[INFO] Fetching data from S3...")
        data = fetch_csv_from_s3(args.file_url)
        print("[INFO] Data fetched successfully:")
        print(data.head())

        # Run predictions
        print("[INFO] Running predictions...")
        predictions = predict_new_data(data, args.bucket_name)
        print("[INFO] Predictions completed successfully:")
        print(predictions)

    except ValueError as ve:
        print(f"[ERROR] Value error: {ve}")
        sys.exit(1)
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred: {e}")
        sys.exit(1)
