
import argparse
import sys
import os
import boto3
import pandas as pd
from io import StringIO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pipeline import train_pipeline

def fetch_csv_from_s3(s3_path):
    # Parse S3 path
    if not s3_path.startswith("s3://"):
        raise ValueError("Invalid S3 path. Make sure it starts with 's3://'.")
    
    s3_parts = s3_path.replace("s3://", "").split("/")
    bucket_name = s3_parts[0]
    object_key = "/".join(s3_parts[1:])
    
    # Initialize boto3 S3 client with hardcoded credentials
    s3 = boto3.client(
        's3',
        aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID'),
        aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY'),
        region_name=os.getenv('AWS_S3_REGION_NAME'),
    )

    
    obj = s3.get_object(Bucket=bucket_name, Key=object_key)
    
    # Read the CSV file content
    csv_content = obj['Body'].read().decode('utf-8')
    df = pd.read_csv(StringIO(csv_content))
    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the train pipeline.")
    parser.add_argument("--file_url", required=True, help="S3 file URL for the CSV data.")
    parser.add_argument("--target_column", required=True, help="Target column for training.")
    args = parser.parse_args()

    # Fetch data from S3
    data = fetch_csv_from_s3(args.file_url)
    print(data)

    # Train the pipeline
    best_model, best_params = train_pipeline(data, args.target_column)
    print("Training complete.")
