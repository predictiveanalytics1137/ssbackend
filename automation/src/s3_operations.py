
import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from django.conf import settings  # Import Django settings

# def get_s3_client():
#     """
#     Creates an S3 client using AWS credentials from Django settings.
#     """
#     return boto3.client(
#         's3',
#         aws_access_key_id= os.getenv('AWS_ACCESS_KEY_ID'),
#         aws_secret_access_key= os.getenv('AWS_SECRET_ACCESS_KEY'),
#         region_name=os.getenv('AWS_S3_REGION_NAME'),
#     )



# def upload_to_s3(file_obj, bucket_name, s3_key):
#     """
#     Uploads a file object to S3.

#     Parameters:
#     - file_obj: A file-like object (e.g., BytesIO)
#     - bucket_name: Name of the S3 bucket
#     - s3_key: Key for the object in S3
#     """
#     s3 = get_s3_client()  # Ensure credentials are passed from settings
#     try:
#         s3.upload_fileobj(file_obj, bucket_name, s3_key)
#         print(f"Uploaded to s3://{bucket_name}/{s3_key}")
#     except Exception as e:
#         print(f"Error uploading to S3: {e}")



# def download_from_s3(bucket_name, s3_key, download_path):
#     """
#     Downloads a file from S3.
#     """
#     s3 = get_s3_client()  # Get the S3 client with credentials
#     try:
#         print(bucket_name,s3_key,download_path)
#         s3.download_file(bucket_name, s3_key, download_path)
#         print(f"Downloaded s3://{bucket_name}/{s3_key} to {download_path}")
#     except NoCredentialsError:
#         print("AWS credentials not available.")
#     except PartialCredentialsError:
#         print("Incomplete AWS credentials configuration.")
#     except Exception as e:
#         print(f"Error downloading file from S3: {e}")



# v2
import os
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError
from src.logging_config import get_logger

logger = get_logger(__name__)

def get_s3_client():
    """
    Creates an S3 client using credentials from environment variables.
    """
    try:
        s3 = boto3.client(
            's3',
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
            region_name=os.getenv('AWS_S3_REGION_NAME'),
        )
        return s3
    except Exception as e:
        logger.error(f"Error creating S3 client: {e}")
        raise

def upload_to_s3(file_obj, bucket_name, s3_key):
    """
    Uploads a file-like object to S3.
    """
    s3 = get_s3_client()
    try:
        s3.upload_fileobj(file_obj, bucket_name, s3_key)
        logger.info(f"Uploaded to s3://{bucket_name}/{s3_key}")
    except (NoCredentialsError, PartialCredentialsError):
        logger.error("AWS credentials not available or incomplete.")
        raise
    except Exception as e:
        logger.error(f"Error uploading to S3: {e}")
        raise

def download_from_s3(bucket_name, s3_key, download_path):
    """
    Downloads a file from S3.
    """
    s3 = get_s3_client()
    try:
        s3.download_file(bucket_name, s3_key, download_path)
        logger.info(f"Downloaded s3://{bucket_name}/{s3_key} to {download_path}")
    except NoCredentialsError:
        logger.error("AWS credentials not available.")
        raise
    except PartialCredentialsError:
        logger.error("Incomplete AWS credentials configuration.")
        raise
    except Exception as e:
        logger.error(f"Error downloading file from S3: {e}")
        raise

# =============================================================================
# def load_from_s3(bucket_name, s3_key):
#     """
#     Loads an object from S3 into a BytesIO buffer.
#     """
#     s3 = get_s3_client()
#     try:
#         buffer = bytearray()
#         with open('C:/sandy', 'wb') as f:
#         #with open('/tmp/temp_s3_download', 'wb') as f:
#             s3.download_fileobj(bucket_name, s3_key, f)
#         with open('C:/sandy', 'rb') as f:
#             buffer = f.read()
#         return bytearray(buffer)
#     except Exception as e:
#         logger.error(f"Error loading from S3: {e}")
#         raise
# =============================================================================


import io

def load_from_s3(bucket_name, s3_key):
    """
    Loads an object from S3 into a BytesIO buffer.
    """
    s3 = get_s3_client()
    try:
        buffer = io.BytesIO()
        s3.download_fileobj(bucket_name, s3_key, buffer)
        buffer.seek(0)  # Reset buffer to the beginning after download
        return buffer
    except Exception as e:
        logger.error(f"Error loading from S3: {e}")
        raise

