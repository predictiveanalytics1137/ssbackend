

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.pipeline import train_pipeline




# import argparse
# import os
# import sys
# import pandas as pd
# import boto3
# from io import StringIO
# from src.pipeline import train_pipeline
# from src.logging_config import get_logger

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# from src.pipeline import train_pipeline
# print(sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))))
# logger = get_logger(__name__)

# def fetch_csv_from_s3(s3_path):
#     """
#     Fetches a CSV file from S3 with error handling for empty data.
#     """
#     try:
#         if not s3_path.startswith("s3://"):
#             raise ValueError(f"Invalid S3 path: {s3_path}. Must start with 's3://'.")
        
#         s3_parts = s3_path.replace("s3://", "").split("/")
#         bucket_name = s3_parts[0]
#         object_key = "/".join(s3_parts[1:])
        
#         logger.info(f"Fetching CSV from S3 bucket: {bucket_name}, key: {object_key}")
        
#         s3 = boto3.client(
#             's3',
#             aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
#             aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
#             region_name=os.getenv('AWS_S3_REGION_NAME'),
#         )

#         obj = s3.get_object(Bucket=bucket_name, Key=object_key)
#         csv_content = obj['Body'].read().decode('utf-8')
#         df = pd.read_csv(StringIO(csv_content))

#         if df.shape[0] == 0:
#             raise ValueError("Fetched CSV is empty. Cannot proceed with training.")

#         logger.info(f"CSV fetched successfully. DataFrame shape: {df.shape}")
#         return df

#     except Exception as e:
#         logger.error(f"Error fetching CSV from S3: {e}")
#         raise

# if __name__ == "__main__":
    

#     # non time based
#     data = "s3://pa-documents-storage-bucket/uploads/4a87aeba/Updated_Dataset_with_Entity_Columnn.csv"
#     data = fetch_csv_from_s3(data)

#     # Example usage:
#     # best_model, best_params = train_pipeline(
#     #     df=data,
#     #     target_column="next_month_revenue",
#     #     user_id="9938938HHDU",
#     #     chat_id="IDSH938749",
#     #     column_id="dealer_id",  # This is your 'multivariate entity' or grouping column
#     #     time_column="sampled_date",  # <--- NEW: specify the time column here
#     #     freq="ME",                # <--- NEW: resample frequency (daily, weekly, etc.)
#     #     forecast_horizon=1,      # <--- how many steps ahead you want to forecast
#     #     use_time_series=True     # <--- tells the pipeline to do time-series preprocessing
#     # )


#     # Example usage for non-time-series dataset:
#     # best_model, best_params = train_pipeline(
#     #     df=data,
#     #     target_column="math_score",
#     #     user_id="9938938HHDU",
#     #     chat_id="IDSH938749",
#     #     column_id="entity",  # This is your 'multivariate entity' or grouping column
#     #     time_column=None,        # Remove time column for non-time-series
#     #     freq=None,               # No need for frequency in non-time-series
#     #     forecast_horizon=0,      # No forecasting needed for regular ML tasks
#     #     use_time_series=False    # Explicitly set to False
#     # )


    
    
    
    

#     parser = argparse.ArgumentParser(description="Run the train pipeline.")
#     parser.add_argument("--file_url", required=True, help="S3 file URL for the CSV data.")
#     parser.add_argument("--target_column", required=True, help="Target column for training.")
#     parser.add_argument("--user_id", required=True, help="User ID for this training session.")
#     parser.add_argument("--chat_id", required=True, help="Chat ID for this training session.")
#     parser.add_argument("--column_id", required=True, help="Column ID (unique entity) for this session.")
#     args = parser.parse_args()

#     logger.info(f"Training pipeline starting with args: {args}")

#     try:
#         data = fetch_csv_from_s3(args.file_url)
#         logger.info("Data fetched from S3. Proceeding with pipeline...")
#         best_model, best_params = train_pipeline(
#             df=data,
#             target_column=args.target_column,
#             user_id=args.user_id,
#             chat_id=args.chat_id,
#             column_id=args.column_id
#         )
#         logger.info(f"Training complete. Best model: {best_model}, Params: {best_params}")

#     except Exception as e:
#         logger.error(f"Error during train pipeline execution: {e}")
#         sys.exit(1)

        



import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import boto3
from io import StringIO
from src.pipeline import train_pipeline
from src.time_series_pipeline import train_pipeline_timeseries
from src.logging_config import get_logger

# Ensure proper path resolution
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

# ✅ Function to trigger pipeline from API (Callable by Celery)
def train_pipeline_api(file_url, target_column, user_id, chat_id, column_id):
    """
    Trigger the training pipeline asynchronously.
    """
    try:
        logger.info(f"Starting training for user_id={user_id}, chat_id={chat_id}")
        
        # Fetch data from S3
        df = fetch_csv_from_s3(file_url)
        
        # Run the training pipeline
        best_model, best_params = train_pipeline(
            df=df,
            target_column=target_column,
            user_id=user_id,
            chat_id=chat_id,
            column_id=column_id,
        )

        logger.info(f"Training completed successfully for user {user_id}.")
        print(f"Training completed successfully for user {user_id}.")
        print(f"Training completed successfully for user {chat_id}.")
        # return {"status": "success", "best_model": str(best_model), "best_params": best_params}
        return best_model, best_params

    except Exception as e:
        logger.error(f"Training failed: {e}")
        # return {"status": "failed", "error": str(e)}
        return None, {"status": "failed", "error": str(e)}



def train_pipeline_timeseries_api(file_url, target_column, user_id, chat_id, column_id, time_column="analysis_time", freq="weekly", forecast_horizon="30 days"):
    """
    Trigger the time-series training pipeline asynchronously.
    """
    try:
        logger.info(f"Starting time-series training for user_id={user_id}, chat_id={chat_id}")
        
        # Fetch data from S3
        df = fetch_csv_from_s3(file_url)
        
        # Run the time-series training pipeline
        best_model, best_params = train_pipeline_timeseries(
            df=df,
            target_column=target_column,
            user_id=user_id,
            chat_id=chat_id,
            column_id=column_id,
            time_column=time_column,
            freq=freq,
            forecast_horizon=forecast_horizon,
            use_time_series=True
        )

        logger.info(f"Time-series training completed successfully for user {user_id}.")
        
        return best_model, best_params

    except Exception as e:
        logger.error(f"Time-series training failed: {e}", exc_info=True)
        return None, {"status": "failed", "error": str(e)}
