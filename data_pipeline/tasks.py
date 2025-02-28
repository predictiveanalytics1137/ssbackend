import os
import sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from automation.src.pipeline import predict_new_data
from automation.src.time_series_pipeline import predict_future_timeseries
from celery import shared_task
from automation.src.logging_config import get_logger
from automation.scripts.train_pipeline import fetch_csv_from_s3, train_pipeline_api, train_pipeline_timeseries_api
from celery import shared_task
logger = get_logger(__name__)

logger.info(f"we are in tasks.py")
@shared_task(bind=True)
def train_model_task(self, file_url, target_column, user_id, chat_id, column_id, ml_type):
    try:
        logger.info(f"Starting Celery task for user_id={user_id}, chat_id={chat_id}, ml_type={ml_type}")
        print(f"Starting Celery task for user_id={user_id}, chat_id={chat_id}, ml_type={ml_type}")
        
        if ml_type:
            result = train_pipeline_timeseries_api(file_url, target_column, user_id, chat_id, column_id)
            logger.info(f"Timeseries API result: {result}")
            print(f"Timeseries API result: {result}")
            model, params = result
        else:
            result = train_pipeline_api(file_url, target_column, user_id, chat_id, column_id)
            logger.info(f"Regular API result: {result}")
            print(f"Regular API result: {result}")
            model, params = result

        if model is None:
            return {"status": "failed", "error": params.get("error", "Unknown error")}

        logger.info(f"Training completed. Result: {model}")
        return {"status": "success", "model": str(model), "params": params}
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {"status": "failed", "error": str(e)}
    



from celery import shared_task
from src.logging_config import get_logger
from src.s3_operations import download_from_s3
import joblib
import pandas as pd

logger = get_logger(__name__)

@shared_task(bind=True)
def predict_model_task(self, file_url, column_id, user_id, chat_id, ml_type):
    """
    Celery task to trigger prediction based on ml_type.
    Triggers predict_new_data if ml_type is False, predict_future_timeseries if ml_type is True.

    Parameters:
    - file_url (str): URL or path to the prediction dataset.
    - column_id (str): Column name for the entity ID.
    - user_id (str): User identifier for logging.
    - chat_id (str): Chat identifier to load corresponding artifacts from S3.
    - ml_type (bool): True for time-series prediction, False for regular prediction.

    Returns:
    - dict: Result containing status and prediction output.
    """
    try:
        logger.info(f"Starting prediction task for user_id={user_id}, chat_id={chat_id}, ml_type={ml_type}")

        # Load the prediction dataset
        logger.info(f"Loading prediction dataset from {file_url}")
        
        # if file_url.startswith("s3://"):
        #     df = pd.read_csv(fetch_csv_from_s3(file_url.replace("s3://", "").split("/")[0], "/".join(file_url.replace("s3://", "").split("/")[1:])))
        # else:
        #     df = pd.read_csv(file_url)
        if file_url.startswith("s3://"):
            df = fetch_csv_from_s3(file_url)
        else:
            df = pd.read_csv(file_url)
            # df = fetch_csv_from_s3(file_url)

        # Drop unwanted columns
        df = df.drop(columns=["entity_id", "date","target_within_30_days_after"], errors="ignore")

        # Determine prediction function based on ml_type
        if ml_type:
            logger.info("Performing time-series prediction...")
            result_df = predict_future_timeseries(df, chat_id, time_column="analysis_time", column_id=column_id, target_column="daily_demand")
        else:
            logger.info("Performing regular prediction...")
            result_df = predict_new_data(df, chat_id, column_id=column_id)  # Assuming predict_new_data exists

        # Prepare result
        result = {
            "status": "success",
            "predictions": result_df.to_dict(orient='records') if not result_df.empty else [],
            "chat_id": chat_id
        }
        logger.info(f"Prediction completed successfully. Result shape: {result_df.shape if not result_df.empty else 'Empty'}")
        return result

    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}", exc_info=True)
        return {"status": "failed", "error": str(e)}