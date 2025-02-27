import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from celery import shared_task
from automation.src.logging_config import get_logger
from automation.scripts.train_pipeline import train_pipeline_api, train_pipeline_timeseries_api
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