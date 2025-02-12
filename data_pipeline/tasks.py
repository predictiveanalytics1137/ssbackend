from celery import shared_task
import subprocess
import logging

from automation.scripts.train_pipeline import train_pipeline_api


logger = logging.getLogger(__name__)

# @shared_task
# def train_model_task(file_url, target_column, user_id, chat_id, column_id):
#     try:
#         # script_path = r"C:\sandy2\ssbackend\automation\scripts\train_pipeline.py"
#         script_path = r"C:\Users\sande\Desktop\chatwork\ssbackend\automation\scripts\train_pipeline.py"
#         command = [
#             "python",
#             script_path,
#             "--file_url", file_url,
#             "--target_column", target_column,
#             "--user_id", str(user_id),
#             "--chat_id", str(chat_id),
#             "--column_id", column_id
#         ]

#         logger.info(f"Executing command: {' '.join(command)}")
        
#         # Run the training script as a subprocess
#         process = subprocess.run(
#             command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
#         )

#         if process.returncode != 0:
#             logger.error(f"Training failed: {process.stderr}")
#             return {"status": "failed", "error": process.stderr}

#         logger.info(f"Training succeeded: {process.stdout}")
#         return {"status": "success", "output": process.stdout}

#     except Exception as e:
#         logger.error(f"Exception during training: {str(e)}")
#         return {"status": "failed", "error": str(e)}


from celery import shared_task


logger = logging.getLogger(__name__)


@shared_task(bind=True)
def train_model_task(self, file_url, target_column, user_id, chat_id, column_id):
    """
    Celery task to run model training asynchronously.
    """
    try:
        logger.info(f"Starting Celery task for user_id={user_id}, chat_id={chat_id}")
        print(f"Starting Celery task for user_id={user_id}, chat_id={chat_id}")
        
        # Call train_pipeline_api directly
        result = train_pipeline_api(file_url, target_column, user_id, chat_id, column_id)

        logger.info(f"Training completed. Result: {result}")
        return result

    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        return {"status": "failed", "error": str(e)}
