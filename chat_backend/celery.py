import os
from celery import Celery

# # Set the default Django settings module
# os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chat_backend.settings")

# celery_app = Celery("chat_backend")

# # Load task modules from Django settings
# celery_app.config_from_object("django.conf:settings", namespace="CELERY")

# # Auto-discover Celery tasks in Django apps
# celery_app.autodiscover_tasks()



import os
from celery import Celery
from automation.src.logging_config import get_logger
import logging

# Set the default Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chat_backend.settings")

# Initialize Celery app
celery_app = Celery("chat_backend")

# Load task modules from Django settings
celery_app.config_from_object("django.conf:settings", namespace="CELERY")

# Prevent Celery from overriding the root logger
celery_app.conf.worker_hijack_root_logger = False

# Customize log format (optional)
celery_app.conf.worker_log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Sync Celery logging with your app's logging
@celery_app.on_after_configure.connect
def setup_logging(**kwargs):
    # Get your app's logger
    logger = get_logger('celery')
    root_logger = logging.getLogger('')
    # Clear Celery's default handlers
    celery_logger = logging.getLogger('celery')
    celery_logger.handlers = []
    # Add the root logger's handlers (logs/app.log)
    for handler in root_logger.handlers:
        celery_logger.addHandler(handler)
    celery_logger.setLevel(logging.INFO)
    celery_logger.propagate = False  # Prevent duplicate logging

# Auto-discover tasks in Django apps
celery_app.autodiscover_tasks()