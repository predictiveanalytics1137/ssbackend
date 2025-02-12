import os
from celery import Celery

# Set the default Django settings module
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "chat_backend.settings")

celery_app = Celery("chat_backend")

# Load task modules from Django settings
celery_app.config_from_object("django.conf:settings", namespace="CELERY")

# Auto-discover Celery tasks in Django apps
celery_app.autodiscover_tasks()
