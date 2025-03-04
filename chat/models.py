

import uuid
from django.db import models
from django.core.exceptions import ValidationError
from django.conf import settings
import datetime


# Dynamic upload path function for organizing files in S3 by date
def upload_to(instance, filename):
    now = datetime.datetime.now()  # Generate the current timestamp
    return f'uploads/{now.strftime("%Y/%m/%d")}/{filename}'  # Organize uploads by year/month/day


class UploadedFile(models.Model):
    """
    Model to store metadata about uploaded files.
    """
    name = models.CharField(max_length=255)  # File name as uploaded by the user
    file = models.FileField(upload_to=upload_to)  # File field with dynamic upload path
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Timestamp when the file was uploaded
    file_url = models.TextField(null=True, blank=True)  # Full URL of the file in S3

    def save(self, *args, **kwargs):
        """
        Overridden save method to auto-generate the S3 file URL dynamically.
        """
        if self.file and not self.file_url:
            # Construct the file URL based on AWS S3 configuration
            self.file_url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{self.file.name}"
        super().save(*args, **kwargs)  # Save the instance

    def clean(self):
        """
        Validation logic to ensure the uploaded file meets requirements.
        """
        # Ensure the file is not empty
        if self.file and self.file.size == 0:
            raise ValidationError("Uploaded file is empty.")
        # Ensure the file has a supported extension (CSV in this case)
        if self.file and not self.file.name.endswith('.csv'):
            raise ValidationError("Only CSV files are allowed.")

    def __str__(self):
        """
        String representation of the model for easier identification.
        """
        return f"{self.name} - {self.file_url or 'No URL'}"


class FileSchema(models.Model):
    """
    Model to store schema metadata for uploaded files.
    """
    file = models.OneToOneField(UploadedFile, on_delete=models.CASCADE, related_name="schema")
    # Store the schema as JSON, with each entry containing column name and data type
    schema = models.JSONField()  # Requires Django 3.1+ for native JSONField support
    created_at = models.DateTimeField(auto_now_add=True)  # Timestamp when the schema was inferred

    def __str__(self):
        """
        String representation of the schema, showing file name and created date.
        """
        return f"Schema for {self.file.name} (Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')})"








from django.db import models
from django.contrib.auth.models import User
import uuid



# models.py
from django.contrib.auth.models import User
from django.db import models

class ChatBackup(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chats")  # Links to `auth_user.id`
    chat_id = models.CharField(max_length=255, unique=True)  # Unique ID for each chat
    title = models.CharField(max_length=255)
    messages = models.JSONField()  # Store chat messages as JSON
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.username} - {self.chat_id}"



# notebooks/models.py
from django.db import models

class Notebook(models.Model):
    user = models.ForeignKey('auth.User', on_delete=models.CASCADE)
    # chat can be a CharField or a ForeignKey; here it is a CharField.
    chat = models.CharField(max_length=255)
    entity_column = models.CharField(max_length=255)
    target_column = models.CharField(max_length=255)
    time_column = models.CharField(max_length=255, null=True, blank=True)
    time_frame = models.CharField(max_length=255, null=True, blank=True)
    time_frequency = models.CharField(max_length=255, null=True, blank=True)
    features = models.JSONField()
    file_url = models.URLField()
    notebook_json = models.JSONField()
    # New JSONField to store the mapping of each cell to its S3 URL
    cell_s3_links = models.JSONField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Notebook for User {self.user.id}, Chat {self.chat}"





# from django.db import models
# from django.contrib.auth.models import User

# class PredictiveSettings(models.Model):
#     """
#     Stores user-chosen predictive columns/settings for each chat in the database.
#     """
#     user = models.ForeignKey(User, on_delete=models.CASCADE)
#     chat_id = models.CharField(max_length=255)

#     # Possible fields the user might set
#     target_column = models.CharField(max_length=255, null=True, blank=True)
#     entity_column = models.CharField(max_length=255, null=True, blank=True)
#     time_frame = models.CharField(max_length=255, null=True, blank=True)
#     predictive_question = models.TextField(null=True, blank=True)

#     # You can add others as needed: time_column, time_frequency, etc.
#     time_column = models.CharField(max_length=255, null=True, blank=True)
#     time_frequency = models.CharField(max_length=255, null=True, blank=True)


#     machine_learning_type = models.CharField(max_length=50, null=True, blank=True)

#     updated_at = models.DateTimeField(auto_now=True)

#     def __str__(self):
#         return f"PredictiveSettings(user={self.user_id}, chat_id={self.chat_id})"


from django.db import models
from django.contrib.auth.models import User


class PredictiveSettings(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    chat_id = models.CharField(max_length=255)
    target_column = models.CharField(max_length=255, null=True, blank=True)
    entity_column = models.CharField(max_length=255, null=True, blank=True)
    time_frame = models.CharField(max_length=255, null=True, blank=True)
    predictive_question = models.TextField(null=True, blank=True)
    time_column = models.CharField(max_length=255, null=True, blank=True)
    time_frequency = models.CharField(max_length=255, null=True, blank=True)
    machine_learning_type = models.CharField(max_length=50, null=True, blank=True)
    features = models.JSONField(null=True, blank=True, default=list)  # NEW: Store feature list as JSON
    prediction_type = models.BooleanField(default=False)
    new_target_column = models.CharField(max_length=255, null=True, blank=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"PredictiveSettings(user={self.user_id}, chat_id={self.chat_id})"


# models.py
from django.db import models
from django.contrib.auth.models import User
from django.conf import settings

# (Existing models: UploadedFile, FileSchema, ChatBackup, Notebook, PredictiveSettings remain unchanged.)

class ChatFileInfo(models.Model):
    """
    Stores file metadata (schema, suggestions, etc.) for a specific chat.
    """
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="chat_file_infos")
    chat = models.ForeignKey('ChatBackup', on_delete=models.CASCADE, related_name="file_infos")
    file = models.OneToOneField(UploadedFile, on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    file_url = models.CharField(max_length=1024)
    schema = models.JSONField()  # List of dicts: [{ "column_name": ..., "data_type": ... }, ...]
    suggestions = models.JSONField(null=True, blank=True)  # e.g. target_column, entity_column, feature_columns.
    has_date_column = models.BooleanField(default=False)
    date_columns = models.JSONField(null=True, blank=True)
    glue_table_name = models.CharField(max_length=255, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"ChatFileInfo for Chat {self.chat.chat_id} - {self.name}"
