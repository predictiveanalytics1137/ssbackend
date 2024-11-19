from django.db import models


class ChatHistory(models.Model):
    user_message = models.TextField()
    assistant_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User: {self.user_message[:30]} - Assistant: {self.assistant_response[:30]}"


# class UploadedFile(models.Model):
#     name = models.CharField(max_length=255)
#     file = models.FileField(upload_to='uploads/')
#     uploaded_at = models.DateTimeField(auto_now_add=True)
#     file_url = models.TextField(null=True, blank=True)  # Store the full S3 URL

#     def __str__(self):
#         return self.name


from django.db import models
from django.core.exceptions import ValidationError
from django.conf import settings

import datetime
from django.db import models
from django.core.exceptions import ValidationError
from django.conf import settings


# Dynamic upload path function
def upload_to(instance, filename):
    now = datetime.datetime.now()  # Generate the timestamp dynamically
    return f'uploads/{now.strftime("%Y/%m/%d")}/{filename}'


class UploadedFile(models.Model):
    name = models.CharField(max_length=255)  # File name
    file = models.FileField(upload_to=upload_to)  # File field with dynamic upload path
    uploaded_at = models.DateTimeField(auto_now_add=True)  # Auto-added timestamp
    file_url = models.TextField(null=True, blank=True)  # S3 file URL

    def save(self, *args, **kwargs):
        # Generate file URL based on S3 configuration
        if self.file and not self.file_url:
            self.file_url = f"https://{settings.AWS_STORAGE_BUCKET_NAME}.s3.{settings.AWS_S3_REGION_NAME}.amazonaws.com/{self.file.name}"
        super().save(*args, **kwargs)

    def clean(self):
        # Validate file content
        if self.file and self.file.size == 0:
            raise ValidationError("Uploaded file is empty.")
        if self.file and not self.file.name.endswith('.csv'):
            raise ValidationError("Only CSV files are allowed.")

    def __str__(self):
        return f"{self.name} - {self.file_url or 'No URL'}"

