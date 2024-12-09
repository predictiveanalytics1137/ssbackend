

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