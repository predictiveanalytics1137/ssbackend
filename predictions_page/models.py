from django.db import models

# Create your models here.


# prediction page
# from django.db import models

# class PredictionMetadata(models.Model):
#     prediction_id = models.CharField(max_length=255)
#     chat_id = models.CharField(max_length=255)
#     user_id = models.CharField(max_length=255)
#     start_time = models.DateTimeField(auto_now_add=True)
#     status = models.CharField(max_length=50, choices=[('inprogress', 'In Progress'), ('success', 'Success'), ('failed', 'Failed')])
#     duration = models.FloatField(null=True, blank=True)  # Duration in seconds
#     entity_count = models.IntegerField()
#     predictions_csv_path = models.CharField(max_length=1024, null=True, blank=True)

#     def __str__(self):
#         return f"PredictionMetadata(chat_id={self.chat_id}, status={self.status})"

from django.db import models

class PredictionMetadata(models.Model):
    prediction_id = models.CharField(max_length=255, unique=True)  # Unique identifier for each prediction
    chat_id = models.CharField(max_length=255, db_index=True)  # Index for faster lookup
    user_id = models.CharField(max_length=255, db_index=True)  # Index for faster lookup
    start_time = models.DateTimeField(auto_now_add=True)  # Automatically set when the record is created
    status = models.CharField(
        max_length=50, 
        choices=[('inprogress', 'In Progress'), ('success', 'Success'), ('failed', 'Failed')]
    )
    duration = models.FloatField(null=True, blank=True)  # Duration in seconds
    entity_count = models.IntegerField()  # Number of entities in the dataset
    predictions_csv_path = models.CharField(max_length=1024, null=True, blank=True)  # Path to the CSV file in S3
    predictions_data = models.JSONField(null=True, blank=True)  # New field to store predictions as JSON

    def __str__(self):
        return f"PredictionMetadata(chat_id={self.chat_id}, user_id={self.user_id}, status={self.status})"
