# from django.db import models
# from django.contrib.auth.models import User

# class PredictionFileInfo(models.Model):
#     """Store metadata for prediction datasets uploaded by users."""
#     user = models.ForeignKey(User, on_delete=models.CASCADE)
#     chat = models.CharField(max_length=255, blank=True)  # Optional chat context
#     file = models.ForeignKey('predictionfile.UploadedFile', on_delete=models.CASCADE)
#     name = models.CharField(max_length=255)
#     file_url = models.URLField()
#     schema = models.JSONField(default=list)  # Store schema as JSON
#     has_date_column = models.BooleanField(default=False)
#     date_columns = models.JSONField(default=list)  # Store possible date columns
#     created_at = models.DateTimeField(auto_now_add=True)

#     class Meta:
#         ordering = ['-created_at']



from django.db import models
from django.contrib.auth.models import User

class PredictionFileInfo(models.Model):
    """Store metadata for prediction datasets uploaded by users."""
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    chat = models.CharField(max_length=255, blank=True)  # Optional chat context
    file = models.ForeignKey('predictionfile.UploadedFile', on_delete=models.CASCADE)
    name = models.CharField(max_length=255)
    file_url = models.URLField()
    schema = models.JSONField(default=list)  # Store schema as JSON
    has_date_column = models.BooleanField(default=False)
    date_columns = models.JSONField(default=list)  # Store possible date columns
    created_at = models.DateTimeField(auto_now_add=True)
    prediction_s3_links = models.JSONField(default=dict)  # Store S3 URLs for prediction results

    class Meta:
        ordering = ['-created_at']

class UploadedFile(models.Model):
    """Store uploaded prediction files."""
    name = models.CharField(max_length=255)
    file_url = models.URLField()
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.name