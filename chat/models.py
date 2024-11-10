from django.db import models


class ChatHistory(models.Model):
    user_message = models.TextField()
    assistant_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User: {self.user_message[:30]} - Assistant: {self.assistant_response[:30]}"


class UploadedFile(models.Model):
    name = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    file_url = models.TextField(null=True, blank=True)  # Store the full S3 URL

    def __str__(self):
        return self.name
