from django.db import models


class ChatHistory(models.Model):
    user_message = models.TextField()
    assistant_response = models.TextField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"User: {self.user_message[:30]} - Assistant: {self.assistant_response[:30]}"
