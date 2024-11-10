# serializers.py
from rest_framework import serializers
from .models import UploadedFile

from django.conf import settings

class UploadedFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = UploadedFile
        fields = '__all__'

    def create(self, validated_data):
        instance = super().create(validated_data)
        # Add the full URL to the file
        instance.file_url = f"{settings.MEDIA_URL}{instance.file.name}"
        instance.save()
        return instance
