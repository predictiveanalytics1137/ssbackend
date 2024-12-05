from rest_framework import serializers
from .models import ModelResult

class ModelResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = ModelResult
        fields = '__all__'
