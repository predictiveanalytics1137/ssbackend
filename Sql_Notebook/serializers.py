# notebook/serializers.py

from rest_framework import serializers

class SQLQuerySerializer(serializers.Serializer):
    query = serializers.CharField()
