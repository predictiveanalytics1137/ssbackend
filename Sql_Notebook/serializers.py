# # notebook/serializers.py

# from rest_framework import serializers

# class SQLQuerySerializer(serializers.Serializer):
#     query = serializers.CharField()


# your_app/serializers.py

from rest_framework import serializers

class SQLQuerySerializer(serializers.Serializer):
    query = serializers.CharField()

    def validate_query(self, value):
        """
        Validates that the SQL query starts with 'SELECT'.
        """
        if not value.strip().lower().startswith('select'):
            raise serializers.ValidationError("Only SELECT queries are allowed.")
        return value
