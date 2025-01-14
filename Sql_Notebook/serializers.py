
# from rest_framework import serializers

# class SQLQuerySerializer(serializers.Serializer):
#     query = serializers.CharField()

#     def validate_query(self, value):
#         """
#         Validates that the SQL query starts with 'SELECT'.
#         """
#         if not value.strip().lower().startswith('select'):
#             raise serializers.ValidationError("Only SELECT queries are allowed.")
#         return value



from rest_framework import serializers

class SQLQuerySerializer(serializers.Serializer):
    query = serializers.CharField()

    def validate_query(self, value):
        """
        Validates that the SQL query starts with 'SELECT' or 'WITH'.
        """
        # Remove leading/trailing whitespace
        cleaned = value.strip().lower()

        # Also skip leading comment lines if you want to handle that in the serializer
        # Or you can just do the minimal check:
        allowed_starts = ("select", "with")
        if not any(cleaned.startswith(start) for start in allowed_starts):
            raise serializers.ValidationError("Only read-only queries (SELECT or WITH) are allowed.")

        return value
