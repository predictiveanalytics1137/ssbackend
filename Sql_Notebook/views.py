# notebook/views.py

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status, permissions, authentication
from django.db import connection
from .serializers import SQLQuerySerializer

class ExecuteSQLView(APIView):
    authentication_classes = [authentication.TokenAuthentication]
    permission_classes = [permissions.IsAuthenticated]

    def post(self, request):
        serializer = SQLQuerySerializer(data=request.data)
        if serializer.is_valid():
            query = serializer.validated_data['query'].strip()
            if not query.lower().startswith('select'):
                return Response({'error': 'Only SELECT queries are allowed.'}, status=status.HTTP_400_BAD_REQUEST)
            try:
                with connection.cursor() as cursor:
                    cursor.execute(query)
                    columns = [col[0] for col in cursor.description]
                    rows = cursor.fetchall()
                    data = {
                        'columns': columns,
                        'rows': rows,
                    }
                    return Response(data, status=status.HTTP_200_OK)
            except Exception as e:
                return Response({'error': str(e)}, status=status.HTTP_400_BAD_REQUEST)
        else:
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
