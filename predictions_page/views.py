from django.shortcuts import render

# Create your views here.
# prediction page

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PredictionMetadata

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PredictionMetadata

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PredictionMetadata

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PredictionMetadata


class UpdatePredictionStatusView(APIView):
    def post(self, request):
        """
        Handles creation of new prediction metadata.
        """
        try:
            data = request.data
            prediction_id = data.get("prediction_id")

            if PredictionMetadata.objects.filter(prediction_id=prediction_id).exists():
                return Response({"error": "Prediction ID already exists."}, status=status.HTTP_400_BAD_REQUEST)

            PredictionMetadata.objects.create(
                prediction_id=prediction_id,
                chat_id=data["chat_id"],
                user_id=data["user_id"],
                status=data["status"],
                entity_count=data["entity_count"],
                start_time=data.get("start_time")
            )
            return Response({"message": "Metadata created successfully."}, status=status.HTTP_201_CREATED)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def patch(self, request, prediction_id):
        """
        Handles updates to existing prediction metadata.
        """
        try:
            metadata = PredictionMetadata.objects.get(prediction_id=prediction_id)
            for key, value in request.data.items():
                setattr(metadata, key, value)
            metadata.save()
            return Response({"message": "Metadata updated successfully."}, status=status.HTTP_200_OK)
        except PredictionMetadata.DoesNotExist:
            return Response({"error": "Prediction ID not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)






from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PredictionMetadata

class GetPredictionMetadataView(APIView):
    def get(self, request):
        """
        Retrieve prediction metadata based on chat_id, user_id, or both.
        """
        try:
            chat_id = request.query_params.get('chat_id')
            user_id = request.query_params.get('user_id')

            # Validate input: At least one parameter must be provided
            if not chat_id and not user_id:
                return Response({"error": "At least one of chat_id or user_id is required."}, status=status.HTTP_400_BAD_REQUEST)

            # Filter by the provided parameters
            query = {}
            if chat_id:
                query['chat_id'] = chat_id
            if user_id:
                query['user_id'] = user_id

            metadata = PredictionMetadata.objects.filter(**query).order_by('-start_time')
            if not metadata.exists():
                return Response({"message": "No prediction metadata found for the provided parameters."}, status=status.HTTP_404_NOT_FOUND)

            # Serialize the data
            metadata_list = [
                {
                    "prediction_id": item.prediction_id,
                    "chat_id": item.chat_id,
                    "user_id": item.user_id,
                    "start_time": item.start_time,
                    "status": item.status,
                    "duration": item.duration,
                    "entity_count": item.entity_count,
                    "predictions_csv_path": item.predictions_csv_path,
                }
                for item in metadata
            ]

            return Response({"metadata": metadata_list}, status=status.HTTP_200_OK)

        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
