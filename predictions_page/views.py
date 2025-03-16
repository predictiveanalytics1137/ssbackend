from django.shortcuts import render

# Create your views here.
# prediction page




from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import PredictionMetadata



# class UpdatePredictionStatusView(APIView):
#     def post(self, request):
#         """
#         Handles creation of new prediction metadata.
#         """
#         try:
#             data = request.data
#             prediction_id = data.get("prediction_id")

#             if PredictionMetadata.objects.filter(prediction_id=prediction_id).exists():
#                 return Response({"error": "Prediction ID already exists."}, status=status.HTTP_400_BAD_REQUEST)

#             PredictionMetadata.objects.create(
#                 prediction_id=prediction_id,
#                 chat_id=data["chat_id"],
#                 user_id=data["user_id"],
#                 status=data["status"],
#                 entity_count=data["entity_count"],
#                 start_time=data.get("start_time")
#             )
#             return Response({"message": "Metadata created successfully."}, status=status.HTTP_201_CREATED)
#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

#     def patch(self, request, prediction_id):
#         """
#         Handles updates to existing prediction metadata.
#         """
#         try:
#             metadata = PredictionMetadata.objects.get(prediction_id=prediction_id)
#             for key, value in request.data.items():
#                 setattr(metadata, key, value)
#             metadata.save()
#             return Response({"message": "Metadata updated successfully."}, status=status.HTTP_200_OK)
#         except PredictionMetadata.DoesNotExist:
#             return Response({"error": "Prediction ID not found."}, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

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
        Handles updates to prediction metadata.
        """
        try:
            prediction = PredictionMetadata.objects.get(prediction_id=prediction_id)
            data = request.data

            prediction.status = data.get("status", prediction.status)
            prediction.duration = data.get("duration", prediction.duration)
            prediction.predictions_csv_path = data.get("predictions_csv_path", prediction.predictions_csv_path)
            prediction.predictions_data = data.get("predictions_data", prediction.predictions_data)
            prediction.save()

            return Response({"message": "Metadata updated successfully."}, status=status.HTTP_200_OK)
        except PredictionMetadata.DoesNotExist:
            return Response({"error": "Prediction ID not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

    def get(self, request, prediction_id):
        """
        Fetches prediction metadata by prediction_id.
        """
        try:
            prediction = PredictionMetadata.objects.get(prediction_id=prediction_id)
            data = {
                'prediction_id': prediction.prediction_id,
                'chat_id': prediction.chat_id,
                'user_id': prediction.user_id,
                'status': prediction.status,
                'duration': prediction.duration,
                'entity_count': prediction.entity_count,
                'predictions_csv_path': prediction.predictions_csv_path,
                'predictions_data': prediction.predictions_data,
                'start_time': prediction.start_time.isoformat(),
            }
            return Response(data, status=status.HTTP_200_OK)
        except PredictionMetadata.DoesNotExist:
            return Response({"error": "Prediction ID not found."}, status=status.HTTP_404_NOT_FOUND)
        except Exception as e:
            return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status  # Correctly import the status module

# class UpdatePredictionStatusView(APIView):
#     def post(self, request):
#         """
#         Handles creation of new metadata (training or prediction).
#         """
#         try:
#             data = request.data
#             chat_id = data.get("chat_id")
#             user_id = data.get("user_id")
#             status_value = data.get("status")  # Avoid shadowing the imported `status`
#             entity_count = data.get("entity_count")
#             start_time = data.get("start_time")
#             prediction_id = data.get("prediction_id", None)  # Optional for training

#             # Validate required fields
#             if not chat_id or not user_id or not status_value or not entity_count or not start_time:
#                 return Response({"error": "Missing required fields"}, status=status.HTTP_400_BAD_REQUEST)

#             # Ensure uniqueness for predictions (if prediction_id is provided)
#             if prediction_id and PredictionMetadata.objects.filter(prediction_id=prediction_id).exists():
#                 return Response({"error": "Prediction ID already exists."}, status=status.HTTP_400_BAD_REQUEST)

#             # Create metadata
#             metadata = PredictionMetadata.objects.create(
#                 prediction_id=prediction_id if prediction_id else None,
#                 chat_id=chat_id,
#                 user_id=user_id,
#                 status=status_value,
#                 entity_count=entity_count,
#                 start_time=start_time,
#             )
#             return Response({"message": "Metadata created successfully.", "id": metadata.id}, status=status.HTTP_201_CREATED)

#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)

#     def patch(self, request, prediction_id=None):
#         """
#         Handles updates to existing metadata (training or prediction).
#         """
#         try:
#             chat_id = request.data.get("chat_id")
#             if prediction_id:
#                 metadata = PredictionMetadata.objects.get(prediction_id=prediction_id)
#             elif chat_id:
#                 metadata = PredictionMetadata.objects.filter(chat_id=chat_id).latest('start_time')
#             else:
#                 return Response({"error": "Either prediction_id or chat_id is required"}, status=status.HTTP_400_BAD_REQUEST)

#             for key, value in request.data.items():
#                 if key != "prediction_id":
#                     setattr(metadata, key, value)
#             metadata.save()
#             return Response({"message": "Metadata updated successfully."}, status=status.HTTP_200_OK)

#         except PredictionMetadata.DoesNotExist:
#             return Response({"error": "Metadata not found."}, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             return Response({"error": str(e)}, status=status.HTTP_400_BAD_REQUEST)



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
            prediction_id = request.query_params.get('prediction_id')

            # Validate input: At least one parameter must be provided
            if not chat_id and not user_id and not prediction_id:
                return Response({"error": "At least one of chat_id or user_id or prediction_id is required."}, status=status.HTTP_400_BAD_REQUEST)

            # Filter by the provided parameters
            query = {}
            if chat_id:
                query['chat_id'] = chat_id
            if user_id:
                query['user_id'] = user_id
            if prediction_id:
                query['prediction_id'] = prediction_id

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
