# from django.shortcuts import render

# # Create your views here.
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from .models import ModelResult
# from .serializers import ModelResultSerializer


# class ModelResultView(APIView):
#     def post(self, request, *args, **kwargs):
#         serializer = ModelResultSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
#         # return Response("hello world")
    
# # def hello():
# #     return Response("hello world")


# class GetModelResultAPIView(APIView):
#     def get(self, request, id):
#         try:
#             # Retrieve the model result by ID
#             model_result = ModelResult.objects.get(pk=id)
#             # Serialize the result
#             serializer = ModelResultSerializer(model_result)
#             return Response(serializer.data, status=status.HTTP_200_OK)
#         except ModelResult.DoesNotExist:
#             # If the record doesn't exist, return a 404 error
#             return Response({"error": "ModelResult not found"}, status=status.HTTP_404_NOT_FOUND)



# modelresults/views.py
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import ModelResult
from .serializers import ModelResultSerializer

class ModelResultView(APIView):
    def post(self, request, *args, **kwargs):
        serializer = ModelResultSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

class GetModelResultByUserChatAPI(APIView):
    def get(self, request):
        user_id = request.query_params.get('user_id')
        chat_id = request.query_params.get('chat_id')

        if not user_id or not chat_id:
            return Response({"error": "user_id and chat_id are required"}, status=status.HTTP_400_BAD_REQUEST)

        # Retrieve the latest model result for this user_id and chat_id
        try:
            model_result = ModelResult.objects.filter(user_id=user_id, chat_id=chat_id).latest('created_at')
            serializer = ModelResultSerializer(model_result)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except ModelResult.DoesNotExist:
            return Response({"error": "No model result found for the given user_id and chat_id"}, status=status.HTTP_404_NOT_FOUND)
