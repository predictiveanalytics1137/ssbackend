# from django.shortcuts import render

# # Create your views here.


# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework_simplejwt.tokens import RefreshToken
# from django.contrib.auth import authenticate
# from .serializers import UserSerializer

# class RegisterView(APIView):
#     def post(self, request):
#         serializer = UserSerializer(data=request.data)
#         if serializer.is_valid():
#             serializer.save()
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
#         return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

# # class LoginView(APIView):
# #     def post(self, request):
# #         username = request.data.get('username')
# #         password = request.data.get('password')
# #         user = authenticate(username=username, password=password)
# #         if user is not None:
# #             refresh = RefreshToken.for_user(user)
# #             return Response({
# #                 'refresh': str(refresh),
# #                 'access': str(refresh.access_token),
# #             })
# #         return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)


# from django.contrib.auth import authenticate

# # class LoginView(APIView):
# #     def post(self, request):
# #         username = request.data.get('username')
# #         password = request.data.get('password')
# #         user = authenticate(username=username, password=password)  # Verifies hashed password
# #         if user is not None:
# #             refresh = RefreshToken.for_user(user)
# #             return Response({
# #                 'refresh': str(refresh),
# #                 'access': str(refresh.access_token),
# #             })
# #         return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)


# class LoginView(APIView):
#     def post(self, request):
#         username = request.data.get('username')
#         password = request.data.get('password')
#         user = authenticate(username=username, password=password)

#         if user is not None:
#             refresh = RefreshToken.for_user(user)
#             # Add user information dynamically
#             return Response({
#                 'refresh': str(refresh),
#                 'access': str(refresh.access_token),
#                 'user': {
#                     'id': user.id,
#                     'username': user.username,
#                     'email': user.email,  # Optional, include as needed
#                     'is_superuser': user.is_superuser,  # Example for roles
#                 }
#             })
#         return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)





# Views
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework_simplejwt.tokens import RefreshToken
from django.contrib.auth import authenticate
from .serializers import UserSerializer


class RegisterView(APIView):
    def post(self, request):
        print("RegisterView POST called.")
        print("Request data:", request.data)  # Debugging input data

        serializer = UserSerializer(data=request.data)
        if serializer.is_valid():
            print("Serializer is valid.")
            user = serializer.save()
            print(f"User {user.username} registered successfully.")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            print("Serializer errors:", serializer.errors)  # Debugging validation errors
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)


class LoginView(APIView):
    def post(self, request):
        print("LoginView POST called.")
        print("Request data:", request.data)  # Debugging input data

        username = request.data.get('username')
        password = request.data.get('password')
        print(f"Authenticating user: {username}")

        user = authenticate(username=username, password=password)
        if user is not None:
            print(f"User {user.username} authenticated successfully.")
            refresh = RefreshToken.for_user(user)
            print("JWT tokens generated successfully.")
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'user': {
                    'id': user.id,
                    'username': user.username,
                    'email': user.email,  # Optional
                    'is_superuser': user.is_superuser,  # Example
                }
            })
        else:
            print("Authentication failed for user:", username)  # Debugging failed authentication
            return Response({"error": "Invalid credentials"}, status=status.HTTP_401_UNAUTHORIZED)