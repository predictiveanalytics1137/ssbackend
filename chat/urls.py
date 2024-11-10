from django.urls import path
from . import views

urlpatterns = [
    path('chat/', views.chat_response, name='chat_response'),
    path('upload/', views.FileUploadView.as_view(), name='file-upload'),  # Added .as_view() for class-based view
    path('delete/<int:pk>/', views.FileDeleteView.as_view(), name='file-delete'),  # Added .as_view() for class-based view
]
