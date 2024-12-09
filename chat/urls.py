# from django.urls import path

# from Sql_Notebook.views import ExecuteSQLView
# from . import views

# urlpatterns = [
#     # path('chat/', views.chat_response, name='chat_response'),
#     # path('upload/', views.FileUploadView.as_view(), name='file-upload'),  # Added .as_view() for class-based view
#     # path('delete/<int:pk>/', views.FileDeleteView.as_view(), name='file-delete'),  # Added .as_view() for class-based view
#     path('execute-sql/', ExecuteSQLView.as_view(), name='execute-sql'),
#     path('chatgpt/', views.UnifiedChatGPTAPI.as_view(), name='chatgpt_chat'),

# ]




from django.urls import path
from .views import UnifiedChatGPTAPI

urlpatterns = [
    path('chatgpt/', UnifiedChatGPTAPI.as_view(), name='chatgpt_chat'),  # ChatGPT-related endpoint
]
