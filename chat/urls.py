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
from .views import ChatListView, MessageListView, UnifiedChatGPTAPI, ChatHistoryByUserView

urlpatterns = [
    path('chatgpt/', UnifiedChatGPTAPI.as_view(), name='chatgpt_chat'),  # ChatGPT-related endpoint
    # path('chats/', ChatListView.as_view(), name='chat_list'),
    # path('chats/<uuid:chat_id>/messages/', MessageListView.as_view(), name='message_list'),
    path('chats/', ChatListView.as_view(), name='chat-list'),
    path('chat_history/', ChatHistoryByUserView.as_view(), name='chat_history'),
]


