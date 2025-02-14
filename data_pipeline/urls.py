from django.urls import path
from .views import DataForAutomationAPI, PredictDataAPI, TrainingStatusAPI, book_demo

urlpatterns = [
    path('automation/', DataForAutomationAPI.as_view(), name='data-for-automation'),
    path('automation/status/<str:task_id>/', TrainingStatusAPI.as_view(), name='training-status'),
    path('prediction/', PredictDataAPI.as_view(), name='data-for-prediction'),
    path('bookdemo/', book_demo, name='book-demo'),
    
]
