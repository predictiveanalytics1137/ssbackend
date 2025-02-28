from django.urls import path
from .views import DataForAutomationAPI, DataForPredictionsAPI, TrainingStatusAPI, book_demo

urlpatterns = [
    path('automation/', DataForAutomationAPI.as_view(), name='data-for-automation'),
    path('automation/status/<str:task_id>/', TrainingStatusAPI.as_view(), name='training-status'),
    # path('prediction/', PredictDataAPI.as_view(), name='data-for-prediction'),
    path('prediction/', DataForPredictionsAPI.as_view(), name='data-for-prediction'),
    path('bookdemo/', book_demo, name='book-demo'),
    
]
