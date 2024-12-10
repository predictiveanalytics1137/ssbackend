from django.urls import path
from .views import DataForAutomationAPI, PredictDataAPI

urlpatterns = [
    path('automation/', DataForAutomationAPI.as_view(), name='data-for-automation'),
    path('prediction/', PredictDataAPI.as_view(), name='data-for-prediction'),
    
]
