from django.urls import path
from .views import DataForAutomationAPI

urlpatterns = [
    path('automation/', DataForAutomationAPI.as_view(), name='data-for-automation'),
]
