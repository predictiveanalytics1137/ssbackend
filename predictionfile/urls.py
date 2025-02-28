from django.urls import path
from .views import PredictionDatasetUploadAPI, SavePredictionResultsView

urlpatterns = [
    path('predict/', PredictionDatasetUploadAPI.as_view(), name='prediction_upload'),
    path('save_prediction_results/', SavePredictionResultsView.as_view(), name='save_prediction_results'),

    
]