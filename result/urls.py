from django.urls import path
from .views import GetModelResultAPIView, ModelResultView

urlpatterns = [
    path('modelresults/', ModelResultView.as_view(), name='model-results'),
    path('modelget/<int:id>/', GetModelResultAPIView.as_view(), name='get_model_result'),
    # path('modelresults/', hello, name='model-results'),
]
