from django.urls import path
from .views import GetModelResultByUserChatAPI, ModelResultView

urlpatterns = [
    path('modelresults/', ModelResultView.as_view(), name='model-results'),
    path('modelget/', GetModelResultByUserChatAPI.as_view(), name='get_model_result'),
    # path('modelresults/', hello, name='model-results'),
]
