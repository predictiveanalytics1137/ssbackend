
# from django.contrib import admin
# from django.urls import path, include

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('api/', include('Sql_Notebook.urls')),
# ]



from django.urls import path
from .views import UpdatePredictionStatusView,GetPredictionMetadataView

# urlpatterns = [
#     path('update_prediction_status/', UpdatePredictionStatus.as_view(), name='update_prediction_status'),
#     path('update_prediction_status/<str:prediction_id>/', UpdatePredictionStatusView.as_view(), name='update_prediction_status_detail'),
# ]

urlpatterns = [
    path('update_prediction_status/', UpdatePredictionStatusView.as_view(), name='update_prediction_status'),  # POST endpoint
    path('update_prediction_status/<str:prediction_id>/', UpdatePredictionStatusView.as_view(), name='update_prediction_status_detail'),  # PATCH endpoint
    path('get_prediction_metadata/', GetPredictionMetadataView.as_view(), name='get_prediction_metadata'),
]

