# """
# URL configuration for chat_backend project.

# The `urlpatterns` list routes URLs to views. For more information please see:
#     https://docs.djangoproject.com/en/5.1/topics/http/urls/
# Examples:
# Function views
#     1. Add an import:  from my_app import views
#     2. Add a URL to urlpatterns:  path('', views.home, name='home')
# Class-based views
#     1. Add an import:  from other_app.views import Home
#     2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
# Including another URLconf
#     1. Import the include() function: from django.urls import include, path
#     2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
# """
# from django.contrib import admin
# from django.urls import path
# from django.urls import path, include

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('api/', include('chat.urls')),
#     path('model/', include('result.urls')),
# ]





from django.contrib import admin
from django.urls import path, include

from chat_backend.views import home

urlpatterns = [
    path('', home, name='home'),  # Add the home route
    path('admin/', admin.site.urls),
    path('api/', include('Sql_Notebook.urls')),  # Notebook-related endpoints
    path('api/', include('data_pipeline.urls')),  # Data pipeline endpoints
    path('api/', include('predictions_page.urls')),
    path('api/', include('chat.urls')),  # Chat endpoints
    path('model/', include('result.urls')),
    path('api/auth/', include('accounts.urls')),
    path('api/', include('predictionfile.urls')),

    # path('save-notebooks/', include('save')),  # Save notebooks endpoint
]
