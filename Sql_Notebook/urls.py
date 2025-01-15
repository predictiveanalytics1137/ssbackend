
# from django.contrib import admin
# from django.urls import path, include

# urlpatterns = [
#     path('admin/', admin.site.urls),
#     path('api/', include('Sql_Notebook.urls')),
# ]



from django.urls import path
from .views import ExecuteSQLView, SaveNotebooksView

urlpatterns = [
    path('execute-sql/', ExecuteSQLView.as_view(), name='execute-sql'),  # SQL execution endpoint
    path('save-notebooks/', SaveNotebooksView.as_view(), name='save-notebooks'),  # Save notebooks endpoint
]
