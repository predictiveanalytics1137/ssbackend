from django.db import models

# Create your models here.
from django.db import models

class ModelResult(models.Model):
    model_metrics = models.JSONField()
    attribute_columns = models.JSONField()
    feature_importance = models.JSONField(null=True, blank=True)
    core_statistics = models.JSONField()
    attribute_statistics = models.JSONField()
    predictions = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)


