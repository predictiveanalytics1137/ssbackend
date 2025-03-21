# Generated by Django 5.1.6 on 2025-02-27 10:55

import django.db.models.deletion
import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('predictionfile', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedfile',
            name='uploaded_at',
            field=models.DateTimeField(auto_now_add=True, default=django.utils.timezone.now),
            preserve_default=False,
        ),
        migrations.AlterField(
            model_name='predictionfileinfo',
            name='file',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='predictionfile.uploadedfile'),
        ),
    ]
