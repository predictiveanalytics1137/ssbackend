# Generated by Django 5.1.2 on 2025-02-05 18:09

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0013_rename_chat_id_notebook_chat'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='PredictiveSettings',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('chat_id', models.CharField(max_length=255)),
                ('target_column', models.CharField(blank=True, max_length=255, null=True)),
                ('entity_column', models.CharField(blank=True, max_length=255, null=True)),
                ('time_frame', models.CharField(blank=True, max_length=255, null=True)),
                ('predictive_question', models.TextField(blank=True, null=True)),
                ('time_column', models.CharField(blank=True, max_length=255, null=True)),
                ('time_frequency', models.CharField(blank=True, max_length=255, null=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
