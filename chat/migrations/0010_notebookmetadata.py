# Generated by Django 5.1.2 on 2025-01-14 07:10

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0009_remove_message_chat_delete_chat_delete_message'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='NotebookMetadata',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('chat_id', models.CharField(max_length=100)),
                ('notebook_type', models.CharField(max_length=50)),
                ('entity_column', models.CharField(max_length=100)),
                ('target_column', models.CharField(max_length=100)),
                ('time_column', models.CharField(blank=True, max_length=100, null=True)),
                ('time_frame', models.CharField(blank=True, max_length=50, null=True)),
                ('features', models.JSONField(default=list)),
                ('metadata_created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
