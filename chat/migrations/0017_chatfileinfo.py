# Generated by Django 5.1.2 on 2025-02-10 21:17

import django.db.models.deletion
from django.conf import settings
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0016_notebook_cell_s3_links'),
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ChatFileInfo',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=255)),
                ('file_url', models.CharField(max_length=1024)),
                ('schema', models.JSONField()),
                ('suggestions', models.JSONField(blank=True, null=True)),
                ('has_date_column', models.BooleanField(default=False)),
                ('date_columns', models.JSONField(blank=True, null=True)),
                ('glue_table_name', models.CharField(blank=True, max_length=255, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('chat', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='file_infos', to='chat.chatbackup')),
                ('file', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to='chat.uploadedfile')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='chat_file_infos', to=settings.AUTH_USER_MODEL)),
            ],
        ),
    ]
