# Generated by Django 5.1.2 on 2025-01-14 19:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0011_notebook_delete_notebookmetadata'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='notebook',
            name='chat',
        ),
        migrations.AddField(
            model_name='notebook',
            name='chat_id',
            field=models.CharField(default=1, max_length=255),
            preserve_default=False,
        ),
    ]
