# Generated by Django 4.2.17 on 2024-12-17 15:44

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0008_chatbackup_created_at_chatbackup_updated_at_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='message',
            name='chat',
        ),
        migrations.DeleteModel(
            name='Chat',
        ),
        migrations.DeleteModel(
            name='Message',
        ),
    ]