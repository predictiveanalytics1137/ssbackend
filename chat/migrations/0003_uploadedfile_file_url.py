# Generated by Django 5.1.2 on 2024-11-10 13:06

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0002_uploadedfile'),
    ]

    operations = [
        migrations.AddField(
            model_name='uploadedfile',
            name='file_url',
            field=models.TextField(blank=True, null=True),
        ),
    ]
