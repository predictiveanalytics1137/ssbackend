# Generated by Django 5.1.2 on 2025-02-05 19:21

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('chat', '0014_predictivesettings'),
    ]

    operations = [
        migrations.AddField(
            model_name='predictivesettings',
            name='machine_learning_type',
            field=models.CharField(blank=True, max_length=50, null=True),
        ),
    ]
