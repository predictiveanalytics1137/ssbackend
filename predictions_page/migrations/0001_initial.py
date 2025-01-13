# Generated by Django 5.1.2 on 2025-01-10 09:03

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='PredictionMetadata',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('chat_id', models.CharField(max_length=255)),
                ('start_time', models.DateTimeField(auto_now_add=True)),
                ('status', models.CharField(choices=[('inprogress', 'In Progress'), ('success', 'Success'), ('failed', 'Failed')], max_length=50)),
                ('duration', models.FloatField(blank=True, null=True)),
                ('entity_count', models.IntegerField()),
                ('predictions_csv_path', models.CharField(blank=True, max_length=1024, null=True)),
            ],
        ),
    ]