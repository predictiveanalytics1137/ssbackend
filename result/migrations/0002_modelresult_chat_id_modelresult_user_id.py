# Generated by Django 4.2.17 on 2024-12-17 15:29

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('result', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='modelresult',
            name='chat_id',
            field=models.CharField(default=0, max_length=50),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='modelresult',
            name='user_id',
            field=models.CharField(default=1, max_length=50),
            preserve_default=False,
        ),
    ]