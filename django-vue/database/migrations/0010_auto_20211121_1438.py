# Generated by Django 3.2.9 on 2021-11-21 22:38

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0009_delete_homeimage'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='home',
            name='created_date',
        ),
        migrations.AddField(
            model_name='home',
            name='image_redirect',
            field=models.CharField(default=0, max_length=500),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='home',
            name='image_url',
            field=models.CharField(default=0, max_length=500),
            preserve_default=False,
        ),
    ]
