# Generated by Django 3.2.9 on 2021-11-16 06:32

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0005_alter_schedule_activity'),
    ]

    operations = [
        migrations.CreateModel(
            name='HomeImages',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image_name', models.CharField(max_length=100)),
                ('url_of_image', models.CharField(max_length=1000)),
                ('image_hyperlink', models.CharField(max_length=1000)),
            ],
        ),
    ]
