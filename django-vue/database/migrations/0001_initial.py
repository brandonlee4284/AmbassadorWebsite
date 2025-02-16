# Generated by Django 3.2.9 on 2021-11-14 17:21

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Home',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('annoucement_title', models.CharField(max_length=500)),
                ('date_time', models.CharField(max_length=500)),
                ('description', models.TextField()),
                ('created_date', models.DateTimeField(default=django.utils.timezone.now, verbose_name='date created')),
            ],
        ),
        migrations.CreateModel(
            name='Pod',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pod_group_number', models.CharField(max_length=250)),
                ('pod_leader', models.CharField(max_length=250)),
                ('room_number', models.CharField(max_length=5)),
            ],
        ),
        migrations.CreateModel(
            name='Resources',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('resource_name', models.CharField(max_length=500)),
                ('link', models.CharField(max_length=1000)),
            ],
        ),
        migrations.CreateModel(
            name='Schedule',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('activity', models.CharField(max_length=1000)),
                ('time_slot', models.CharField(max_length=100)),
                ('activity_description', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Student',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('First_Name', models.CharField(max_length=100)),
                ('Last_Name', models.CharField(max_length=100)),
            ],
        ),
    ]
