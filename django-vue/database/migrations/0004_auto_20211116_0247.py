# Generated by Django 3.2.9 on 2021-11-16 02:47

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0003_auto_20211115_0148'),
    ]

    operations = [
        migrations.AlterField(
            model_name='home',
            name='annoucement_title',
            field=models.CharField(max_length=250),
        ),
        migrations.AlterField(
            model_name='pod',
            name='pod_group_number',
            field=models.CharField(max_length=10),
        ),
        migrations.AlterField(
            model_name='pod',
            name='pod_leader',
            field=models.CharField(max_length=100),
        ),
        migrations.AlterField(
            model_name='pod',
            name='pod_room_number',
            field=models.CharField(max_length=10),
        ),
        migrations.AlterField(
            model_name='resources',
            name='resource_name',
            field=models.CharField(max_length=250),
        ),
        migrations.AlterField(
            model_name='schedule',
            name='activity',
            field=models.CharField(max_length=20),
        ),
        migrations.AlterField(
            model_name='schedule',
            name='time_slot',
            field=models.CharField(max_length=20),
        ),
    ]
