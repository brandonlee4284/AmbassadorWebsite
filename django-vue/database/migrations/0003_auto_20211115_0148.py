# Generated by Django 3.2.9 on 2021-11-15 01:48

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('database', '0002_remove_pod_room_number'),
    ]

    operations = [
        migrations.AddField(
            model_name='pod',
            name='pod_group_members',
            field=models.TextField(default=101),
            preserve_default=False,
        ),
        migrations.AddField(
            model_name='pod',
            name='pod_room_number',
            field=models.CharField(default=101, max_length=250),
            preserve_default=False,
        ),
    ]
