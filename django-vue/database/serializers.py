from rest_framework import serializers
from .models import Pod, Schedule, Home, Resources, HomeImage


class PodSerializer(serializers.ModelSerializer):
    class Meta:
        model = Pod
        fields = '__all__'

class ScheduleSerializer(serializers.ModelSerializer):
    class Meta:
        model = Schedule
        fields = '__all__'

class HomeSerializer(serializers.ModelSerializer):
    class Meta:
        model = Home
        fields = '__all__'

class ResourcesSerializer(serializers.ModelSerializer):
    class Meta:
        model = Resources
        fields = '__all__'

class HomeImageSerializer(serializers.ModelSerializer):
    class Meta:
        model = HomeImage
        fields = '__all__'
