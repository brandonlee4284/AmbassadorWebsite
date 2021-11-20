from django.shortcuts import render
from rest_framework.response import Response
from rest_framework import generics
from . models import Pod, Home, Resources, Schedule, HomeImage
from . serializers import PodSerializer, HomeSerializer, ResourcesSerializer, ScheduleSerializer, HomeImageSerializer
from rest_framework.permissions import IsAuthenticated

# RetrieveAPIView - used for read-only endpoints to represent a single model instance.
# https://www.django-rest-framework.org/api-guide/generic-views/#retrieveapiview
class PodView(generics.RetrieveAPIView):
    permission_classes = (IsAuthenticated,)  
    queryset = Pod.objects.all()

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = PodSerializer(queryset, many=True)
        return Response(serializer.data)

class HomeView(generics.RetrieveAPIView):
    permission_classes = (IsAuthenticated,)  
    queryset = Home.objects.all()

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = HomeSerializer(queryset, many=True)
        return Response(serializer.data)

class ResourceView(generics.RetrieveAPIView):
    permission_classes = (IsAuthenticated,)  
    queryset = Resources.objects.all()

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = ResourcesSerializer(queryset, many=True)
        return Response(serializer.data)

class ScheduleView(generics.RetrieveAPIView):
    permission_classes = (IsAuthenticated,)  
    queryset = Schedule.objects.all()

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = ScheduleSerializer(queryset, many=True)
        return Response(serializer.data)

class HomeImageView(generics.RetrieveAPIView):
    permission_classes = (IsAuthenticated,)  
    queryset = HomeImage.objects.all()

    def get(self, request, *args, **kwargs):
        queryset = self.get_queryset()
        serializer = HomeImageSerializer(queryset, many=True)
        return Response(serializer.data)