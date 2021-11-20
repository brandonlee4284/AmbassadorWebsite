from django.urls import path
from . views import PodView, HomeView, ResourceView, ScheduleView, HomeImageView
from rest_framework_simplejwt.views import (TokenObtainPairView,TokenRefreshView)


urlpatterns = [
    path('api-token/', TokenObtainPairView.as_view()),
    path('api-token-refresh/', TokenRefreshView.as_view()),
    path('pod/', PodView.as_view(), name='pod_view'),
    path('home/', HomeView.as_view(), name='home_view'),
    path('resources/', ResourceView.as_view(), name='resources_view'),
    path('schedule/', ScheduleView.as_view(), name='schedule_view'),
    path('homeImage/', HomeImageView.as_view(), name='homeImage_view'),
]