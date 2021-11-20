from django.db import models
from django.utils import timezone


class Student(models.Model):
  First_Name = models.CharField(max_length=100)
  Last_Name = models.CharField(max_length=100)

  def __str__(self):
    return self.First_Name

class Pod(models.Model):
  pod_group_number = models.CharField(max_length=10)
  pod_leader = models.CharField(max_length=100)
  pod_room_number = models.CharField(max_length=10)
  pod_group_members = models.TextField()
  

  def __str__(self):
    return self.pod_group_number

class Schedule(models.Model):
  activity = models.CharField(max_length=15)
  time_slot = models.CharField(max_length=20)
  activity_description = models.TextField()

  def __str__(self):
    return self.activity

class Resources(models.Model):
  resource_name = models.CharField(max_length=250)
  link = models.CharField(max_length=1000)

  def __str__(self):
    return self.resource_name

class Home(models.Model):
  annoucement_title = models.CharField(max_length=250)
  date_time = models.CharField(max_length=500)
  description = models.TextField()
  created_date = models.DateTimeField('date created', default=timezone.now)
  
  def __str__(self):
    return self.annoucement_title


class HomeImage(models.Model):
  image_name = models.CharField(max_length=25)
  url_of_image = models.CharField(max_length=500)
  image_redirect = models.CharField(max_length=500)
  
  
  def __str__(self):
    return self.image_name