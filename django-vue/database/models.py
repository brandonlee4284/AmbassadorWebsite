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
  additional_notes = models.TextField(blank=True)
  gender_ratio = models.CharField(max_length=200,blank=True)
  school_ratio = models.CharField(max_length=200,blank=True)
  total_students = models.CharField(max_length=200,blank=True)
  

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
  
  def __str__(self):
    return self.annoucement_title

