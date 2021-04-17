from django.db import models

# Create your models here.

class hotel(models.Model):
    country = models.CharField(max_length=50)
    city = models.CharField(max_length=50)
    userid = models.CharField(max_length=50)
    hotel = models.CharField(max_length=50)
