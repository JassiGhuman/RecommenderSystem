
from django.conf.urls import include, url
from HotelReco import views
from django.urls import path
from . import views

urlpatterns = [
    url('^predict$',views.predict_result,name='predict'),
    path('', views.index, name='index'),
]