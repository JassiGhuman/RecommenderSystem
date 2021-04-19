from django.shortcuts import render
from django.http import HttpResponse, request
from subprocess import run,PIPE
import sys
from django.template import RequestContext
import requests
# Create your views here.
#Each view must return an HttpResponse Object

def index(request):
    return render(request,'HotelReco/index.html')

def predict_result(request,X_test_row = [6707,2,133,0,2,0,8215,951,208,31],model_name = 'XGB'):
        input = request.POST.get('is_package')
        print(input)
        result = run([sys.executable,'xg_b.py'],shell=False,stdout = PIPE)
        return render(request, 'HotelReco/index.html',{'result': result})




#
#