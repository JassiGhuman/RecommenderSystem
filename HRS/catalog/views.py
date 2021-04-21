from django.shortcuts import render
from .models import RecordTable
from subprocess import run, PIPE
import sys
import datetime

# view to render UI
today = datetime.date.today()


def index(request):
    # Render the HTML template index.html with the data in the context variable.
    return render(request, 'index.html')
    
    
def contact(request):
    # Render the HTML template contact.html with the data in the context variable.
    return render(request, 'contact.html')


def about(request):
    # Render the HTML template about.html with the data in the context variable.
    return render(request, 'about.html')


def popdata(request):
    # To populate the parameters for the user to provide input for retrieving the accuracy of the model
    data1 = RecordTable.objects.order_by('is_package').values('is_package').distinct()
    data2 = RecordTable.objects.order_by('user_location_region').values('user_location_region').distinct()
    data3 = RecordTable.objects.order_by('site_name').values('site_name').distinct()
    data4 = RecordTable.objects.order_by('srch_adults_cnt').values('srch_adults_cnt').distinct()
    data5 = RecordTable.objects.order_by('srch_children_cnt').values('srch_children_cnt').distinct()
    data6 = RecordTable.objects.order_by('srch_destination_id').values('srch_destination_id').distinct()
    data7 = RecordTable.objects.order_by('hotel_market').values('hotel_market').distinct()
    data8 = RecordTable.objects.order_by('hotel_country').values('hotel_country').distinct()
    data9 = RecordTable.objects.order_by('hotel_cluster').values('hotel_cluster').distinct()
    context = {
        'data1': data1,
        'data2': data2,
        'data3': data3,
        'data4': data4,
        'data5': data5,
        'data6': data6,
        'data7': data7,
        'data8': data8,
        'data9': data9,
    }
    return render(request, 'hotels.html', context)


def testdata(request):
    # retrieving data for charts and graphs
    data1 = RecordTable.objects.order_by('user_location_region').values('user_location_region').distinct()
    data2 = RecordTable.objects.all().order_by('user_location_region')
    context = {
        'data1': data1,
        'data2': data2,
    }
    return render(request, 'visualize.html', context)


def traindata(request):
    if request.method == 'POST':
        site = request.POST.get('site')
        location = request.POST.get('userloc')
        package = request.POST.get('package')
        adult = request.POST.get('adultcount')
        children = request.POST.get('childcount')
        dest = request.POST.get('destination')
        market = request.POST.get('market')
        country = request.POST.get('country')
        cluster = request.POST.get('cluster')
        userdata = [site, location, package, adult, children, dest, market, country, cluster]
        str_userdata = str(userdata)
        result = run([sys.executable, 'RecSys.py',str_userdata], shell=False, stdout=PIPE)
        return render(request, 'hotels.html', {'result': result})
    else:
        return render(request, 'hotels.html')


def predict_result(request, X_test_row=[6707, 2, 133, 0, 2, 0, 8215, 951, 208, 31], model_name='XGB'):
    input = request.POST.get('is_package')
    print('Inside method')
    print(input)
    result = run([sys.executable, 'RecSys.py'], shell=False, stdout=PIPE)
    return render(request, 'hotels.html', {'result': result})



