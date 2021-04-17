from django.shortcuts import render
from.models import hotel

# Create your views here.

def index(request):
    """View function for home page of site. Pass data to page"""
    context = {
    'num_books': 4,
    'num_instances': 5,
    'num_instances_available': 3,
    'num_authors': 6,
    }

    # Render the HTML template index.html with the data in the context variable.
    return render(request, 'index.html', context)
    
    
def contact(request):
    """View function for home page of site. Pass data to page"""
    context = {
    'num_books': 4,
    'num_instances': 5,
    'num_instances_available': 3,
    'num_authors': 6,
    }

    # Render the HTML template index.html with the data in the context variable.
    return render(request, 'contact.html', context)
    
def about(request):
    """View function for home page of site. Pass data to page"""
    context = {
    'num_books': 4,
    'num_instances': 5,
    'num_instances_available': 3,
    'num_authors': 6,
    }

    # Render the HTML template index.html with the data in the context variable.
    return render(request, 'about.html', context)

def hotels(request):
    if(request.method=='POST'):
        country = request.POST.get('country')
        city = request.POST.get('city')
        userid = request.POST.get('user')
        hotel_r = request.POST.get('hotel')
        print(country)
        print(city)
        print(userid)
        print(hotel_r)
        return render(request, 'hotels.html')
    else:
        return render(request, 'hotels.html')
    # Render the HTML template index.html with the data in the context variable.
