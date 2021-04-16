from django.shortcuts import render

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