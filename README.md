# RecommenderSystem

## Overview
This web application which uses recommender system to predict hotel clusters.

## Quick Start
To get this project up and running locally on your computer:
1. Set up the [Python development environment](https://developer.mozilla.org/en-US/docs/Learn/Server-side/Django/development_environment).
   We recommend using a Python virtual environment. or you can follow instructions given in Instruction.docx file.

2. Assuming you have Python setup, run the following commands (if you're on Windows you may use `py` or `py -3` instead of `python` to start Python):
   ```
   pip install -r requirements.txt
   python manage.py makemigrations
   python manage.py migrate
   python manage.py createsuperuser # Create a superuser
   python manage.py runserver
   ```
3. Open a browser to `http://127.0.0.1:8000/admin/` to open the admin site.

4. Open tab to `http://127.0.0.1:8000` to see the main site, with your new objects.


## Instructions to test the model:
* Install all the dependencies in the Dependencies.txt
* After Installing the dependencies create a directory named "expedia-hotel-recommendations"
* Add the dataset files in the directory just created with the name "expedia-hotel-recommendations".
* Download the Rec_Sys.py file and place it in the project folder containing the directory "expedia-hotel-recommendations".
* Finally open the Command Prompt, and give the following comand "python Rec_Sys.py"

## Description (Rec_API):
1. train_model(): 3 arguments
* X_train: contains all the features the model is trained on
* Y_train: contains the ground truth values
* model_name: string argument representing a model
2. test_model(): 3 arguments
* X_test: contains the features the model is tested on
* X_test: contains the ground truth values
* model_name: string argument representing a model
- returns test score
3. predict_result(): 2 Arguments
* X_test_row: row on which the prediction is made
* model_name: string argument representing a model
- returns result
4. preprocess(): no arguments
* reads and pre_processes the data
* returns X_train, X_test,Y_train,Y_test
