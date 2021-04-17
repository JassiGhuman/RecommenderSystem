import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import joblib


#function to get the stripped year datetime object
def strip_year(year_val):
    if year_val is not None and type(year_val) is not float:
        try:
            return datetime.strptime(year_val, '%Y-%m-%d').year
        except ValueError:
            return datetime.strptime(year_val, '%Y-%m-%d %H:%M:%S').year
    else:
        return 2013
    pass
#function to get the stripped month datetime object
def strip_month(month_val):
    if month_val is not None and type(month_val) is not float:
        try:
            return datetime.strptime(month_val, '%Y-%m-%d').month
        except:
            return datetime.strptime(month_val, '%Y-%m-%d %H:%M:%S').month
    else:
        return 1
    pass


#Function that pre-processes the dataset and get the train-test splits
def preprocess_dataset():
    train_rows = 50000
    print("Reading dataset................................")
    dataset = pd.read_csv('E:/802/Final Project/expedia-hotel-recommendations/train.csv', sep=',',nrows=train_rows).dropna()
    print("Dataset read.")

    print("Reading destinations.............")
    destinations = pd.read_csv('E:/802/Final Project/expedia-hotel-recommendations/destinations.csv') 
    print("Destinations read.")

    dataset['date_time_year'] = pd.Series(dataset.date_time, index = dataset.index)
    dataset['date_time_month'] = pd.Series(dataset.date_time, index = dataset.index)

    dataset.date_time_year = dataset.date_time_year.apply(lambda year_val: strip_year(year_val))
    dataset.date_time_month = dataset.date_time_month.apply(lambda month_val: strip_month(month_val))
    del dataset['date_time']

    dataset['srch_ci_year'] = pd.Series(dataset.srch_ci, index=dataset.index)
    dataset['srch_ci_month'] = pd.Series(dataset.srch_ci, index=dataset.index)

    dataset.srch_ci_year = dataset.srch_ci_year.apply(lambda year_val: strip_year(year_val))
    dataset.srch_ci_month = dataset.srch_ci_month.apply(lambda month_val: strip_month(month_val))
    del dataset['srch_ci']

    dataset['srch_co_year'] = pd.Series(dataset.srch_co, index=dataset.index)
    dataset['srch_co_month'] = pd.Series(dataset.srch_co, index=dataset.index)

    dataset.srch_co_year = dataset.srch_co_year.apply(lambda year_val: strip_year(year_val))
    dataset.srch_co_month = dataset.srch_co_month.apply(lambda month_val: strip_month(month_val))
    del dataset['srch_co']

    groups = [dataset.groupby(['srch_destination_id','hotel_country','hotel_market','hotel_cluster'])['is_booking'].agg(['sum','count'])]
    aggregate = pd.concat(groups).groupby(level=[0,1,2,3]).sum()
    aggregate.dropna(inplace=True)

    aggregate['sum_and_cnt'] = 0.85*aggregate['sum'] + 0.15*aggregate['count']
    aggregate = aggregate.groupby(level=[0,1,2]).apply(lambda x: x.astype(float)/x.sum())
    aggregate.reset_index(inplace=True)

    pivot_aggr = aggregate.pivot_table(index=['srch_destination_id','hotel_country','hotel_market'], columns='hotel_cluster', values='sum_and_cnt').reset_index()

    dataset = pd.merge(dataset, destinations, how='left', on='srch_destination_id')
    dataset = pd.merge(dataset, pivot_aggr, how='left', on=['srch_destination_id','hotel_country','hotel_market'])
    dataset.fillna(0, inplace=True)

    X = dataset.drop(['user_id', 'hotel_cluster', 'is_booking'], axis=1)
    Y = dataset.hotel_cluster
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.10)
    print("Data Pre-processed!!")
    return X_train, X_test, Y_train, Y_test

#Function that takes in 3 arguments: X_train (contains all the features the model is trained on.), Y_train (contains the ground truth values) and model_name (string argument representing a model)
def train_model(X_train, Y_train, model_name):
    scaler = StandardScaler().fit(X_train)
    X_train_transformed = scaler.transform(X_train)
    if model_name == 'SVM':
        modelSVM = svm.SVC(decision_function_shape='ovo').fit(X_train_transformed, Y_train)
        filename = 'model_SVM.sav'
        joblib.dump(modelSVM,filename)
        print("model_SVM saved")
    elif model_name == 'SGD':
        modelSGD = SGDClassifier(max_iter=1000, tol=1e-3).fit(X_train_transformed, Y_train)
        filename = 'model_SGD.sav'
        joblib.dump(modelSGD,filename)
        print("model_SGD saved")
    elif model_name == 'RF':
        modelRF = RandomForestClassifier(n_estimators=273,max_depth=10,random_state=0).fit(X_train_transformed,Y_train)
        filename = 'model_RF.sav'
        joblib.dump(modelRF,filename)
        print("model_RF saved")
    elif model_name == 'KNN':
        modelRF = KNeighborsClassifier(n_neighbors=5).fit(X_train_transformed,Y_train)
        filename = 'model_KNN.sav'
        joblib.dump(modelRF,filename)
        print("model_KNN saved")
    elif model_name == 'GNB':
        modelGNB = GaussianNB(priors=None).fit(X_train_transformed,Y_train)
        filename = 'model_GNB.sav'
        joblib.dump(modelGNB,filename)
        print("model_GNB saved")
    else:
        print("Provide the argument model_name from one of the supported models.")

#Function that takes in 3 Arguments : X_test (contains all the features the model is trained on.) , Y_test (contains the ground truth values) and model_name (string argument representing a model)
def evaluate_model(X_test, Y_test, model_name):
    scaler = preprocessing.StandardScaler().fit(X_test)
    X_test_transformed = scaler.transform(X_test)
    if model_name == 'SVM':
        filename = 'model_SVM.sav'
        loaded_model = joblib.load(filename)
        print("model_SVM Loaded")
        print(loaded_model.score(X_test_transformed,Y_test))
    elif model_name == 'SGD':
        filename = 'model_SGD.sav'
        loaded_model = joblib.load(filename)
        print("model_SGD Loaded")
        print(loaded_model.score(X_test_transformed,Y_test))
    elif model_name == 'RF':
        filename = 'model_RF.sav'
        loaded_model = joblib.load(filename)
        print("model_RF Loaded")
        print(loaded_model.score(X_test_transformed,Y_test))
    elif model_name == 'KNN':
        filename = 'model_KNN.sav'
        loaded_model = joblib.load(filename)
        print("mode_KNN Loaded")
        print(loaded_model.score(X_test_transformed,Y_test))
    elif model_name == 'GNB':
        filename = 'model_GNB.sav'
        loaded_model = joblib.load(filename)
        print("model_GNB Loaded")
        print(loaded_model.score(X_test_transformed,Y_test))
    else:
        print("Provide the argument model_name from one of the supported models.")

#Function that takes in 2 Arguments : X-test_row (row on which the prediction is made), model_name (string argument repersenting a model) 
def predict_result(X_test_row,model_name):
    result = 0;
    if model_name == 'SVM':
        filename = 'model_SVM.sav'
        loaded_model = joblib.load(filename)
        print("model_SVM Loaded")
        result = loaded_model.predict(X_test_row)
        print("Prediction:")
        print(result)
    elif model_name == 'SGD':
        filename = 'model_SGD.sav'
        loaded_model = joblib.load(filename)
        print("model_SGD Loaded")
        result = loaded_model.predict(X_test_row)
        print("Prediction:")
        print(result)
    elif model_name == 'RF':
        filename = 'model_RF.sav'
        loaded_model = joblib.load(filename)
        print("model_RF Loaded")
        result = loaded_model.predict(X_test_row)
        print("Prediction:")
        print(result)
    elif model_name == 'KNN':
        filename = 'model_KNN.sav'
        loaded_model = joblib.load(filename)
        print("mode_KNN Loaded")
        result = loaded_model.predict(X_test_row)
        print("Prediction:")
        print(result)
    elif model_name == 'GNB':
        filename = 'model_GNB.sav'
        loaded_model = joblib.load(filename)
        print("model_GNB Loaded")
        result = loaded_model.predict(X_test_row)
        print("Prediction:")
        print(result)
    else:
        print("Provide the argument model_name from one of the supported models.")
    return result;


#Controller function that makes the calls and get the result
def main():
    X_train, X_test, Y_train, Y_test = preprocess_dataset()
    models = ['SVM','SGD','RF','KNN','GNB']
    train_model(X_train, Y_train,models[0])
    evaluate_model(X_test,Y_test,models[0])

    train_model(X_train, Y_train,models[1])
    evaluate_model(X_test,Y_test,models[1])

    train_model(X_train, Y_train,models[2])
    evaluate_model(X_test,Y_test,models[2])

    train_model(X_train, Y_train,models[3])
    evaluate_model(X_test,Y_test,models[3])

    train_model(X_train, Y_train,models[4])
    evaluate_model(X_test,Y_test,models[4])
    predict_result(X_test.iloc[[1]],'SVM');


main()







