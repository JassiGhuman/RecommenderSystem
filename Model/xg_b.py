import numpy as np
import pandas as pd 
import xgboost as xgb
import sys
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
import joblib

def read_csv(no_of_rows,is_random):
    dataset = "./expedia-hotel-recommendations/train.csv"
    if is_random == True:
        dataFrame = pd.read_csv(dataset,nrows = no_of_rows, random_state = 100)
    else:
        dataFrame = pd.read_csv(dataset, nrows=no_of_rows)
    return dataFrame


def process_data(dataFrame):
    feature_selection = ['site_name', 'user_location_region', 'is_package', 'srch_adults_cnt', 'srch_children_cnt','srch_destination_id', 'hotel_market', 'hotel_country', 'hotel_cluster']
    processed_data = pd.DataFrame(columns=feature_selection)
    processed_data = pd.concat([processed_data, dataFrame[dataFrame['is_booking'] == 1][feature_selection]])
    for column in processed_data:
        processed_data[column] = processed_data[column].astype(str).astype(int);
    X = processed_data
    Y = processed_data['hotel_cluster'].values
    print(processed_data.head());
    return X, Y;


def create_tt_split(X,Y,split_ratio = 0.2):
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=split_ratio)
    return X_train, X_test, Y_train, Y_test;

def trian_xgboost(X_train, Y_train, max_dep = 5, learn_rate = 0.01):
    xgb_classifier = xgb.XGBClassifier(objective = 'multi:softmax',max_depth = max_dep,n_estimators=300,learning_rate=learn_rate,nthread=4,subsample=0.7,colsample_bytree=0.7,min_child_weight = 3,silent=False)
    modelXGB = xgb_classifier.fit(X_train, Y_train)
    print(modelXGB.score(X_train,Y_train));
    filename = 'model_XGB.sav'
    joblib.dump(modelXGB,filename)
    print("model_XGB saved")



#Function that takes in 3 Arguments : X_test (contains all the features the model is tested on.) , Y_test (contains the ground truth values) and model_name (string argument representing a model)
def evaluate_model(X_test, Y_test, model_name = 'XGB'):
    X_test_transformed = X_test
    score = 0;
    if model_name == 'SVM':
        filename = 'model_SVM.sav'
        loaded_model = joblib.load(filename)
        print("model_SVM Loaded")
        score=loaded_model.score(X_test_transformed,Y_test)
        print(score)
    elif model_name == 'SGD':
        filename = 'model_SGD.sav'
        loaded_model = joblib.load(filename)
        print("model_SGD Loaded")
        score=loaded_model.score(X_test_transformed,Y_test)
        print(score)
    elif model_name == 'RF':
        filename = 'model_RF.sav'
        loaded_model = joblib.load(filename)
        print("model_RF Loaded")
        score=loaded_model.score(X_test_transformed,Y_test)
        print(score)
    elif model_name == 'KNN':
        filename = 'model_KNN.sav'
        loaded_model = joblib.load(filename)
        print("mode_KNN Loaded")
        score=loaded_model.score(X_test_transformed,Y_test)
        print(score)
    elif model_name == 'GNB':
        filename = 'model_GNB.sav'
        loaded_model = joblib.load(filename)
        print("model_GNB Loaded")
        score=loaded_model.score(X_test_transformed,Y_test)
    elif model_name == 'XGB':
        filename = 'model_XGB.sav'
        loaded_model = joblib.load(filename)
        print("model_XGB Loaded")
        score=loaded_model.score(X_test_transformed,Y_test)
        print(score)
    else:
        print("Provide the argument model_name from one of the supported models.")
    return score;


def predict_result(X_test_row,model_name = 'XGB'):
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
    elif model_name == 'XGB':
        filename = 'model_XGB.sav'
        loaded_model = joblib.load(filename)
        print("model_XGB Loaded")
        result = loaded_model.predict(X_test_row)
        print("Prediction:")
        print(result)
    else:
        print("Provide the argument model_name from one of the supported models.")
    return result;

if __name__ == "__main__":
    p_data = read_csv(10000,False);
    X,Y = process_data(p_data);
    X_train,X_test,Y_train,Y_test = create_tt_split(X,Y)
    X_test_row = X_train.iloc[[1]]
    print(X_test_row);
    actual_hotel = Y_train[1];
    print("**********************************training**************************************************")
    trian_xgboost(X_train,Y_train)
    print("*************************************evaluating***********************************************")
    evaluate_model(X_test,Y_test);
    print("***************************************predicting*********************************************")
    result = predict_result(X_test_row)
    print("****************************************results********************************************")
    print("Result:", result)
    print("Actual:",actual_hotel)


