import datetime
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from datetime import datetime

def strip_year(year_val):
    if year_val is not None and type(year_val) is not float:
        try:
            return datetime.strptime(year_val, '%Y-%m-%d').year
        except ValueError:
            return datetime.strptime(year_val, '%Y-%m-%d %H:%M:%S').year
    else:
        return 2013
    pass

def strip_month(month_val):
    if month_val is not None and type(month_val) is not float:
        try:
            return datetime.strptime(month_val, '%Y-%m-%d').month
        except:
            return datetime.strptime(month_val, '%Y-%m-%d %H:%M:%S').month
    else:
        return 1
    pass


print("Reading training data............")
dataset = pd.read_csv('./expedia-hotel-recommendations/train.csv', sep=',').dropna()
print("Training data read.")


print("Reading destinations.............")
destinations = pd.read_csv('./expedia-hotel-recommendations/destinations.csv')
dataset = dataset.sample(frac=0.01, random_state=99)
print("Reading destinations finished.")


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

print("Correlations to hotel_cluster:",dataset.corr()["hotel_cluster"].sort_values())

groups = [dataset.groupby(['srch_destination_id','hotel_country','hotel_market','hotel_cluster'])['is_booking'].agg(['sum','count'])]
aggregate = pd.concat(groups).groupby(level=[0,1,2,3]).sum()
aggregate.dropna(inplace=True)
#print(aggregate.head());

aggregate['sum_and_cnt'] = 0.85*aggregate['sum'] + 0.15*aggregate['count']
aggregate = aggregate.groupby(level=[0,1,2]).apply(lambda x: x.astype(float)/x.sum())
aggregate.reset_index(inplace=True)
#print(aggregate.head());


# Creating pivot table to extract trends
pivot_aggr = aggregate.pivot_table(index=['srch_destination_id','hotel_country','hotel_market'], columns='hotel_cluster', values='sum_and_cnt').reset_index()

dataset = pd.merge(dataset, destinations, how='left', on='srch_destination_id')
dataset = pd.merge(dataset, pivot_aggr, how='left', on=['srch_destination_id','hotel_country','hotel_market'])
dataset.fillna(0, inplace=True)

dataset = dataset.loc[dataset['is_booking'] == 1]

X = dataset.drop(['user_id', 'hotel_cluster', 'is_booking'], axis=1)
y = dataset.hotel_cluster


print("*********************************************************************************************************")
modelSVM = make_pipeline(preprocessing.StandardScaler(), svm.SVC(decision_function_shape='ovo'))
print("SVM score:",np.mean(cross_val_score(modelSVM, X, y, cv=10)))
print("Saving SVM Model...............")
filename = 'modelSVM.sav'
joblib.dump(modelSVM, filename);
print("Model Saved Successfully !")


print("*********************************************************************************************************")
modelSGD = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3))
print("SGD Classifier:",np.mean(cross_val_score(modelSGD, X, y, cv=10)))
print("Saving SGD Model...............")
filename = 'modelSGD.sav'
joblib.dump(modelSGD, filename);
print("Model Saved Successfully !")

print("*********************************************************************************************************")
modelRF = make_pipeline(preprocessing.StandardScaler(), RandomForestClassifier(n_estimators=273,max_depth=10,random_state=0))
print("Random Forest:",np.mean(cross_val_score(modelRF, X, y, cv=10)))
print("Saving Random Forest Model...............")
filename = 'modelRF.sav'
joblib.dump(modelSVM, filename);
print("Model Saved Successfully !")

print("*********************************************************************************************************")
modelKNN = make_pipeline(preprocessing.StandardScaler(), KNeighborsClassifier(n_neighbors=5))
print("KNN:",np.mean(cross_val_score(modelKNN, X, y, cv=10, scoring='accuracy')))
print("Saving K-Nearest Neighbors Model...............")
filename = 'modelKNN.sav'
joblib.dump(modelKNN, filename);
print("Model Saved Successfully !")

print("*********************************************************************************************************")
modelGNB = make_pipeline(preprocessing.StandardScaler(), GaussianNB(priors=None))
print("GaussianNB:",np.mean(cross_val_score(modelGNB, X, y, cv=10)))
print("Saving GaussianNB Model...............")
filename = 'modelGNB.sav'
joblib.dump(modelGNB, filename);
print("Model Saved Successfully !")