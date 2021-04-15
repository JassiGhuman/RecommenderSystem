import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import AgglomerativeClustering
import os
"""
dirname = './expedia-hotel-recommendations'
for filename in dirname:
        print(os.path.join(dirname, filename))
"""
NUMBER_OF_ROWS = 1000 # Train data is too big, get some rows
train_df = pd.read_csv('./expedia-hotel-recommendations/train.csv', nrows=NUMBER_OF_ROWS)

# Aggregation data
groupby1 = train_df.groupby(['srch_destination_id', 'hotel_cluster'])['is_booking'].agg(['count'])
groupby1.head()

#print(groupby1.head())

single_index_df = groupby1.reset_index(level=[0,1])
#print(single_index_df)

def list_2_str(items):
    if (items is None) or (len(items) <= 0):
        return ''
    result = ''
    for item in items:
        result = result + str(item) + ','
    return result[:(len(result) - 1)]


total_count_of_hotel_cluster = 0
destination_id_n_cluster_list = dict()
for index, row in single_index_df.iterrows():
    srch_destination_id = row['srch_destination_id']
    hotel_cluster = row['hotel_cluster']

    hotel_clusters = list()
    if srch_destination_id in destination_id_n_cluster_list:
        hotel_clusters = destination_id_n_cluster_list[srch_destination_id]
    hotel_clusters.append(hotel_cluster)
    total_count_of_hotel_cluster += 1
    destination_id_n_cluster_list[srch_destination_id] = hotel_clusters

destination_id_n_clusters = dict()
for key, value in destination_id_n_cluster_list.items():
    str_value = list_2_str(value)
    destination_id_n_clusters[key] = str_value

final_df = pd.DataFrame(destination_id_n_clusters.items(), columns=['srch_destination_id', 'hotel_clusters'])
final_df.head()

# print(final_df.head())

#Train Manipulation
NUMBER_OF_ROWS = 5000 # Train data is too big, get some rows
test_df = pd.read_csv('./expedia-hotel-recommendations/train.csv', nrows=NUMBER_OF_ROWS, usecols=['srch_destination_id'])
test_df.head()

#print(test_df.head())

merged_test_df = test_df.merge(final_df, how = 'left')
merged_test_df[['hotel_clusters']] = merged_test_df[['hotel_clusters']].fillna(value = 'NA')
merged_test_df.head(10)

#print(merged_test_df.head(10))


plt.scatter(merged_test_df['srch_destination_id'].values, merged_test_df['hotel_clusters'].values)
plt.show()
"""
#K-NN ALG - Train
NUMBER_OF_ROWS = 5000 # Train data is too big, get some rows
train_df = pd.read_csv('./expedia-hotel-recommendations/train.csv', nrows=NUMBER_OF_ROWS)
test_df = pd.read_csv('./expedia-hotel-recommendations/test.csv', nrows=NUMBER_OF_ROWS)
k_nearest_train_points = train_df[['srch_destination_id']]
k_nearest_train_labels = train_df[['hotel_cluster']]
k_nearest_test_points = test_df[['srch_destination_id']]

k_nearest_classifier = KNeighborsClassifier(n_neighbors = 100)
k_nearest_classifier.fit(k_nearest_train_points, k_nearest_train_labels)
k_nearest_result = k_nearest_classifier.predict(k_nearest_test_points)

plt.scatter(k_nearest_test_points, k_nearest_result, s=50, alpha=0.5)
plt.show()
"""
# Agglomerative clustering Train

NUMBER_OF_ROWS = 5000 # Train data is too big, get some rows
train_df = pd.read_csv('./expedia-hotel-recommendations/train.csv', nrows=NUMBER_OF_ROWS)
test_df = pd.read_csv('./expedia-hotel-recommendations/test.csv', nrows=NUMBER_OF_ROWS)

agg_cluster_X = train_df[['srch_destination_id']]
agg_cluster_Y = train_df[['hotel_cluster']]
agg_cluster_destination_id = test_df[['srch_destination_id']]

agglomerativeCluster = AgglomerativeClustering(n_clusters = 100, linkage = 'ward')
agglomerativeCluster.fit(agg_cluster_X, agg_cluster_Y)
agg_recommend_hotel = agglomerativeCluster.fit_predict(agg_cluster_destination_id)

plt.scatter(agg_cluster_destination_id, agg_recommend_hotel, s = 50, alpha=0.5)
plt.show()


