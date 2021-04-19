

import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'HRS.settings')
import django
django.setup()

import pandas as pd
from HotelReco.models import RecordTable


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

def populate(data,N=10000):
    site_name = data['site_name']
    user_location_region = data['user_location_region']
    is_package = data['is_package']
    srch_adults_cnt = data['srch_adults_cnt']
    srch_children_cnt = data['srch_children_cnt']
    srch_destination_id = data['srch_destination_id']
    hotel_market = data['hotel_market']
    hotel_country = data['hotel_country']
    hotel_cluster = data['hotel_cluster']
    for entry in range(N):
        user_search_destination = RecordTable.objects.get_or_create(site_name=site_name.iloc[[entry]],user_location_region = user_location_region.iloc[[entry]], is_package = is_package.iloc[[entry]], srch_adults_cnt = srch_adults_cnt.iloc[[entry]], srch_children_cnt = srch_children_cnt.iloc[[entry]], srch_destination_id = srch_destination_id.iloc[[entry]], hotel_market = hotel_market.iloc[[entry]], hotel_country = hotel_country.iloc[[entry]], hotel_cluster = hotel_cluster.iloc[[entry]])
        print(entry)
    print("Done Populating data!!")

def main():
    dataset = read_csv(10000,False)
    X,Y = process_data(dataset);
    populate(X,10000)

main()
