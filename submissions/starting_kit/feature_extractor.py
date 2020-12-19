# Selected feature extractor
import pandas as pd
import os
import math 
import numpy as np
from geopy.distance import geodesic
from sklearn.preprocessing import RobustScaler


def compute_distance(X_encoded):
    return X_encoded.apply(
        lambda x: geodesic(
            (x["d_latitude_deg"],x["d_longitude_deg"]),
            (x["a_latitude_deg"],x["a_longitude_deg"])
        ).km,
        axis=1,
    )
                          
class FeatureExtractor(object):
    def __init__(self):
        pass


    def fit(self, X_df, y_array):
        pass


    def transform(self, X_df):
        X_encoded = X_df.copy()
        path = os.path.dirname(__file__)
        

        external_data = pd.read_csv(os.path.join(path,'external_data.csv'))
        external_data.loc[:,"Date"] = pd.to_datetime(external_data.loc[:,"Date"])
        
        # Building column names for conditions at departure and arrival 
        col_dep = ['d_' + name for name in list(external_data.columns)]
        col_arr = [w.replace('d_', 'a_') for w in col_dep]
        # Fitting the names of the first 2 columns to match our original dataframe 
        col_dep = [w.replace('d_AirPort', 'Departure') for w in col_dep]
        col_dep = [w.replace('d_Date', 'DateOfDeparture') for w in col_dep]
        col_arr = [w.replace('a_AirPort', 'Arrival') for w in col_arr]
        col_arr = [w.replace('a_Date', 'DateOfDeparture') for w in col_arr]
        
        # Building 2 dataframes from dexternal_data to get the information for the departure and arrival airports of each flight
        # Departure airport 
        external_dataDeparture = external_data.copy()
        external_dataDeparture.columns = col_dep
        # Arrival airport
        external_dataArrival = external_data.copy()
        external_dataArrival.columns = col_arr
        

        # Merging them with X_encoded 
        X_encoded.loc[:,'DateOfDeparture'] = pd.to_datetime(X_encoded.loc[:,'DateOfDeparture'])
        X_encoded = pd.merge(X_encoded, external_dataDeparture, how='left',left_on=['DateOfDeparture', 'Departure'],
                             right_on=['DateOfDeparture', 'Departure'],sort=False)
        X_encoded = pd.merge(X_encoded, external_dataArrival, how='left',left_on=['DateOfDeparture', 'Arrival'],
                             right_on=['DateOfDeparture', 'Arrival'],sort=False) 

        ### Feature engineering
        ## Creating columns to distinguish between the two main airports for flights and the rest
        X_encoded['d_ManyFlights'] = 0  
        X_encoded['a_ManyFlights'] = 0
        X_encoded.loc[X_encoded.loc[:,'Departure'] == 'ORD', "d_ManyFlights"] = 1
        X_encoded.loc[X_encoded.loc[:,'Arrival'] == 'ORD', "a_ManyFlights"] = 1
        X_encoded.loc[X_encoded.loc[:,'Departure'] == 'ATL', "d_ManyFlights"] = 1
        X_encoded.loc[X_encoded.loc[:,'Arrival'] == 'ATL', "a_ManyFlights"] = 1

        # Getting the interaction of departure and arrival on the output
        # We inputted the average output by departure/arrival airport in external data
        X_encoded["logPAX_interaction"] = X_encoded.loc[:,"d_departure_avglogPAX"]*X_encoded.loc[:,"a_arrival_avglogPAX"]
        X_encoded["DistanceToHoliday_interaction"] = X_encoded.loc[:,"d_DistanceToClosestHoliday"]*X_encoded.loc[:,"a_DistanceToClosestHoliday"]
        X_encoded["GDPPerCapita_interaction"] = X_encoded.loc[:,"d_GDPPerCapita_2017"]*X_encoded.loc[:,"a_GDPPerCapita_2017"]
        X_encoded["Region_interaction"] = X_encoded.loc[:,"d_M"]*X_encoded.loc[:,"a_M"] \
        + X_encoded.loc[:,"d_S"]*X_encoded.loc[:,"a_S"] + \
        X_encoded.loc[:,"d_N"]*X_encoded.loc[:,"a_N"] +\
        X_encoded.loc[:,"d_W"]*X_encoded.loc[:,"a_W"]
        X_encoded["passenger_interaction"] = X_encoded["d_passengers_2015"]*X_encoded["a_passengers_2015"] 
        X_encoded["MaxTemperature_interaction"] = X_encoded.loc[:,"d_Max TemperatureC"]*X_encoded.loc[:,"a_Max TemperatureC"]


        # Distance
        X_encoded["Distance"] = compute_distance(X_encoded)

        
        ## Categorical encoding of departure and arrival airports
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded.loc[:,'Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded.loc[:,'Arrival'], prefix='a'))
                                   
        ## Categorical encoding of the dates 
        X_encoded['year'] = X_encoded.loc[:,'DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded.loc[:,'DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded.loc[:,'DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded.loc[:,'DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded.loc[:,'DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded.loc[:,'DateOfDeparture'].apply(lambda date: 
                                                                         (date - pd.to_datetime("1970-01-01")).days)
        
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
    
        # Finally getting rid of departure, arrival, and date columns now that we do not need them to merge
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        X_encoded = X_encoded.drop('DateOfDeparture',axis = 1)

    
        return X_encoded