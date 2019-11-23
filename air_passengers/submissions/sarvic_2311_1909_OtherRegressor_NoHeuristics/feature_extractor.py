# Feature extractor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import os
import math 
import numpy as np

class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        path = os.path.dirname(__file__)
        
        ## External data processing
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
        
        # Building 2 dataframes from data_add to get the information for the departure and arrival airports of each flight
        # Departure airport 
        external_dataDeparture = external_data.copy()
        external_dataDeparture.columns = col_dep
        # Arrival airport
        external_dataArrival = external_data.copy()
        external_dataArrival.columns = col_arr
        
        # Merging them with X_encoded 
        X_encoded = X_df.copy()
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
        
        ## Creating the distance variable
        # Creating latitude and longitude difference for the purpose of computing distance. 
        radius = 6371
        X_encoded["latitude_difference"] = (X_encoded["d_latitude_deg"] - X_encoded["a_latitude_deg"])
        X_encoded["longitude_difference"] = (X_encoded["d_longitude_deg"] - X_encoded["a_longitude_deg"])
        # Creating my distance column
        n,p = X_encoded.shape
        X_encoded["Distance"] = 0
        # Getting the indexes I will need 
        j = X_encoded.columns.get_loc("d_latitude_deg")
        l = X_encoded.columns.get_loc("a_latitude_deg")
        o = X_encoded.columns.get_loc("d_longitude_deg")
        t = X_encoded.columns.get_loc("a_longitude_deg")
        u = X_encoded.columns.get_loc("latitude_difference")
        v = X_encoded.columns.get_loc("longitude_difference")

        for i in np.arange(n):
            lat1 = X_encoded.iloc[i,j]
            lat2 = X_encoded.iloc[i,l]
            lon1 = X_encoded.iloc[i,o]
            lon2 = X_encoded.iloc[i,t]
            dlat = math.radians(X_encoded.iloc[i,u])
            dlon = math.radians(X_encoded.iloc[i,v])
            a = math.sin(dlat/2) * math.sin(dlat/2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon/2) * math.sin(dlon/2)
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            d = radius * c
            X_encoded.iloc[i,-1] = d
        # Now dropping latitude and longitude difference
        X_encoded = X_encoded.drop(columns={"latitude_difference", "longitude_difference"})

        
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