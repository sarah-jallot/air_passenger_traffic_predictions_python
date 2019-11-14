import pandas as pd
import os


class FeatureExtractor(object):
    def __init__(self):
        pass

    def fit(self, X_df, y_array):
        pass

    def transform(self, X_df):
        X_encoded = X_df
        path = os.path.dirname(__file__)
        external_data = pd.read_csv(os.path.join(path, 'external_data.csv'))
        
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
        data_departure = external_data.copy()
        data_departure.columns = col_dep
        # Arrival airport
        data_arrival = external_data.copy()
        data_arrival.columns = col_arr
        
        # Concatenating original data and data at departure 
        # Data at departure
        X_encoded = pd.merge(X_df, data_departure, how='left', left_on=['Departure', 'DateOfDeparture'], right_on=['Departure', 'DateOfDeparture'])
        # Data at arrival
        X_encoded = pd.merge(X_encoded, data_arrival, how='left', left_on=['Arrival', 'DateOfDeparture'], right_on=['Arrival', 'DateOfDeparture'])
       
        #Drop non numerical variables as already written differently
        
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
    
        X_array = X_encoded.values
        return X_array