import pandas as pd
import os
from sklearn import preprocessing

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
       
        #Drop non numerical variables keeping all information 
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        

         #Data engineering due to results in the introduction part         
        X_encoded['DateOfDeparture'] = pd.to_datetime(X_encoded['DateOfDeparture'])
        X_encoded['year'] = X_encoded['DateOfDeparture'].dt.year
        X_encoded['month'] = X_encoded['DateOfDeparture'].dt.month
        X_encoded['day'] = X_encoded['DateOfDeparture'].dt.day
        X_encoded['weekday'] = X_encoded['DateOfDeparture'].dt.weekday
        X_encoded['week'] = X_encoded['DateOfDeparture'].dt.week
        X_encoded['n_days'] = X_encoded['DateOfDeparture'].apply(lambda date: (date - pd.to_datetime("1970-01-01")).days)

        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['year'], prefix='y'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['month'], prefix='m'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['day'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['weekday'], prefix='wd'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['week'], prefix='w'))
        
        #Now we can finally drop DateOfDeparture that is not numerical 
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        
        #Scaling our data as it might be useful for some regression methods 
        X_array = X_encoded.values
        X_array = preprocessing.scale(X_array, axis = 0)   
        return X_array