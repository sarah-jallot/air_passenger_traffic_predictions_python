# Feature extractor
from sklearn.preprocessing import StandardScaler
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
        
        # External data processing
        external_data = pd.read_csv(os.path.join(path,'external_data.csv'))
        external_data.loc[:,"Date"] = pd.to_datetime(external_data.loc[:,"Date"])
        external_dataDeparture = external_data.rename(columns={'Date': 'DateOfDeparture', 'AirPort': 'Departure',  
                                                               'Max TemperatureC': 'd_Max TemperatureC' , 'MeanDew PointC':'d_MeanDew PointC', 
                                                               'Max Humidity':'d_Max Humidity','Max Sea Level PressurehPa':'d_Max Sea Level PressurehPa'
                                                               , 'Max VisibilityKm': 'd_Max VisibilityKm',
                                                               'Mean VisibilityKm':'d_Mean VisibilityKm',
                                                               'Min VisibilitykM':'d_Min VisibilitykM', 'Max Wind SpeedKm/h':'d_Max Wind SpeedKm/h', 'Mean Wind SpeedKm/h':'d_Mean Wind SpeedKm/h',
                                                               'Precipitationmm':'d_Precipitationmm', 'CloudCover':'d_CloudCover', 'WindDirDegrees':'d_WindDirDegrees', 'Rain':'d_Rain',
                                                               'Thunderstorm':'d_Thunderstorm', 'Fog':'d_Fog', 'Snow':'d_Snow', 'Hail':'d_Hail', 'Tornado':'d_Tornado', 'latitude_deg':'d_latitude_deg',
                                                               'longitude_deg':'d_longitude_deg', 'elevation_ft':'d_elevation_ft', '2018':'d_2018', '2017':'d_2017', '2016':'d_2016', '2015':'d_2015'})
        external_dataArrival = external_data.rename(columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival',  
                                                               'Max TemperatureC': 'a_Max TemperatureC' , 'MeanDew PointC':'a_MeanDew PointC', 
                                                               'Max Humidity':'a_Max Humidity','Max Sea Level PressurehPa':'a_Max Sea Level PressurehPa'
                                                               , 'Max VisibilityKm': 'a_Max VisibilityKm',
                                                               'Mean VisibilityKm':'a_Mean VisibilityKm',
                                                             'Min VisibilitykM':'a_Min VisibilitykM', 'Max Wind SpeedKm/h':'a_Max Wind SpeedKm/h', 'Mean Wind SpeedKm/h':'a_Mean Wind SpeedKm/h',
                                                             'Precipitationmm':'a_Precipitationmm', 'CloudCover':'a_CloudCover', 'WindDirDegrees':'a_WindDirDegrees', 'Rain':'a_Rain',
                                                             'Thunderstorm':'a_Thunderstorm', 'Fog':'a_Fog', 'Snow':'a_Snow', 'Hail':'a_Hail', 'Tornado':'a_Tornado', 'latitude_deg':'a_latitude_deg',
                                                             'longitude_deg':'a_longitude_deg', 'elevation_ft':'a_elevation_ft', '2018':'a_2018', '2017':'a_2017', '2016':'a_2016', '2015':'a_2015'})
        
        # Merging them with X_encoded 
        X_encoded = X_df.copy()
        X_encoded.loc[:,'DateOfDeparture'] = pd.to_datetime(X_encoded.loc[:,'DateOfDeparture'])
        X_encoded = pd.merge(X_encoded, external_dataDeparture, how='left',left_on=['DateOfDeparture', 'Departure'],
                             right_on=['DateOfDeparture', 'Departure'],sort=False)
        X_encoded = pd.merge(X_encoded, external_dataArrival, how='left',left_on=['DateOfDeparture', 'Arrival'],
                             right_on=['DateOfDeparture', 'Arrival'],sort=False) 
        
        # Feature engineering
        # Creating columns to distinguish between the two main airports for flights and the rest
        X_encoded['d_ManyFlights'] = 0  
        X_encoded['a_ManyFlights'] = 0
        X_encoded.loc[X_encoded.loc[:,'Departure'] == 'ORD', "d_ManyFlights"] = 1
        X_encoded.loc[X_encoded.loc[:,'Arrival'] == 'ORD', "a_ManyFlights"] = 1
        X_encoded.loc[X_encoded.loc[:,'Departure'] == 'ATL', "d_ManyFlights"] = 1
        X_encoded.loc[X_encoded.loc[:,'Arrival'] == 'ATL', "a_ManyFlights"] = 1

        
        # Categorical encoding of departure and arrival airports
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded.loc[:,'Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded.loc[:,'Arrival'], prefix='a'))
                                   
        # Categorical encoding of the dates 
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
        
        # Scaling our data
        scaler = StandardScaler()
        scaler.fit(X_encoded)
    
        return scaler.transform(X_encoded)