# Feature extractor
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
        # Feature engineering
        # Creating columns to distinguish between the two main airports for flights and the rest
        X_encoded['d_ManyFlights'] = 0
        X_encoded['a_ManyFlights'] = 0
        X_encoded['d_ManyFlights'][X_df['Departure'] == 'ORD'] = 1
        X_encoded['d_ManyFlights'][X_df['Departure'] == 'ATL'] = 1
        X_encoded['a_ManyFlights'][X_df['Arrival'] == 'ORD'] = 1
        X_encoded['a_ManyFlights'][X_df['Arrival'] == 'ATL'] = 1

        # Categorical encoding of departure and arrival airports
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Departure'], prefix='d'))
        X_encoded = X_encoded.join(pd.get_dummies(X_encoded['Arrival'], prefix='a'))

        # Categorical encoding of the dates 
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



        # Auxiliary dataset we will use to complete our data
        external_dataTest= pd.read_csv(os.path.join(path, 'external_data.csv'))
        external_data = external_dataTest['Date', 'AirPort', 'Max TemperatureC', 'Mean TemperatureC',
                                          'Min TemperatureC', 'Dew PointC', 'MeanDew PointC', 'Min DewpointC',
                                          'Max Humidity', 'Mean Humidity', 'Min Humidity',
                                          'Max Sea Level PressurehPa', 'Mean Sea Level PressurehPa',
                                          'Min Sea Level PressurehPa', 'Max VisibilityKm', 'Mean VisibilityKm',
                                          'Min VisibilitykM', 'Max Wind SpeedKm/h', 'Mean Wind SpeedKm/h',
                                          'Max Gust SpeedKm/h', 'Precipitationmm', 'CloudCover', 'Events',
                                          'WindDirDegrees', 'Rain', 'Thunderstorm', 'Fog', 'Snow', 'Hail',
                                          'Tornado', 'a_latitude_deg', 'a_longitude_deg', 'a_elevation_ft',
                                          '2018', '2017', '2016', '2015']

        # Now merging external data with out dataset 
        # Creating two tables, one for departures, the other for arrivals
        external_dataDeparture = external_data.rename(columns={'Date': 'DateOfDeparture', 'AirPort': 'Departure'})
        external_dataArrival = external_data.rename(columns={'Date': 'DateOfDeparture', 'AirPort': 'Arrival'})

        # Merging them with X_encoded   
        X_encoded = pd.merge(X_encoded, external_dataDeparture, how='left',left_on=['DateOfDeparture', 'Departure'],
                             right_on=['DateOfDeparture', 'Departure'],sort=False)
        X_encoded = pd.merge(X_encoded, external_dataArrival, how='left',left_on=['DateOfDeparture', 'Arrival'],
                             right_on=['DateOfDeparture', 'Arrival'],sort=False) 

        # Finally getting rid of departure, arrival, and date columns now that we do not need them to merge
        X_encoded = X_encoded.drop('Departure', axis=1)
        X_encoded = X_encoded.drop('Arrival', axis=1)
        X_encoded = X_encoded.drop('DateOfDeparture', axis=1)
        X_array = X_encoded.values