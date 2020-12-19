# Basic combined regressor 
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = HistGradientBoostingRegressor(max_iter= 500, min_samples_leaf = 30)
      

    def fit(self, X, y):
        self.reg.fit(X,y)
     


    def predict(self, X):
        pred1 = self.reg.predict(X)
     
        return pred1