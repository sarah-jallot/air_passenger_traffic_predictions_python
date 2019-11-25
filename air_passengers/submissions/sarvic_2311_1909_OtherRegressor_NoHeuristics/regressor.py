# Basic combined regressor 
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = RandomForestRegressor(n_estimators=205)
        self.reg2 = GradientBoostingRegressor(learning_rate = 0.2, n_estimators = 300)
        self.metareg = RandomForestRegressor(n_estimators=205)

    def fit(self, X, y):
        self.reg.fit(X,y)
        self.reg2.fit(X,y)
        X_combined = np.vstack([self.reg.predict(X), self.reg2.predict(X)]).T
        self.metareg.fit(X_combined, y)


    def predict(self, X):
        pred1 = self.reg.predict(X)
        pred2 = self.reg2.predict(X)
        X_combined = np.vstack([pred1, pred2]).T
        return self.metareg.predict(X_combined)