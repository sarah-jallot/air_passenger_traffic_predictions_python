# Combined regressor 
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge 
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLars
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import RandomForestRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = RandomForestRegressor(n_estimators=20)
        self.reg2 = Lasso()
        self.reg3 = LassoLars()
        self.reg4 = Ridge()
        self.reg5 = LinearRegression()
        self.reg6 = BayesianRidge()
        self.metareg = RandomForestRegressor(n_estimators=20)

    def fit(self, X, y):
        self.reg.fit(X, y)
        self.reg2.fit(X, y)
        self.reg3.fit(X,y)
        self.reg4.fit(X,y)
        self.reg5.fit(X,y)
        self.reg6.fit(X,y)
        X_combined = np.vstack([self.reg.predict(X), self.reg2.predict(X),self.reg3.predict(X),self.reg4.predict(X),self.reg5.predict(X)]).T
        self.metareg.fit(X_combined, y)


    def predict(self, X):
        pred1 = self.reg.predict(X)
        pred2 = self.reg2.predict(X)
        pred3 = self.reg3.predict(X)
        pred4 = self.reg4.predict(X)
        pred5 = self.reg5.predict(X)
        pred6 = self.reg6.predict(X)
        X_combined = np.vstack([pred1, pred2, pred3, pred4, pred5]).T
        return self.metareg.predict(X_combined)