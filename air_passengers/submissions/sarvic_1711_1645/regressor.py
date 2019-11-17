from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = Ridge()
        self.reg2 = Lasso()
        self.reg3 = RandomForestRegressor()
        self.reg4 = LinearRegression()
        self.metareg = RandomForestRegressor()

    def fit(self, X, y):
        self.reg.fit(X, y)
        self.reg2.fit(X, y)
        self.reg3.fit(X, y)
        self.reg4.fit(X, y)
        X_combined = np.vstack([self.reg.predict(X), self.reg2.predict(X), self.reg3.predict(X), self.reg4.predict(X)]).T
        self.metareg.fit(X_combined, y)


    def predict(self, X):
        pred1 = self.reg.predict(X)
        pred2 = self.reg2.predict(X)
        pred3 = self.reg3.predict(X)
        pred4 = self.reg3.predict(X)
        X_combined = np.vstack([pred1, pred2, pred3, pred4]).T
        return self.metareg.predict(X_combined)