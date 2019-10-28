from sklearn.base import BaseEstimator
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = Ridge()
        self.reg2 = Lasso()
        self.metareg = RandomForestRegressor()

    def fit(self, X, y):
        self.reg.fit(X, y)
        self.reg2.fit(X, y)
        X_combined = np.vstack([self.reg.predict(X), self.reg2.predict(X)]).T
        self.metareg.fit(X_combined, y)


    def predict(self, X):
        pred1 = self.reg.predict(X)
        pred2 = self.reg2.predict(X)
        X_combined = np.vstack([pred1, pred2]).T
        return self.metareg.predict(X_combined)