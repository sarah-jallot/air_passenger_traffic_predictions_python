# Gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator
class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = GradientBoostingRegressor(n_estimators = 200)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

