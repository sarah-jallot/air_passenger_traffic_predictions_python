# Gradient boosting regressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.base import BaseEstimator
class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = GradientBoostingRegressor(learning_rate = 0.02, n_estimators = 400)

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)

