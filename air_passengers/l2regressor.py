from sklearn.linear_model import Ridge 
from sklearn import BaseEstimator


class Regressor(BaseEstimator):
    def __init__(self):
        self.reg = ()

    def fit(self, X, y):
        self.reg.fit(X, y)

    def predict(self, X):
        return self.reg.predict(X)