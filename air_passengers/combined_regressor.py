from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor

class Regressor():
    def __init__(self):
        self.reg = Ridge()
        self.reg2 = Lasso()
        self.metareg = RandomForestRegressor()

    def fit(self, X, y):
        self.reg.fit(X, y)
        self.reg2.fit(X, y)
        
        X_combined = np.hstack([reg1.predict(X), reg2.predict(X)])
        self.metareg.fit(X_combined, y)



    def predict(self, X):
        pred1 = self.reg.predict(X)
        pred2 = self.reg2.predict(X)
        return self.meta.predict(X)