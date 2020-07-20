"""

 linear_regr.py  (author: Anson Wong / git: ankonzoid)

"""
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

class lasso:

    def __init__(self):
        from sklearn.linear_model import Lasso
        self.model = Lasso(alpha = 0.2, fit_intercept = False, max_iter = 1e6, normalize = True, tol = 1e-3)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def loss(self, X, y, y_pred):
        return mean_squared_error(y, y_pred)

