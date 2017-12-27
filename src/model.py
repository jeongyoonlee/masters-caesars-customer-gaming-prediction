from scipy.optimize import minimize
from sklearn import base
from sklearn.utils import check_random_state
import numpy as np
import logging

from evaluate import kappa


class Minimizer(base.BaseEstimator):

    def __init__(self, algo='Nelder-Mead', tol=1e-6, random_state=None):
        self.algo = algo
        self.tol = tol
        self.random_state = check_random_state(random_state)

    def fit(self, X, y):
        X = np.asarray(X)
        res = minimize(lambda x: -kappa(y, X.dot(x)),
                       x0=self.random_state.rand(X.shape[1]),
                       method=self.algo,
                       tol=self.tol)
        self.coef_ = res.x

        return self

    def predict(self, X):
        X = np.asarray(X)
        return X.dot(self.coef_)
