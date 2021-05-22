import scipy as sp
import numpy as np
import pandas as pd

from functools import partial


'''
https://skywalker0803r.medium.com/kaggle%E7%AB%B6%E8%B3%BD-kernel%E5%B0%8E%E8%AE%80-7e5114712433
https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved
optR = OptimizedRounder()
optR.fit(outputs, labels)
coefficients = optR.coefficients()
opt_outputs = optR.predict(predictions, coefficients)
'''
class OptimizedRounder(object):
    """
    An optimizer for rounding thresholds to maximize score
    :param score: Using score funtion
    :param labels: label list
    :param initial_coef: label rate coef list
    """
    def __init__(self, score, labels, initial_coef):
        self.coef_ = 0
        self.score_ = score
        self.labels_ = labels
        self.initial_coef_ = initial_coef
    
    """
    Get loss according to
    using current coefficients
    :param coef: A list of coefficients that will be used for rounding
    :param X: The raw predictions
    :param y: The ground truth labels
    """
    def _loss_(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = self.labels_)
        return -self.score_(y, preds, weights = 'quadratic')
    
    """
    Optimize rounding thresholds
    :param X: The raw predictions
    :param y: The ground truth labels
    """
    def fit(self, X, y):
        loss_partial = partial(self._loss_, X = X, y = y)
        initial_coef = self.initial_coef_
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef, method = 'nelder-mead')
    
    """
    Make predictions with specified thresholds
    :param X: The raw predictions
    :param coef: A list of coefficients that will be used for rounding
    """
    def predict(self, X, coef):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels = self.labels_)
        return preds
    
    """
    Return the optimized coefficients
    """
    def coefficients(self):
        return self.coef_['x']