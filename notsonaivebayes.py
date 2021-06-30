#coding=utf-8
'''
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
https://stackoverflow.com/questions/54296405/what-is-the-corresponding-function-for-mean-validation-score-in-grid-cv-results
https://blog.csdn.net/qq_21579045/article/details/91435570

@author: DMao
@time: 21.07.20    15:36
'''
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity

class KDEClassifier(BaseEstimator, ClassifierMixin):
    """
    Parameters
    ----------
    bandwidth: float
    kernel: str
    """
    def __init__(self, bandwidth=1.0, kernel='gaussian'):
        self.bandwidth = bandwidth
        self.kernel = kernel

    # fit() to handel the training data
    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        training_sets = [X[y == yi] for yi in self.classes_]
        self.models_ = [KernelDensity(bandwidth=self.bandwidth, kernel=self.kernel).fit(Xi) for Xi in training_sets]
        self.logpriors_ = [np.log(Xi.shape[0] / X.shape[0]) for Xi in training_sets]
        return self

    # predict labels on new data
    def predict_proba(self, X):
        logprobs = np.vstack([model.score_samples(X) for model in self.models_]).T
        result = np.exp(logprobs + self.logpriors_)
        return result / result.sum(1, keepdims=True)

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), 1)]

from sklearn.datasets import load_digits
from sklearn.model_selection import GridSearchCV

digits = load_digits()
bandwidths = 10 ** np.linspace(0, 2, 100)
grid = GridSearchCV(KDEClassifier(), {'bandwidth': bandwidths})
grid.fit(digits.data, digits.target)


import pandas as pd
cv_results = pd.DataFrame(grid.cv_results_)
print(type(cv_results['mean_test_score']))
#print(cv_results['mean_test_score'])
print(type(grid.cv_results_['mean_test_score']))

#scores = [ val.mean_validation_score for val in grid.cv_results_]

scores = grid.cv_results_['mean_test_score']


plt.semilogx(bandwidths, scores)
plt.xlabel('bandwidth')
plt.ylabel('accuracy')
plt.title('KDE Model Performance')
print(grid.best_params_)
print('accuracy=  ', grid.best_score_)
plt.show()

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
r = cross_val_score(GaussianNB(), digits.data, digits.target).mean()
print(r)

