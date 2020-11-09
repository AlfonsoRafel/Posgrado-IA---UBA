import numpy as np
from scipy.stats import zscore

X = np.array([[200, 7, 3], [4, 5, 8], [4, 5, 8]])
expected_value = zscore(X, axis=0)

def z_score(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X-mean)/std


