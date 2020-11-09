import numpy as np

x = np.array([[1, 3, np.nan], [1, 3, 5], [np.nan, 3, 5], [1, 3, 5], [1, 3, 7], [1, 3, 5]])

def replace_nan(x):
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    return x