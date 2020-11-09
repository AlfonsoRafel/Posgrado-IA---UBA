import numpy as np



def filtrar_NaN (x):
    x = x[~np.isnan(x).any(axis=1)]
    return x