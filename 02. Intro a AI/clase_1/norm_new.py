import numpy as np

def vector_norm_l0(matrix):
    mask = matrix > 0
    return np.sum(mask, axis=1)

def vector_norm_l1(matrix):
    norm = np.sum(np.abs(matrix),axis=1)
    return norm

def vector_norm_l2(matrix):
    norm=np.sqrt(np.sum(np.abs(matrix**2),axis=1))
    return norm

def vector_norm_inf(matrix):
    norm=np.max(matrix,axis=1)
    return norm