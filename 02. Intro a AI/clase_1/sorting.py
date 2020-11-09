import numpy as np
from norm_new import vector_norm_l2

def sorting_vectors_by_norm_l2(matrix):
    norm = vector_norm_l2(matrix)
    norm_arg = np.argsort(norm*-1)
    return matrix[norm_arg, :]