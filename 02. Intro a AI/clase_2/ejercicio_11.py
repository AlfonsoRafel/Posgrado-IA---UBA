import numpy as np

def split(x):
    indices = np.random.permutation(x.shape[0])
    ind_train = np.round(0.7 * len(indices)).astype(int)
    ind_val = np.round(0.9 * len(indices)).astype(int)
    train_idx, val_idx, test_idx = indices[:ind_train], indices[ind_train:ind_val], indices[ind_val:]
    training, validation, test = x[train_idx, :], x[val_idx, :], x[test_idx, :]
    return training, validation, test