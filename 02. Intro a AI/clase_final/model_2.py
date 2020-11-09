import numpy as np
import matplotlib.pyplot as plt
from metrics import *

class Data_Classification(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        structure = [('X1', np.float),
                     ('X2', np.float),
                     ('y', np.int)]

        with open(path, encoding="utf8") as data_csv:
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]), float(line.split(',')[2]))
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage):  # 0.8
        X = np.vstack((self.dataset['X1'].T, self.dataset['X2'].T)).T
        y = self.dataset['y']

        # X.shape[0] -> 10 (filas)

        permuted_idxs = np.random.permutation(X.shape[0])
        # 2,1,3,4,6,7,8,5,9,0

        train_idxs = permuted_idxs[0:int(percentage * X.shape[0])]
        # permuted_idxs[0:8]
        # [2,1,3,4,5,6,7,8,5]

        test_idxs = permuted_idxs[int(percentage * X.shape[0]): X.shape[0]]
        # [9,0]

        X_train = X[train_idxs]
        X_test = X[test_idxs]

        y_train = y[train_idxs]
        y_test = y[test_idxs]

        return X_train, X_test, y_train, y_test

def gradient_descent_logistic(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        a = np.matmul(X_train, W)
        prediction = sigmoid(a)  # nx1
        error = - prediction - y_train.reshape(-1, 1)  # nx1

        grad_sum = np.sum(error * X_train, axis=0)
        grad_mul = 1 / n * grad_sum  # 1xm
        gradient = np.transpose(grad_mul).reshape(-1, 1)

        W = W - (lr * gradient)

    return W

def stochastic_gradient_descent_logistic(X_train, y_train, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        for j in range(n):
            a = np.matmul(X_train, W)
            prediction = sigmoid(a)  # nx1
            error = - prediction - y_train.reshape(-1, 1)  # nx1

            grad_sum = np.sum(error * X_train, axis=0)
            grad_mul = 1 / n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)

            W = W - (lr * gradient)

    return W

def mini_batch_gradient_descent_logistic(X_train, y_train, b=16, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    for i in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]

        batch_size = int(len(X_train) / b)
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end]

            a = np.matmul(batch_X, W)
            prediction = sigmoid(a)  # nx1
            error = - prediction - batch_y.reshape(-1, 1)  # nx1

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = 1 / n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)

            W = W - (lr * gradient)

    return W

def sigmoid(X):
    return (1 / (1 + np.exp(-X)))

def z_score(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X-mean)/std

def probability(y, threshold=0.5):
    mask_true = np.argwhere(y >= threshold)
    mask_false = np.argwhere(y < threshold)
    y[mask_true[:,0]] = 1
    y[mask_false[:,0]] = 0
    return y.astype(int)