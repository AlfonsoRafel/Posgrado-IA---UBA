import numpy as np
import matplotlib.pyplot as plt
from metrics import *


class BaseModel(object):

    def _init_(self):
        self.model = None

    def fit(self, X, y):
        # train del model
        return NotImplemented

    def predict(self, X):
        # return Y hat
        return NotImplemented

class ConstantModel(BaseModel):

    def fit(self, X, Y):
        W = Y.mean()
        self.model = W

    def predict(self, X):
        return np.ones(len(X)) * self.model

class LinearRegression(BaseModel):

    def fit(self, X, y):
        if len(X.shape) == 1:
            W = X.T.dot(y) / X.T.dot(X)
        else:
            W = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
        self.model = W

    def predict(self, X):
        return X.dot(self.model)

class LinearRegressionWithB(BaseModel):

    def fit(self, X, y):
        X_expanded = np.vstack((X, np.ones(len(X)))).T
        W = np.linalg.inv(X_expanded.T.dot(X_expanded)).dot(X_expanded.T).dot(y)
        self.model = W

    def predict(self, X):
        X_expanded = np.vstack((X, np.ones(len(X)))).T
        return X_expanded.dot(self.model)


def gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
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
        prediction = np.matmul(X_train, W)  # nx1
        error = y_train - prediction  # nx1

        grad_sum = np.sum(error * X_train, axis=0)
        grad_mul = -2/n * grad_sum  # 1xm
        gradient = np.transpose(grad_mul).reshape(-1, 1)

        W = W - (lr * gradient)

    return W

def stochastic_gradient_descent(X_train, y_train, lr=0.01, amt_epochs=100):
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
            prediction = np.matmul(X_train[j].reshape(1, -1), W)  # 1x1
            error = y_train[j] - prediction  # 1x1

            grad_sum = error * X_train[j]
            grad_mul = -2/n * grad_sum  # 2x1
            gradient = np.transpose(grad_mul).reshape(-1, 1)  # 2x1

            W = W - (lr * gradient)

    return W

def mini_batch_gradient_descent(X_train, y_train, X_test, y_test, rr=0, lr=0.01, amt_epochs=100):
    """
    shapes:
        X_t = nxm
        y_t = nx1
        W = mx1
    """
    b = 15
    n = X_train.shape[0]
    m = X_train.shape[1]

    # initialize random weights
    W = np.random.randn(m).reshape(m, 1)

    error_train = np.zeros(amt_epochs)
    error_test = np.zeros(amt_epochs)

    for j in range(amt_epochs):
        idx = np.random.permutation(X_train.shape[0])
        X_train = X_train[idx]
        y_train = y_train[idx]
        mse = MSE()
        batch_size = int(len(X_train) / b)
        for i in range(0, len(X_train), batch_size):
            end = i + batch_size if i + batch_size <= len(X_train) else len(X_train)
            batch_X = X_train[i: end]
            batch_y = y_train[i: end]

            prediction = np.matmul(batch_X, W)  # nx1
            error = batch_y - prediction  # nx1

            grad_sum = np.sum(error * batch_X, axis=0)
            grad_mul = -2/n * grad_sum  # 1xm
            gradient = np.transpose(grad_mul).reshape(-1, 1)   # mx1
            rigde_reg = 2 * rr * W # Regularization term
            W = W - (lr * gradient + rigde_reg)
        prediction_test = np.matmul(X_test, W)
        error_train[j] = mse(batch_y, prediction)
        error_test[j] = mse(y_test, prediction_test)
    return W, error_train, error_test



def k_folds(X_train, y_train, X_test, y_test, k=5):
    l_regression = LinearRegressionWithB()
    error = MSE()
    l_reg_models = []
    chunk_size = int(len(X_train) / k)
    mse_list_train = []
    mse_list_test = []
    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        # Validation fold
        new_X_valid = X_train[i: end]
        new_y_valid = y_train[i: end]
        # Picks other n-1 folds to train
        new_X_train = np.concatenate([X_train[: i], X_train[end:]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        l_regression.fit(new_X_train, new_y_train)
        prediction = l_regression.predict(new_X_valid)
        l_reg_models.append(l_regression.model)
        mse_list_train.append(error(new_y_valid, prediction))
        validation = l_regression.predict(X_test)
        mse_list_test.append(error(y_test, validation))

    return mse_list_train, mse_list_test, l_reg_models

def k_folds_n(X_train, y_train, X_test, y_test, k=5):
    l_regression = LinearRegression()
    error = MSE()
    l_reg_models = []
    chunk_size = int(len(X_train) / k)
    mse_list_train = []
    mse_list_test = []
    for i in range(0, len(X_train), chunk_size):
        end = i + chunk_size if i + chunk_size <= len(X_train) else len(X_train)
        # Validation fold
        new_X_valid = X_train[i: end, :]
        new_y_valid = y_train[i: end]
        # Picks other n-1 folds to train
        new_X_train = np.concatenate([X_train[: i, :], X_train[end:, :]])
        new_y_train = np.concatenate([y_train[: i], y_train[end:]])

        l_regression.fit(new_X_train, new_y_train)
        prediction = l_regression.predict(new_X_valid)
        l_reg_models.append(l_regression.model)
        mse_list_train.append(error(new_y_valid, prediction))
        validation = l_regression.predict(X_test)
        mse_list_test.append(error(y_test, validation))

    return mse_list_train, mse_list_test, l_reg_models





