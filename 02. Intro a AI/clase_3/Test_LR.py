import unittest
import time
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate
from models import *
from metrics import *

class LR(unittest.TestCase):
    dataset = Data('income.data.csv')
    n_order = 15
    X_train, X_test, y_train, y_test = dataset.split(0.8)
    X_train.reshape(X_train.shape[0], 1)
    X_train_0, X_test_0 = X_train, X_test
    y_hat = []
    lr_mse_train = np.zeros(n_order)
    lr_mse_test = np.zeros(n_order)
    mse = MSE()

    for i in range(n_order):
        if i == 0:
            constant_model = ConstantModel()
            constant_model.fit(X_train, y_train)
            y_hat.append(constant_model.predict(X_test))

            lr_mse_train[i] = mse(y_train, constant_model.predict(X_train.T))
        elif i < 2:
            linear_regression = LinearRegression()
            linear_regression.fit(X_train.T, y_train)
            y_hat.append(linear_regression.predict(X_test.T))

            lr_mse_train[i] = mse(y_train, linear_regression.predict(X_train.T))
        else:
            X_train, X_test = np.vstack((X_train, X_train_0 ** i)), np.vstack((X_test, X_test_0 ** i))
            linear_regression = LinearRegression()
            linear_regression.fit(X_train.T, y_train)
            y_hat.append(linear_regression.predict(X_test.T))

            lr_mse_train[i] = mse(y_train, linear_regression.predict(X_train.T))

        lr_mse_test[i] = mse(y_test, y_hat[i])

    x_plot = np.arange(0, n_order)
    l = np.vstack((x_plot.T, lr_mse_train.T, lr_mse_test.T))
    table = tabulate(l.T, headers=['Order Model', 'Training Error', 'Test Error'], tablefmt='fancy_grid', numalign='center')
    print(table)

    l_arg = np.argsort(lr_mse_train)
    l = np.vstack((x_plot[l_arg].T, lr_mse_train[l_arg].T, lr_mse_test[l_arg].T))
    table_min = tabulate(l.T, headers=['Order Model', 'Min Training Error', 'Test Error'], tablefmt='fancy_grid',
                     numalign='center')
    print(table_min)





    # Plot without constant aproximation
    # plt.plot(x_plot[1:], lr_mse_train[1:], color='b', label=f'Training MSE')
    # plt.plot(x_plot[1:], lr_mse_test[1:], color='r', label=f'Test MSE')

    # Plot with constant aproximation
    plt.plot(x_plot, lr_mse_train, color='b', label=f'Training MSE')
    plt.plot(x_plot, lr_mse_test, color='r', label=f'Test MSE')
    plt.legend()
    plt.show()



if __name__ == '__main__':
    unittest.main()