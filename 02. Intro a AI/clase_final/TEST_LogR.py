import numpy as np
from models import *
from metrics import *
from model_2 import *
import unittest
import time

class LogisticRegression(unittest.TestCase):

    dataset = Data_Classification('clase_6_dataset.txt')

    X_train, X_test, y_train, y_test = dataset.split(0.8)
    X_train, X_test = z_score(X_train), z_score(X_test)
    X_expanded = np.hstack((X_train, np.ones(X_train.shape[0]).reshape(X_train.shape[0], 1)))


    W_GD = gradient_descent_logistic(X_expanded, y_train, lr=0.001, amt_epochs=10000)
    W_SGD = stochastic_gradient_descent_logistic(X_train, y_train, lr=0.001, amt_epochs=1000)
    W_MBGD = mini_batch_gradient_descent_logistic(X_train, y_train, b=16, lr=0.05, amt_epochs=1000)

    weights = [W_GD, W_SGD, W_MBGD]
    i= 0
    y_hat = []
    y_hat_p = []
    precision_grad = []
    recall_grad = []
    F1_score_grad = []
    precision = Precision()
    recall = Recall()
    f1_score = F1_score()
    y_test = y_test.reshape(-1, 1)

    for weight in weights:
        y_hat.append(sigmoid(np.matmul(X_test, weight)))
        y_hat_p.append(probability(y_hat[i]))

        precision_grad.append(precision(y_test, y_hat[i]))
        recall_grad.append(recall(y_test, y_hat[i]))
        F1_score_grad.append(f1_score(y_test, y_hat[i]))
        i = i + 1

    print('Precision', precision_grad)
    print('Recall', recall_grad)
    print('F1 Score', F1_score_grad)

    # PLOTS
    # filter out the applicants that got admitted
    admitted = X_train[y_train == 1]
    # filter out the applicants that didn't get admission
    not_admitted = X_train[y_train == 0]

    # logistic regression
    x_regression = np.linspace(30, 100, 70)
    y_regression = (-x_regression * W_GD[0] - W_GD[2]) / W_GD[1]

    # plots
    plt.scatter(admitted['exam_1'], admitted['exam_2'], s=10, label='Admitted')
    plt.scatter(not_admitted['exam_1'], not_admitted['exam_2'], s=10, label='Not Admitted')
    plt.plot(x_regression, y_regression, '-', color='green', label='Regression')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    unittest.main()
