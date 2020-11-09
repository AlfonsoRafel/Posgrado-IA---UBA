import numpy as np
from model_2 import *
import matplotlib.pyplot as plt


def XOR(X,Y):
    b = np.ones(X.shape[0]).reshape(-1,1)
    X = np.hstack((X,b))
    W = np.linalg.inv(np.matmul(X.T, X)) @ X.T @ Y
    J = np.sum((Y.reshape(-1, 1) - X @ W.reshape(-1, 1)) ** 2)
    return W, J

def NN_XOR(X, Y, lr = 0.01, b = 4, amt_nn = [2,1], n_epochs = 100):
    # Parameter Initialization
    J_seq = np.zeros(n_epochs)
    W1 = np.random.random(size=(amt_nn[0], X.shape[1]))
    W2 = np.random.random(size=(amt_nn[1], amt_nn[0]))
    b1 = np.ones(shape=amt_nn[0]).reshape(-1,1)
    b2 = np.ones(shape=amt_nn[1]).reshape(-1,1)
    m = X.shape[0] # Cantidad de datos de entrenamiento

    dW1 = np.zeros(shape=(amt_nn[0], X.shape[1]))
    dW2 = np.zeros(shape=(amt_nn[1], amt_nn[0]))
    db2 = np.zeros(shape=amt_nn[1]).reshape(-1, 1)

    for i in range(n_epochs):
        #idx = np.random.permutation(X.shape[0])
        #X = X[idx]
        #Y = Y[idx]
        batch_size = int(len(X) / b)
        for r in range(0, len(X), batch_size):
            end = r + batch_size if r + batch_size <= len(X) else len(X)
            batch_X = X[r: end]
            batch_Y = Y[r: end]
            J = 0
            for j in range(0,len(batch_X)):
                # Forward propagation
                Z1 = batch_X[j, 0] * W1[0, 0] + batch_X[j, 1] * W1[0, 1] + b1[0]
                Z2 = batch_X[j, 0] * W1[1, 0] + batch_X[j, 1] * W1[1, 1] + b1[1]
                a1 = sigmoid(Z1)
                a2 = sigmoid(Z2)
                y_hat = a1 * W2[0, 0] + a2 * W2[0, 1] + b2

                # Backward propagation
                error = batch_Y[j] - y_hat
                dW2[0, 0] = -2 * error * a1
                dW2[0, 1] = -2 * error * a2
                db2 = -2 * error

                dW1[0, 0] = (-2 * error * W2[0, 0]) * (a1 * (1 - a1)) * batch_X[j, 0]
                dW1[0, 1] = (-2 * error * W2[0, 0]) * (a1 * (1 - a1)) * batch_X[j, 1]
                dW1[1, 0] = (-2 * error * W2[0, 1]) * (a2 * (1 - a2)) * batch_X[j, 0]
                dW1[1, 1] = (-2 * error * W2[0, 1]) * (a2 * (1 - a2)) * batch_X[j, 1]


                # Weights update
                W2[0, 0] = W2[0, 0] - lr * dW2[0, 0]
                W2[0, 1] = W2[0, 1] - lr * dW2[0, 1]
                b2 = b2 - lr * db2

                W1[0, 0] = W1[0, 0] + lr * dW1[0, 0]
                W1[0, 1] = W1[0, 1] + lr * dW1[0, 1]
                W1[1, 0] = W1[1, 0] + lr * dW1[1, 0]
                W1[1, 1] = W1[1, 1] + lr * dW1[1, 1]

            # Cost update
            y_hat = a1 * W2[0, 0] + a2 * W2[0, 1] + b2
            J += (Y[j] - y_hat) ** 2/batch_size
        J_seq[i] = J
    return W1, W2, b1, b2, J_seq

def NN_XOR_predict(X, Y, W1, W2, b1, b2, threshold = 0.5):
    y_hat = np.zeros(X.shape[0]).reshape(-1, 1)
    for j in range(X.shape[0]):
        Z1 = X[j, 0] * W1[0, 0] + X[j, 1] * W1[0, 1] + b1[0]
        Z2 = X[j, 0] * W1[1, 0] + X[j, 1] * W1[1, 1] + b1[1]
        a1 = sigmoid(Z1)
        a2 = sigmoid(Z2)
        y_hat[j] = a1 * W2[0, 0] + a2 * W2[0, 1] + b2
        if y_hat[j]>threshold:
            y_hat[j] = 1
        else:
            y_hat[j] = 0
    return y_hat


if __name__ == "__main__":
    X = np.array([[0, 0, 1, 1], [0, 1, 0, 1]]).T
    Y = np.array([0, 1, 1, 0]).T

    W, J = XOR(X, Y)
    print('W_xor :', W)
    print('J_xor :', J)

    W1, W2, b1, b2, J_seq = NN_XOR(X, Y, lr=0.01)
    print('W1 :', W1)
    print('W2 :', W2)
    print('b2 :', b2)
    print('b1 :', b1)
    # print('J_nn :', J_seq)
    y_hat = NN_XOR_predict(X, Y, W1, W2, b1, b2)
    print('Total Error :', np.sum(Y-y_hat))

    W1, W2, b1, b2, J_seq2 = NN_XOR(X, Y, lr=0.1)
    # print('W_nn :', W)
    #print('J_nn :', J_seq2)


    W1, W2, b1, b2,J_seq3 = NN_XOR(X, Y, lr=0.001)
    # print('W_nn :', W)
    #print('J_nn :', J_seq2)

    plot = plt.figure(1)
    plt.ylabel('Cost Function')
    plt.xlabel('n_epochs')
    x_plot = np.arange(0, len(J_seq))
    a = 1
    plt.plot(x_plot, J_seq, label=f'NN_XOR - LR=' + str(0.01))
    plt.plot(x_plot, J_seq2, label=f'NN_XOR - LR=' + str(0.1))
    plt.plot(x_plot, J_seq3, label=f'NN_XOR - LR=' + str(0.001))
    plt.plot(x_plot, J * np.ones(len(x_plot)), label=f'XOR_Closed Solution')
    plt.legend()
    plt.show()

    W1, W2, b1, b2, J_seq4 = NN_XOR(X, Y, b=3,  lr=0.01)
    # print('W_nn :', W)
    #print('J_nn :', J_seq4)

    W1, W2, b1, b2, J_seq5 = NN_XOR(X, Y, b=2, lr=0.01)
    # print('W_nn :', W)
    #print('J_nn :', J_seq5)

    plot = plt.figure(2)
    plt.ylabel('Cost Function')
    plt.xlabel('n_epochs')
    x_plot = np.arange(0, len(J_seq))
    a = 1
    plt.plot(x_plot, J_seq, label=f'NN_XOR - b=' + str(4))
    plt.plot(x_plot, J_seq4, label=f'NN_XOR - b=' + str(3))
    plt.plot(x_plot, J_seq5, label=f'NN_XOR - b=' + str(2))
    plt.legend()
    plt.show()


#TODO SGD
#TODO COSTO/BATCH
#TODO b2 parameter
