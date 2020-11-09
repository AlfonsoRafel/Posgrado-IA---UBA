import numpy as np
from tabulate import tabulate
from Preprocessing import *
from models import *


# 2. Pre-procesamiento del dataset
# 2. a - Obtener el dataset desde el siguiente link. La primera columna representa
# los datos de entrada y la segunda columna representa los datos de salida.
dataset = Data('clase_8_dataset.csv')

# 2. b - Levantar el dataset en un arreglo de Numpy.
X = dataset.dataset['X']
y = dataset.dataset['y']

# 2. c - Graficar el dataset de manera tal que sea posible visualizar la nube de puntos.
plot_1 = plt.figure(1)
plt.ylabel('y')
plt.xlabel('X')
plt.title('Nube de puntos del dataset')
plt.scatter(X, y, color='b')
plt.show()

# 2. d -  Partir el dataset en train (80%) y test (20%).
X_train, X_test, y_train, y_test = dataset.split(0.8)

# 3. Utilizar regresión polinómica para hacer “fit” sobre la nube de puntos del train.
# Para este ejercicio, se desea utilizar la fórmula cerrada de la optimización polinómica.
# El modelo es de la forma y = [Wn … W0] * [X^n    X^(n-1)    …    1].

# 3. a - Para n = 1 (modelo lineal con ordenada al origen), hacer un fit del modelo utilizando K-FOLDS.
# Para K-FOLDS partir el train dataset en 5 partes iguales, utilizar 4/5 para entrenar y 1/5 para validar.
# Informar el mejor modelo obtenido y el criterio utilizado para elegir dicho modelo (dejar comentarios en el código).
mse_list_train, mse_list_test, l_reg_models = k_folds(X_train, y_train, X_test, y_test, k=5)
mse_min = np.argmin(mse_list_test)

arg = X_test.argsort(axis=0)
x_plot = X_test[arg]
X_expanded = expand_ones(x_plot.reshape(-1, 1))
y_hat = np.matmul(X_expanded, l_reg_models[mse_min])

plot1 = plt.figure(2)
plt.ylabel('y')
plt.xlabel('X')
plt.scatter(x_plot, y_test[arg], color='r', label=f'Test data')
plt.plot(x_plot, y_hat, color='b', label=f'Best Linear Aprox')
plt.legend()
plt.show()

n_fold = np.arange(0, len(mse_list_test))
l = np.vstack((n_fold.T, np.asarray(mse_list_train).T, np.asarray(mse_list_test).T))
table = tabulate(l.T, headers=['Fold number', 'Training Error', 'Test Error'], tablefmt='fancy_grid',
                 numalign='center')
print(table)


# Se observa de la tabla que para el fold 1 se obtiene el mejor (es decir menor) error sobre el training set.
# Más allá de eso los errores sobre el dataset de test se mantienen alrededor del mismo valor.
# Como estamos ante una nube de punto con características claramente no lineales, los valores obtenidos para los errores
# son altos, lo cual se verifica en la figura anterior. Considerando los resultados del k-fold, el mejor modelo obtenido
# para n=1 corresponde al fold que arroja el menor test error.


# 3. b - Repetir el punto (a), para n = {2,3,4}. Computar el error de validación y test del mejor modelo para cada n.
best_model = []
mse_min_list = []
mse_train_min_list = []
X_train_pol = X_train
X_test_pol = X_test
for i in range(2, 5):
    X_train_pol, X_test_pol = np.vstack((X_train_pol, X_train ** i)), np.vstack((X_test_pol, X_test ** i))
    X_train_expanded, X_test_expanded = np.vstack((np.ones(X_train_pol.shape[1]), X_train_pol)), np.vstack((np.ones(X_test_pol.shape[1]), X_test_pol))
    mse_list_train, mse_list_test, l_reg_models = k_folds_n(X_train_expanded.T, y_train,
                                                            X_test_expanded.T, y_test, k=5)
    mse_min_arg = np.argmin(mse_list_test)
    best_model.append(l_reg_models[mse_min_arg])
    mse_min_list.append(mse_list_test[mse_min_arg])
    mse_train_min_list.append(mse_list_train[mse_min_arg])


n_fold = np.arange(2, 5)
l = np.vstack((n_fold.T, np.asarray(mse_train_min_list).T, np.asarray(mse_min_list).T))
table = tabulate(l.T, headers=['Order', 'Best Training Error', 'Best Test Error'], tablefmt='fancy_grid',
                 numalign='center')
print(table)

# 3. c - Elegir el polinomio que hace mejor fit sobre la nube de puntos y explicar el criterio seleccionado (dejar comentarios en el código).
# Primero se elije el modelo para cada orden que arroje el menor valor de error de test. Una vez obtenido los mejores tres
# se analiza cual es el polinomio que mejor fitea a la nube de puntos. Considerando los valores de error de test, siendo
# el error aproximadamente 10 veces más chicos para n=3 y n=4 que para n=2, se dercarta el modelo cuadrático. Entre los dos
# restantes, como la diferencia entre los errores de test es mínima, se opta por el elegir el modelo de orden n=3, debido a
# la menor complejidad.

# 3. d - Graficar el polinomio obtenido y el dataset de test.
mse_min = np.argmin(mse_min_list)
arg = X_test.argsort(axis=0)
x_plot = X_test[arg]

plot1 = plt.figure(3)
plt.ylabel('y')
plt.xlabel('X')
for i in range(0,3):
    y_hat = np.matmul(X_test_expanded[:i+3, :].T, best_model[i].T)
    plt.plot(x_plot, y_hat[arg], label=f'Polinomial n=' + str(i + 2)+'- MSE =' + str(np.round(mse_min_list[i], 2)))
plt.scatter(x_plot, y_test[arg], color='y', label=f'Test data')

plt.legend()
plt.show()



# 4. Para el mejor modelo seleccionado en (3c) (el mejor “n”),
# hacer la optimización utilizando Mini-Batch Gradient Descent
# (partir el train dataset en 4/5 para entrenar y 1/5 para validar).
# Se eligió el modelo n=3 considerando las variables de error de test y complejidad.
X_train_scaled = z_score(X_train)
X_test_scaled = z_score(X_test)
X_train_pol = X_train_scaled
X_test_pol = X_test_scaled
for i in range(2, 4):
    X_train_pol, X_test_pol = np.vstack((X_train_pol, X_train_scaled ** i)), np.vstack((X_test_pol, X_test_scaled ** i))
X_train_expanded_new, X_test_expanded_new = np.vstack((np.ones(X_train_pol.shape[1]), X_train_pol)), \
                                    np.vstack((np.ones(X_test_pol.shape[1]), X_test_pol))

# 4. a - Para cada epoch, calcular el error de train y el error de validation.
W, error_train, error_test = mini_batch_gradient_descent(X_train_expanded_new.T, y_train.reshape(-1, 1),
                                                         X_test_expanded_new.T, y_test.reshape(-1, 1),
                                                         lr=0.01, amt_epochs=1000)

# 4. b - Graficar el error de train y el error de validación en función del número de epoch.
arg = X_test.argsort(axis=0)
x_plot = X_test[arg]
y_hat_LR = np.matmul(X_test_expanded[:4, :].T, best_model[1].T)
y_hat_MBGD = np.matmul(X_test_expanded_new.T, W)
last_error = error_test[-1]

plot1 = plt.figure(4)
plt.ylabel('y')
plt.xlabel('X')
plt.scatter(x_plot, y_test[arg], color='r', label=f'Test data')
plt.plot(x_plot, y_hat_LR[arg], color='b', label=f'Polinomial n=3 - MSE =' + str(np.round(mse_min_list[1], 2)))
plt.plot(x_plot, y_hat_MBGD[arg], color='g', label=f'MBGD n=3 - MSE=' +  str(np.round(error_test[-1], 2)))
plt.legend()
plt.show()

# 4. c - Comparar los resultados obtenidos para el modelo entrenado con Mini-Batch, contra el modelo obtenido en (3c)
plot1 = plt.figure(5)
epochs = np.arange(0, len(error_train))
plt.ylabel('MSE')
plt.xlabel('Epochs')
plt.plot(epochs, error_train, color='b', label=f'Error de Entrenamiento')
plt.plot(epochs, error_test, color='g', label=f'Error de Test')
plt.legend()
plt.show()

# Se observa que el error de test es mayor para el caso del algoritmo MBGD, lo cual en parte se debe a la dificultad para
# tunear los hiperparámetros.



# 5. [EXTRA] Para el mejor modelo seleccionado en (3c), hacer la optimización utilizando Mini-Batch y regularización Ridge.
# Se agrega una regularización muy baja denotada por el parámetro rr, para no causar la divergencia del modelo.
W, error_train, error_test = mini_batch_gradient_descent(X_train_expanded_new.T, y_train.reshape(-1, 1),
                                                         X_test_expanded_new.T, y_test.reshape(-1, 1), rr=0.0001,
                                                         lr=0.01, amt_epochs=1000)

arg = X_test.argsort(axis=0)
x_plot = X_test[arg]
y_hat_LR = np.matmul(X_test_expanded[:4, :].T, best_model[1].T)
y_hat_MBGD_reg = np.matmul(X_test_expanded_new.T, W)

plot1 = plt.figure(6)
plt.ylabel('y')
plt.xlabel('X')
plt.scatter(x_plot, y_test[arg], color='r', label=f'Test data')
plt.plot(x_plot, y_hat_LR[arg], color='b', label=f'Polinomial n=3 - MSE =' + str(np.round(mse_min_list[1], 2)))
plt.plot(x_plot, y_hat_MBGD[arg], color='g', label=f'MBGD n=3 - MSE=' +  str(np.round(last_error, 2)))
plt.plot(x_plot, y_hat_MBGD_reg[arg], color='y', label=f'MBGD with reg n=3 - MSE=' +  str(np.round(error_test[-1], 2)))
plt.legend()
plt.show()
