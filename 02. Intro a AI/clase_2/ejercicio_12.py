import numpy as np
import pickle
from matplotlib import pyplot as plt
from ejercicio_8 import pca
from ejercicio_3 import k_means, k_means_loop

# 1. Generar un dataset sintético que clusterice data en 4 clusters utilizando números random.
# a. Utilizar 4 dimensiones.
# b. Generar un dataset con 100K de muestras.
sintetic = np.random.normal(0, 1, size=(100000, 4))
centroids = sintetic[np.random.choice(len(sintetic), size=4, replace=False)]

# 2. Cambiar algunos puntos de manera aleatoria y agregar NaN (0.1% del dataset).
sintetic_flatten = np.matrix.flatten(sintetic, order='C')
sintetic_flatten = np.random.permutation(sintetic_flatten)
nan_indices = np.random.random_integers(0, len(sintetic_flatten), size=np.round(0.01 * len(sintetic_flatten)).astype(int))
sintetic_flatten[nan_indices] = np.nan
sintetic = np.reshape(sintetic_flatten, (100000, 4))

# 3. Guardar el dataset en un .pkl
pickle.dump(sintetic, open('sintetic' + '.pkl', 'wb'))

# 4. Cargar el dataset con Numpy desde el .pkl
sintetic = pickle.load(open('sintetic' + '.pkl', "rb"))

# 5. Completar NaN con la media de cada feature.
col_mean = np.nanmean(sintetic, axis=0)
inds = np.where(np.isnan(sintetic))
sintetic[inds] = np.take(col_mean, inds[1])

# 6. Calcular la norma l2, la media y el desvío de cada feature con funciones numpy vectorizadas.
norm_l2 = np.sqrt(np.sum(np.abs(sintetic**2), axis=0))
mean_feature = np.mean(sintetic, axis=0)
std_feature = np.std(sintetic, axis=0)

# 7. Agregar una columna a partir de generar una variable aleatoria exponencial a todos los puntos.
uniform_random_variable = np.random.uniform(low=0.0, high=1.0, size=sintetic.shape[0])
lambda_param = 5
exp = (-1 / lambda_param) * np.log(1 - uniform_random_variable)
exp = exp[:, None]
sintetic = np.append(sintetic, exp, axis=1)

# 8. Hacer el histograma de la distribución exponencial.
plt.hist(sintetic[:,4])
plt.title("histogram")
plt.show()
#TODO agregar mas bins

# 9. Aplicar PCA al dataset reduciendo a 2 dimensiones y graficar el cluster.
sintetic_pca = pca(sintetic, components=2)
plt.scatter(sintetic_pca[:,0], sintetic_pca[:,1])
plt.show()

# 10. Hacer la clusterización con el k-means desarrollado en clase.
centroids, cluster_ids = k_means(sintetic_pca, 10, 1)

# 11. Volver a graficar el cluster con lo obtenido en (10) y comparar resultados con (9).
plt.scatter(sintetic_pca[:,0], sintetic[:,1], c=cluster_ids)
plt.show()

# 12. Analizar que pasa si los clusters comienzan a tener overlapping.
centroids, cluster_ids = k_means(sintetic_pca, 3, 0.1)
plt.scatter(sintetic_pca[:,0], sintetic[:,1], c=cluster_ids)
plt.show()
