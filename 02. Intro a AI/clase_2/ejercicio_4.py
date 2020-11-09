import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

n_clusters = 50
inv_overlap = 10

def build_clusters(n_clusters, inv_overlap):
    # Será necesario agregar el type
    centroids = np.array([[1, 0], [0, 1]], dtype=np.float32)
    # Si divido por el overlap o multiplico por el inv_overlap entonces acerco los centroides entre si
    centroids = centroids * inv_overlap
    Cloud = np.repeat(centroids, n_clusters/2, axis=0)
    Cloud_rand = Cloud + np.random.normal(0, 1, size=Cloud.shape)
    # normal_noise = np.random.normal(loc=0, scale=1, size=(n_samples, 4))
    # data = data + normal_noise
    cluster_ids = np.array([[0], [1],])
    cluster_ids = np.repeat(cluster_ids, n_clusters / 2, axis=0)
    #
    plt.scatter(Cloud_rand[:, 0], Cloud_rand[:, 1])
    plt.show()
    return Cloud_rand, cluster_ids

#TODO Revisar los plots para diferentes centroides y overlaps - Porque toma valores cercanos a 10