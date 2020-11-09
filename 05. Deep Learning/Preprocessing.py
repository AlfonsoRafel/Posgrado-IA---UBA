import numpy as np
import matplotlib.pyplot as plt
import pickle

# Open file and split data
# 1. Must set structure
# 2. Set data gen with datatypes
# 3. Set data split

class Data(object):

    def __init__(self, path):
        self.dataset = self._build_dataset(path)

    def _build_dataset(self, path):
        # Set structure
        structure = [('X', np.float),
                     ('y', np.float)]
        with open(path, encoding="utf8") as data_csv:
            # Set data_gen
            data_gen = ((float(line.split(',')[0]), float(line.split(',')[1]))
                        for i, line in enumerate(data_csv) if i != 0)
            embeddings = np.fromiter(data_gen, structure)

        return embeddings

    def split(self, percentage):  # 0.8
        # Set X & y
        X = self.dataset['X']
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
class SyntheticDataset(object):

    def __init__(self, n_order, n_samples, n_clusters,  inv_overlap):
        self.n_clusters = n_clusters
        self.n_order = n_order
        self.n_samples = n_samples
        self.inv_overlap = inv_overlap
        self.data, self.cluster_ids = self._build_cluster()

    def train_valid_split(self):
        idxs = np.random.permutation(self.n_samples)
        n_train_samples = int(self.n_samples * 0.8)
        train = self.data[idxs[:n_train_samples]]
        train_cluster_ids = self.cluster_ids[idxs[:n_train_samples]]
        valid = self.data[idxs[n_train_samples:]]
        valid_cluster_ids = self.cluster_ids[idxs[n_train_samples:]]
        return train, train_cluster_ids, valid, valid_cluster_ids

    @staticmethod
    def pca(X, d):
        x2 = (X - X.mean(axis=0))
        cov_1 = np.cov(x2.T)
        w, v = np.linalg.eig(cov_1)
        idx = w.argsort()[::-1]
        w = w[idx]
        v = v[:, idx]
        return np.matmul(x2, v[:, :d])

    @staticmethod
    def plot_cluster(low_dim_dataset, cluster_ids):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        scatter = ax.scatter(low_dim_dataset[:, 0], low_dim_dataset[:, 1], c=cluster_ids, s=50)
        fig.show()

    def _build_cluster(self):
        centroids = np.random.random(size=(self.n_clusters, self.n_order))
        centroids = centroids * self.inv_overlap
        data = np.repeat(centroids, self.n_samples / 2, axis=0)
        normal_noise = np.random.normal(loc=0, scale=1, size=(self.n_samples, self.n_order))
        data = data + normal_noise

        cluster_ids = np.arange(0, self.n_clusters)
        cluster_ids = np.repeat(cluster_ids, self.n_samples / self.n_clusters, axis=0)

        return data, cluster_ids



def pca(X, d):
    x2 = (X - X.mean(axis=0))
    cov_1 = np.cov(x2.T)
    w, v = np.linalg.eig(cov_1)
    idx = w.argsort()[::-1]
    w = w[idx]
    v = v[:, idx]
    return np.matmul(x2, v[:, :d])
def replace_nan(x):
    col_mean = np.nanmean(x, axis=0)
    inds = np.where(np.isnan(x))
    x[inds] = np.take(col_mean, inds[1])
    return x
def expand_ones(X):
    X_expanded = np.hstack((X, np.ones(X.shape[0]).reshape(X.shape[0],1)))
    return X_expanded
def z_score(X):
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return (X-mean)/std
def add_nan(X, percentage):
    sintetic_flatten = np.matrix.flatten(X, order='C')
    sintetic_flatten = np.random.permutation(sintetic_flatten)
    nan_indices = np.random.random_integers(0, len(sintetic_flatten),
                                            size=np.round(percentage * len(sintetic_flatten)).astype(int))
    sintetic_flatten[nan_indices] = np.nan
    return np.reshape(sintetic_flatten, (X.shape[0], X.shape[1]))
def save_pickle(X, name):
    pickle.dump(X, open(name + '.pkl', 'wb'))
def load_pickle(name):
    return pickle.load(open(name + '.pkl', "rb"))

def exponential_random_variable(lambda_param, size):
    uniform_random_variable = np.random.uniform(low=0.0, high=1.0, size=size)
    return (-1 / lambda_param) * np.log(1 - uniform_random_variable)
def k_means(X, n_clusters, inv_overlap, MAX_ITERATIONS):
    # Al definir los centroides con np.eye uno de los centroides termina arrojando valores NaN
    # centroids = np.eye(n_clusters, X.shape[1])
    centroids = X[np.random.choice(len(X), size=n_clusters, replace=False)]
    centroids = centroids * inv_overlap
    print(centroids)
    for i in range(MAX_ITERATIONS):
        print("Iteration # {}".format(i))
        centroids, cluster_ids = k_means_loop(X, centroids)
        print(centroids)
    return centroids, cluster_ids
def k_means_loop(X, centroids):
    # find labels for rows in X based in centroids values
    expanded_centroids = centroids[:, None]
    distances = np.sqrt(np.sum((expanded_centroids - X) ** 2, axis=2))
    arg_min = np.argmin(distances, axis=0)
    # recompute centroids
    for i in range(centroids.shape[0]):
        centroids[i] = np.mean(X[arg_min == i, :], axis=0)

    return centroids, arg_min


def adapt_data_order(X, order):
    """
    Toma un dataset X, devuelve el dataset que corresponde a un polinomio de orden 'orden'.
    Cada columna del dataset devuelto corresponde a X^i, con i creciente hasta orden.
    NO TIENE EN CUENTA EL ORDEN 0!
    In:
    ndarray
    Out:
    ndarray
    """
    X_repeat_order = np.repeat(X, order, axis=1)
    orders = np.array(range(1, order + 1))
    X_order = np.apply_along_axis(np.power, 0, X_repeat_order.T, orders).T

    return X_order
def adapt_multidimensional_data_order(X, order, structured=False):
    """
    Toma un dataset X multifeatures, devuelve el dataset que corresponde a un polinomio de orden 'orden'.
    Cada columna del dataset devuelto corresponde a X^i, con i creciente hasta orden.
    NO TIENE EN CUENTA EL ORDEN 0!
    In:
    ndarray
    Out:
    ndarray
    """

    lock = 0

    if not structured:
        m = X.shape[1]
        n = X.shape[0]

        X_new = np.empty(shape=[n, order])

        for i in range(m):
            if lock == 0:
                X_new = adapt_data_order(X[:,i].reshape(-1,1), order)
                lock += 1
            else:
                X_new = np.hstack([X_new, adapt_data_order(X[:,i].reshape(-1,1), order)])
    else:
        keys = [x for x in X.dtype.fields.keys()]
        for key in keys:
            if lock == 0:
                X_new = adapt_data_order(X[key].reshape(-1,1), order)
                lock += 1
            else:
                X_new = np.hstack([X_new, adapt_data_order(X[key].reshape(-1,1), order)])
    return X_new