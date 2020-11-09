import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler



def pca_scikit(x, components):
    pca = PCA(n_components=components)
    x_std = StandardScaler(with_std=False).fit_transform(x)
    return pca.fit_transform(x_std)


def pca (x, components):
    x_mean = (x - np.mean(x, axis=0))
    x_cov = np.cov(x_mean.T)
    (w, v) = np.linalg.eig(x_cov)
    arg = w.argsort()[::-1]
    w = w[arg]
    v_ord = v[:, arg]
    # pca = np.dot(x_mean,v_ord[:,:2])
    # np.matmul es más explicito
    pca = np.matmul(x_mean, v_ord[:, :components])
    return pca

# TODO La primera columna arroja signos invertidos en pca vs pca_scikit

