"""
PCA implementation

The main ineficciency of PCA does not hide behind the conputation of the covariance matrix. It is only computed once after all,
and it is unlikely that the feature vectors will be that big.

The actual optimization is required during the transformation of the initial representations to the mapping into the reduced space.
Therefore we need TheanoVariables first and foremost for the transformation matrix.
"""

import theano
import numpy as np

from theano import tensor as T
from scipy import linalg
from itertools import product

import matplotlib.pyplot as plot
from scipy.misc import toimage
from scipy.misc import imresize

"""
TODO: do this blockwise
"""


class PCA:
    def __init__(self, dim_size, n_latent=None, variance_preserve=0.9):
        """
        TODO: here, define all the TheanoVariables that we will use for the tranformation of X.
        V : the tranformation matrix V
        eigenvals: the list of the eigenvalues
        """

        if n_latent is not None:
            self.n_latent = n_latent;
        self.variance_preserve = variance_preserve
        self.V = theano.shared(np.zeros([dim_size, dim_size]))
        self.eigenvals = theano.shared(np.zeros(dim_size))

        self._X = T.dmatrix('X')

        self._transform = theano.function(inputs=[self._X], outputs=T.dot(self._X, self.V))

    def apply(self, X):
        print("Transforming dataset of size %i" % X.shape[0])
        mean = X.mean(axis=0)
        X_centered = X - mean
        transformed = self._transform(X_centered)
        return transformed

    def train(self, X):
        print("Training PCA model ...")
        mean = X.mean(axis=0)
        X_centered = X - mean

        # compute covariance matrix, do PCA
        eigenvals, V = self._compute_eigen(X_centered)

        # select remaining components
        cutoff = 0
        if self.n_latent is not None:
            cutoff = min(self.n_latent, len(eigenvals))
        else:
            cutoff = self._compute_cutoff(eigenvals)
        self.V.set_value(V[:, :self.n_latent])
        self.eigenvals.set_value(eigenvals[:self.n_latent])

    def _compute_cutoff(self, eigenvals):
        cutoff = 0
        total = eigenvals.sum()
        sum = 0
        for eigenval in eigenvals:
            cutoff = cutoff + 1
            sum = sum + eigenval
            if (sum / total) >= self.variance_preserve:
                break;
        return cutoff

    def _compute_eigen(self, X):
        print("Dims {}").format(X.shape)
        U, s, V_T = linalg.svd(X, full_matrices=False)

        return s ** 2, V_T.T


"""
Load data
"""
import gzip
import pickle

print("Loading data ...")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict



def prep_image(arr):
    return imresize(toimage(arr.reshape(3, 32, 32)).convert('L'), size=0.375).flatten()

cifar_x = []
cifar_y = []

for i in range(5):
    print("Loading cifar batch %i" % (i+1))
    cifar_batch = unpickle('cifar-10-batches-py/data_batch_%i' % (i+1))

    # convert to grayscale
    x = cifar_batch['data']
    x = np.asarray(map(lambda im: prep_image(im), x))
    y = np.asarray(cifar_batch['labels'])
    cifar_x.append(x)
    cifar_y.append(y)

cifar_x = np.concatenate(cifar_x)
cifar_y = np.concatenate(cifar_y)
print("{} {}").format(cifar_x.shape, cifar_y.shape)

"""
Compute PCA
"""


def scatter_plot(train_x, train_y, filename):
    pca = PCA(train_x.shape[1], n_latent=2)

    fig, plots = plot.subplots(10, 10)
    fig.set_size_inches(50, 50)
    plot.prism()
    for i, j in product(xrange(10), repeat=2):
        if i > j:
            continue
        X_ = train_x[(train_y == i) + (train_y == j)]
        y_ = train_y[(train_y == i) + (train_y == j)]


        # train on each pair of vars separately
        pca.train(X_)
        X_transformed = pca.apply(X_)

        plots[i, j].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
        plots[i, j].set_xticks(())
        plots[i, j].set_yticks(())

        plots[j, i].scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
        plots[j, i].set_xticks(())
        plots[j, i].set_yticks(())
        if i == 0:
            plots[i, j].set_title(j)
            plots[j, i].set_ylabel(j)

            # plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y_)
    plot.tight_layout()
    plot.savefig(filename)


scatter_plot(cifar_x, cifar_y, 'scatterplotCIFAR.png')
scatter_plot(np.concatenate([train_x, valid_x, test_x]), np.concatenate([train_y, valid_y, test_y]), 'scatterplotMNIST.png')
