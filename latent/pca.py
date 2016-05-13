"""
PCA implementation, by Atanas Mirchev

I implemented the application of PCA onto the given input in theano. I decided not to
reimplement the Eigendecomposition / SVD which numpy already offers: chances are I will get them wrong or make them
not as efficient even in theano.

One of the questions on piazza asked whether we should use numpy / scipy for the covariance computation during PCA.
Prof. Smagt responded that we should really reuse what is there, since it is already efficient enough, and I think this
is indeed the right decision. Moreover, I checked the PCA implementation at pylearn for some ideas, and they also default
to numpy for the computation of the eigen-components when it comes to PCA. Similarly, pylearn implements the actual
application of the PCA in theano.
"""

import theano
import numpy as np
import gzip
import pickle
import tarfile

from theano import tensor as T
from scipy import linalg
from itertools import product
from scipy.misc import toimage
from scipy.misc import imresize

import matplotlib.pyplot as plot

class PCA:
    def __init__(self, dim_size, n_latent=None, variance_preserve=0.9):
        """
        Define a PCA model
        :param dim_size: dimensionality of the data points to be transformed
        :param n_latent: how many dimensions to map to. If specified, variance_preserve does not play a row.
        :param variance_preserve: only valid if n_latent is unset. Preserves that much of the variance when the
                eigenvalues are filtered.
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
        """
        Trains the model (finding the eigencomponents for the transformation)
        :param X:
        :return:
        """
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
        self.V.set_value(V[:, :cutoff])
        self.eigenvals.set_value(eigenvals[:cutoff])

    def _compute_cutoff(self, eigenvals):
        """
        Computes a 90% cutoff index from a list of eigenvalues.

        :param eigenvals: a list of eigenvalues
        :return:
        """
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
        """
        Compute the eigenvectos and eigenvalues of X through SVD.

        :param X: data point matrix which we will compute the eigenvectors and eigenvalues of
        :return: (eigenvalues, eigenvectors)
        """
        U, s, V_T = linalg.svd(X, full_matrices=False)

        return s ** 2, V_T.T



def get_mnist():
    print("mnist.pkl.gz must be directly in this directory, besides pca.py")
    dataset = 'mnist.pkl.gz'
    import os

    # Download the MNIST dataset if it is not present
    if not os.path.isfile(dataset):
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading MNIST from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

def get_cifar():
    print("The extracted cifar content must be directly in subfolder cifar-10-batches-py in this directory, besides pca.py")
    dataset = 'cifar-10-python.tar.gz'

    import os
    # Download the MNIST dataset if it is not present
    if not os.path.isfile(dataset):
        from six.moves import urllib
        origin = (
            'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        )
        print('Downloading CIFAR-10 from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)
    tar = tarfile.open('cifar-10-python.tar.gz', 'r:gz')
    for item in tar:
        tar.extract(item)
    print("Done extracting CIFAR-10 content")

def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

if __name__ == '__main__':
    """
    Load data
    """
    get_mnist()
    get_cifar()
    print("Loading data ...")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        mnist_train_set, mnist_valid_set, mnist_test_set = pickle.load(f)
        mnist_train_x, mnist_train_y = mnist_train_set
        mnist_valid_x, mnist_valid_y = mnist_valid_set
        mnist_test_x, mnist_test_y = mnist_test_set

    # As already suggested, I first convert the CIFAR images to grayscale
    def prep_image(arr):
        return np.asarray(toimage(arr.reshape(3, 32, 32)).convert('L')).flatten()

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
    print("{} {}".format(cifar_x.shape, cifar_y.shape))

    """
    Compute PCA, create plots
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

        plot.tight_layout()
        plot.savefig(filename)

    print("The scatterplots will be saved in the current directory")
    scatter_plot(cifar_x, cifar_y, 'scatterplotCIFAR.png')
    scatter_plot(np.concatenate([mnist_train_x, mnist_valid_x, mnist_test_x]), np.concatenate([mnist_train_y, mnist_valid_y, mnist_test_y]), 'scatterplotMNIST.png')
