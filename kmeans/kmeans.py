import numpy as np
import theano

from theano import tensor as T
from numpy import random
from scipy.misc import toimage
from scipy.misc import imresize
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self):
        pass

    def train(self, X, n_clusters, max_iterations=20):
        X = self._preprocess(X)
        # D x K matrix
        self.D = theano.shared(random.randn(X.shape[1], n_clusters))
        # N x K - for each point, say which cluster it's in
        self.S = theano.shared(np.zeros([X.shape[0], n_clusters]))
        # N x D matrix of points
        self._X = theano.shared(X, borrow=True)

        update_D = theano.function(inputs=[], updates=[(self.D, T.dot(self._X.T, self.S) + self.D)])

        assign = T.argmax(T.dot(self._X, self.D), axis=1)

        update_S = theano.function(inputs=[],
                                   # replace the assigned clusters with 1
                                   updates=[(self.S, T.set_subtensor(T.zeros(self.S.shape)[T.arange(self.S.shape[0]), assign], 1))])

        for i in range(max_iterations):
            print("iteration {}").format(i)
            update_D()
            update_S()
        return self.D.get_value()
    def _preprocess(self, X):
        # standardize
        # TODO: figure out how to do this better
        X = (X - np.mean(X)) / np.std(X)

        # whiten
        # TODO: find a way to do this in theano
        return X


"""
Load the cifar dataset
"""
def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

cifar_batch_1 = unpickle('cifar-10-batches-py/data_batch_1')
cifar_x = cifar_batch_1['data']
cifar_y = np.asarray(cifar_batch_1['labels'])

# cifar_x = map(lambda image: imresize(image, size=0.375), cifar_x)

# plt.imshow()
def prep_image(arr):
    return imresize(toimage(arr.reshape(3, 32, 32)).convert('L'), size=0.375).flatten()

cifar_x = np.asarray(map(lambda im: prep_image(im), cifar_x))

kmeans = KMeans()
centroids = kmeans.train(cifar_x, n_clusters=500)

for i in range(500):
    plt.imshow(centroids[:, i].reshape((12, 12)), cmap=cm.Greys_r)
    plt.show()