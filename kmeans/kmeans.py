import numpy as np
import theano

from theano import tensor as T
from numpy import random
from scipy import linalg
from scipy.misc import toimage
from scipy.misc import imresize

import matplotlib.cm as cm
import matplotlib.pyplot as plt


class KMeans:
    def __init__(self):
        pass

    def train(self, X, n_clusters, max_iterations=10):
        print("Training ...")
        X = self._preprocess(X)
        # K x D matrix
        self._D = theano.shared(np.asarray(0.1 * random.randn(n_clusters, X.shape[1]), dtype=theano.config.floatX))

        # self._D_normalized = self._D / self._D.norm(2, axis=1).reshape((self._D.shape[0], 1))
        # N x K - for each point, say which cluster it's in
        self._S = theano.shared(np.zeros([X.shape[0], n_clusters], dtype=theano.config.floatX))
        # N x D matrix of points
        self._X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)

        cluster_instances = self._S.sum(axis=0, keepdims=True)
        next_D = T.dot(self._S.T, self._X) / T.transpose(cluster_instances)
        # take only those cluster which got assigned an instance,
        # set the rest to 0
        cleaned = T.set_subtensor(next_D[T.eq(cluster_instances, 0), :], 0)
        update_D = theano.function(inputs=[], updates=[(self._D, cleaned)])

        self.cleaned = theano.function(inputs=[], outputs=cleaned)

        # maximize this term. It comes from rewriting the cluster distances (d - x)^T(d - x)
        # that we need to minimize d^2 - 2d^Tx + x^2, or in other words maximize d^Tx
        xd_prod = T.dot(self._D, self._X.T)

        # for each point, find the cluster which is maximizer
        cluster_assignments = T.argmax(xd_prod, axis=0)
        # also take the actual maximizing vals
        max_vals = T.max(xd_prod, axis=0)

        update_S = theano.function(inputs=[],
                                   # replace the assigned clusters with 1
                                   updates=[(self._S, T.set_subtensor(
                                       T.zeros(self._S.shape)[T.arange(self._S.shape[0]), cluster_assignments],
                                       1))])

        self.get_D = theano.function(inputs=[], outputs=self._D)
        self.get_S = theano.function(inputs=[], outputs=self._S)
        for i in range(max_iterations):
            print("iteration {}".format(i))
            update_S()
            update_D()

            # plt.plot(X[:, 0], X[:, 1], 'x')
            # plt.scatter(self.get_D()[:, 0], self.get_D()[:, 1], s=50, c=[0.7, 0, 0], zorder=10)
            # plt.show()

        # assigned = np.sum(self.S.get_value(), axis=0)

        # print("Non empty clusters: {} ").format(assigned)

        return self.get_D()

    def _preprocess(self, X):
        print("Preprocessing input ...")
        # normalize the brightness and contrast of each single image we pass to k-Means
        X = (X - np.mean(X, axis=1).reshape(X.shape[0], 1)) / np.sqrt(np.var(X, axis=1) + 10).reshape(X.shape[0], 1)

        # TODO: whiten
        # # whiten
        e_zca = 0.1
        U, s, V_T = linalg.svd(X, full_matrices=False)

        D_noise = np.diag(1 / np.sqrt(s ** 2 + e_zca))
        factor = np.dot(np.dot(V_T.T, D_noise), V_T)
        return np.dot(X, factor)


"""
Load the cifar dataset
"""


def unpickle(file):
    import cPickle
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict


# cifar_x = map(lambda image: imresize(image, size=0.375), cifar_x)

# plt.imshow()
def prep_image(arr):
    return imresize(toimage(arr.reshape(3, 32, 32)).convert('L'), size=0.375).flatten()


cifar_x = []
cifar_y = []

for i in range(5):
    print("Loading cifar batch %i" % (i + 1))
    cifar_batch = unpickle('cifar-10-batches-py/data_batch_%i' % (i + 1))

    # convert to grayscale
    x = cifar_batch['data']
    x = np.asarray(map(lambda im: prep_image(im), x))
    y = np.asarray(cifar_batch['labels'])
    cifar_x.append(x)
    cifar_y.append(y)

cifar_x = np.concatenate(cifar_x)
cifar_y = np.concatenate(cifar_y)
print("{} {}").format(cifar_x.shape, cifar_y.shape)

kmeans = KMeans()
centroids = kmeans.train(cifar_x, n_clusters=500)

print("{}".format(centroids))
for i in range(500):
    plt.imshow(centroids[i, :].reshape((12, 12)), cmap=cm.Greys_r)
    plt.show()


# mean1 = [0, 50]
# mean2 = [50, 0]
# mean3 = [-50, 1]
# cov1 = [[1, 0], [0, 10]]
# cov2 = [[1, 0], [0, 1]]
# cov3 = [[10, 0], [0, 1]]
#
# means = [mean1, mean2, mean3]
# covs = [cov1, cov2, cov3]
#
# points = np.zeros([1, 2])
# for mean, cov in zip(means, covs):
#     import matplotlib.pyplot as plt
#
#     x, y = np.random.multivariate_normal(mean, cov, 1000).T
#     points = np.concatenate((points, np.column_stack((x, y))))
# print("Points {}").format(points.shape)
#
# plt.axis('equal')
# kmeans = KMeans()
# kmeans.train(points, n_clusters=3)
# plt.plot(points[:, 0], points[:, 1], 'x')
# plt.scatter(kmeans.get_D()[:, 0], kmeans.get_D()[:, 1], s=50, c=[0.7, 0, 0], zorder=10)
print("D: {}").format(kmeans.get_D())
