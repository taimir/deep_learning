#!/usr/bin/env python

'''
A simple Python wrapper for the bh_tsne binary that makes it easier to use it
for TSV files in a pipeline without any shell script trickery.

Note: The script does some minimal sanity checking of the input, but don't
    expect it to cover all cases. After all, it is a just a wrapper.

Example:

    > echo -e '1.0\t0.0\n0.0\t1.0' | ./bhtsne.py -d 2 -p 0.1
    -2458.83181442  -6525.87718385
    2458.83181442   6525.87718385

The output will not be normalised, maybe the below one-liner is of interest?:

    python -c 'import numpy; d = numpy.loadtxt("/dev/stdin");
        d -= d.min(axis=0); d /= d.max(axis=0);
        numpy.savetxt("/dev/stdout", d, fmt='%.8f', delimiter="\t")'

Author:     Pontus Stenetorp    <pontus stenetorp se>
Version:    2013-01-22
'''

# Copyright (c) 2013, Pontus Stenetorp <pontus stenetorp se>
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

from argparse import ArgumentParser, FileType
from os.path import abspath, dirname, isfile, join as path_join
from shutil import rmtree
from struct import calcsize, pack, unpack
from subprocess import Popen
from sys import stderr, stdin, stdout
from tempfile import mkdtemp
from scipy.misc import toimage

import tarfile
import gzip
import pickle
import matplotlib.pyplot as plot
import numpy as np


### Constants
BH_TSNE_BIN_PATH = path_join(dirname(__file__), 'bh_tsne.exe')
assert isfile(BH_TSNE_BIN_PATH), ('Unable to find the bh_tsne binary in the '
                                  'same directory as this script, have you forgotten to compile it?: {}'
                                  ).format(BH_TSNE_BIN_PATH)
# Default hyper-parameter values from van der Maaten (2013)
DEFAULT_NO_DIMS = 2
DEFAULT_PERPLEXITY = 30.0
DEFAULT_THETA = 0.5
EMPTY_SEED = -1


###


class TmpDir:
    def __enter__(self):
        self._tmp_dir_path = mkdtemp()
        return self._tmp_dir_path

    def __exit__(self, type, value, traceback):
        rmtree(self._tmp_dir_path)


def _read_unpack(fmt, fh):
    return unpack(fmt, fh.read(calcsize(fmt)))


def bh_tsne(samples, no_dims=DEFAULT_NO_DIMS, perplexity=DEFAULT_PERPLEXITY, theta=DEFAULT_THETA, randseed=EMPTY_SEED,
            verbose=False):
    # Assume that the dimensionality of the first sample is representative for
    #   the whole batch
    sample_dim = len(samples[0])
    sample_count = len(samples)

    # bh_tsne works with fixed input and output paths, give it a temporary
    #   directory to work in so we don't clutter the filesystem
    with TmpDir() as tmp_dir_path:
        # Note: The binary format used by bh_tsne is roughly the same as for
        #   vanilla tsne
        with open(path_join(tmp_dir_path, 'data.dat'), 'wb') as data_file:
            # Write the bh_tsne header
            data_file.write(pack('iiddi', sample_count, sample_dim, theta, perplexity, no_dims))
            # Then write the data
            for sample in samples:
                data_file.write(pack('{}d'.format(len(sample)), *sample))
            # Write random seed if specified
            if randseed != EMPTY_SEED:
                data_file.write(pack('i', randseed))

            bh_tsne_p = Popen((abspath(BH_TSNE_BIN_PATH),), cwd=tmp_dir_path,
                              # bh_tsne is very noisy on stdout, tell it to use stderr
                              #   if it is to print any output
                              stdout=stderr if verbose else None)
            bh_tsne_p.wait()
            assert not bh_tsne_p.returncode, ('ERROR: Call to bh_tsne exited '
                                              'with a non-zero return code exit status, please ' +
                                              ('enable verbose mode and ' if not verbose else '') +
                                              'refer to the bh_tsne output for further details')

            # Read and pass on the results
            with open(path_join(tmp_dir_path, 'result.dat'), 'rb') as output_file:
                print(output_file)
                # The first two integers are just the number of samples and the
                #   dimensionality
                result_samples, result_dims = _read_unpack('ii', output_file)
                # Collect the results, but they may be out of order
                results = [_read_unpack('{}d'.format(result_dims), output_file)
                           for _ in xrange(result_samples)]
                # Now collect the landmark data so that we can return the data in
                #   the order it arrived
                results = [(_read_unpack('i', output_file), e) for e in results]
                # Put the results in order and yield it
                results.sort()
                for _, result in results:
                    yield result
                    # The last piece of data is the cost for each sample, we ignore it
                    # _read_unpack('{}d'.format(sample_count), output_file)

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

    """
    I took some of the sklearn example datasets when evaluating tsne
    """
    # from sklearn import datasets
    # diabetes = datasets.load_diabetes()
    # X, y = diabetes.data, diabetes.target
    # faces = datasets.fetch_olivetti_faces()
    # X, y = faces.data, faces.target
    # covtype = datasets.fetch_covtype()
    # X, y = covtype.data, covtype.target

    def create_embedding(data_x, data_y, file):
        print("Creating an embedding with t-SNE. Sit back, this might take a while ...")
        plot_x = []
        plot_y = []
        for mapped_point in bh_tsne(data_x, verbose=True):
            plot_x.append(mapped_point[0])
            plot_y.append(mapped_point[1])
        plot.scatter(plot_x, plot_y, 20, data_y);
        plot.savefig(file)
        plot.show();


    # create_embedding(cifar_x, cifar_y, 'cifar_embedding.png')
    create_embedding(mnist_train_x, mnist_train_y, 'figure5_reproduction.png')
    # create_embedding(X[:10000], y[:10000], 'covtype_embedding.png')
