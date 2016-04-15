import climin
import itertools
import theano
import numpy as np
import pickle
import gzip

from theano import tensor as T
from theano import shared
from numpy import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt

"""
Implementation of a fully connected feed forward neural network
based on Theano
"""


class NeuralLayer:
    def __init__(self, inputs, in_dim, out_dim, activation=None):
        """
        A layer is defined by the number (dim) of the input units, the number of the output units
        and the activation function used by those units.

        The parameters are stored into a weight matrix W (matrix because many to many relation)
        and a bias vector b, which has length equal to out_dim.
        :param inputs: a TheanoVector of the inputs to this layer
        :param in_dim: the number of input units
        :param out_dim: the number of output units
        :param activation: the activation function used (e.g. Theano.tanh)
        """
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        print("Creating layer with {} and {} with activation {}").format(in_dim, out_dim, activation)

        # define the weights for this layer
        self.W = shared(0.1 * random.randn(in_dim, out_dim), name='W', borrow=True)
        self.b = shared(np.zeros(out_dim), name="b", borrow=True)

        net = T.dot(inputs, self.W) + self.b
        if activation is None:
            self.outputs = net
        else:
            self.outputs = activation(net)


class NeuralNetAutoencoder:
    MAX_EPOCHS = 100.0

    def __init__(self, network_spec, L1_coef=0.0001):
        """
        :param network_spec: A dictionary, which specifies only the hidden layers.
        Has the following form:

        {
            "in_dim": <int> // number of input units
            "out_dim": <int> // number of output units
            "layer_specs": [ { "in_dim": <int>, "out_dim": <int>, "activation": <e.g. Theano.tanh> }, ...]
        }

        In and out in the layer_spec specify the number of inputs and outputs.
        """

        self.network_spec = network_spec
        self.layers = []

        # create a theano variable for the input
        self.net_input = T.dmatrix('init_input')

        # construct all the hidden layers
        prev_out = self.net_input
        for layer_spec in network_spec["layer_specs"]:
            next_layer = NeuralLayer(prev_out, layer_spec["in_dim"], layer_spec["out_dim"], layer_spec["activation"])
            prev_out = next_layer.outputs
            self.layers.append(next_layer)

        # Create a softmax classification output layer
        self.output_layer = NeuralLayer(prev_out, self.layers[-1].out_dim, network_spec["out_dim"])
        self.layers.append(self.output_layer)

        # Define a L1 sparsity constraint for the parameters
        L1 = T.mean([abs(layer.W).mean() for layer in self.layers])
        # now we form the MSE loss. Mean so that it is not dependent on the number of samples used
        self.loss = T.mean((self.net_input - self.output_layer.outputs)**2) + L1_coef*L1

        self._error_rate = theano.function(inputs=[self.net_input], outputs=self.loss)

        # Build a list of gradient tuples per layer: First gradient is grad_W (matrix) and
        # second gradient is grad_b (bias)
        grads = [(T.grad(self.loss, layer.W), T.grad(self.loss, layer.b)) for layer in self.layers]

        # define the gradient descent updates
        updates = []
        for (layer, (grad_W, grad_b)) in zip(self.layers, grads):
            updates.append((layer.W, layer.W - 0.1 * grad_W))
            updates.append((layer.b, layer.b - 0.1 * grad_b))

        self.train_step = theano.function(
            inputs=[self.net_input],
            outputs=self.loss,
            updates=updates
        )

    def error_rate(self, X):
        return self._error_rate(X)

    def train(self, train_x, valid_x, test_x, batch_size=20,
              stop_threshold=10):
        """
        Trains the logistic regression model with stochastic gradient descent.
        :param train_x: the training feature vectors
        :param train_y: the training labels
        :param valid_x: the validation feature vectors
        :param valid_y: the validation labels
        :param test_x:
        :param test_y:
        :param batch_size: size of the stochastic descent mini batches
        :param stop_threshold: how many epochs to wait for after the error stagnates
        """
        print("Training ...")
        # now let's define the minibatch learning
        # stop early when validation error starts to increase
        final_params = [(layer.W.get_value(), layer.b.get_value()) for layer in self.layers]
        best_err = np.inf
        train_errors = []
        valid_errors = []
        test_errors = []
        batch_err = 0
        no_better_since = 0

        counter = 0
        epoch_counter = 0
        validation_threshold = train_x.shape[0] // batch_size
        while epoch_counter < NeuralNetAutoencoder.MAX_EPOCHS:
            mini_batch_x = train_x[counter * batch_size: (counter + 1) * batch_size]
            batch_err = batch_err + self.train_step(mini_batch_x)
            counter += 1
            # one training epoch has passed
            if counter == validation_threshold:
                if no_better_since >= stop_threshold:
                    break

                epoch_counter += 1
                print("Epoch {} passed. Validating model:").format(epoch_counter)
                new_err = self.error_rate(valid_x)
                if new_err < best_err:
                    no_better_since = 0
                    best_err = new_err
                    # save the best model
                    final_params = [(layer.W.get_value(), layer.b.get_value()) for layer in self.layers]
                else:
                    no_better_since += 1
                print("Error rate: {}").format(new_err)

                # append errors to plotting data

                valid_errors.append(new_err)
                train_errors.append(batch_err / validation_threshold)
                test_errors.append(self.error_rate(test_x))

                counter = 0
                batch_err = 0

        print("Training done ...")

        plt.ylabel('error rate')
        valid_legend, = plt.plot(valid_errors, label="validation")
        train_legend, = plt.plot(train_errors, label="training")
        test_legend, = plt.plot(test_errors, label="test")
        plt.legend(handles=[valid_legend, train_legend, test_legend])
        plt.show()

        # set W and b of each layer to the values before the validation error started rising
        for (layer, (final_W, final_b)) in zip(self.layers, final_params):
            layer.W.set_value(final_W)
            layer.b.set_value(final_b)

        print("Final error rate: {}").format(self.error_rate(valid_x))


"""
Load data
"""
print("Loading data ...")
with gzip.open('mnist.pkl.gz', 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)
    train_x, train_y = train_set
    valid_x, valid_y = valid_set
    test_x, test_y = test_set

"""
Init
"""
nnet = NeuralNetAutoencoder({
    "out_dim": train_x.shape[1],
    "in_dim": train_x.shape[1],
    "layer_specs": [
        {"in_dim": train_x.shape[1], "out_dim": 300, "activation": T.tanh}
    ]
})

"""
Train autoencoder
"""
nnet.train(train_x, valid_x, test_x)
