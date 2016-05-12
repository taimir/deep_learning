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
    def __init__(self, inputs, in_dim, out_dim, activation):
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

        # define the weights for this layer
        self.W = shared(np.asarray(0.1 * random.randn(in_dim, out_dim), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = shared(np.zeros(out_dim, dtype=theano.config.floatX), name="b", borrow=True)

        net = T.dot(inputs, self.W) + self.b
        self.outputs = activation(net)


class NeuralNetClassifier:
    MAX_EPOCHS = 100.0

    def __init__(self, network_spec):
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
        self.output_layer = NeuralLayer(prev_out, self.layers[-1].out_dim, network_spec["out_dim"], T.nnet.softmax)
        self.layers.append(self.output_layer)

        # Define the cost function
        # first we need a placeholder for the target labels
        self.targets = T.lvector('targets')

        # NLL is sum of only the log(prob) for the CORRECT class per instance
        # we can select them by indexing the probs matrix (like in logregression)
        filtered = self.output_layer.outputs[
            T.arange(self.targets.shape[
                         0]), self.targets]  # broadcasted indexes, the result is a vector of prob. for each instance

        # now we form the NLL. Mean so that it is not dependent on the number of samples used
        self.loss = -T.log(filtered).mean()

        # define the error rate
        predictions = T.argmax(self.output_layer.outputs, axis=1)  # map the softmax probabilities to a digit (0 to 9)
        missclass_rate = T.neq(predictions, self.targets).mean()
        self._error_rate = theano.function(inputs=[self.net_input, self.targets], outputs=missclass_rate)

        # Build a list of gradient tuples per layer: First gradient is grad_W (matrix) and
        # second gradient is grad_b (bias)
        grads = [(T.grad(self.loss, layer.W), T.grad(self.loss, layer.b)) for layer in self.layers]
        grads_climin = []
        for grad_W, grad_b in grads:
            grads_climin.append(grad_W)
            grads_climin.append(grad_b)
        self._flat_grad = theano.function(inputs=[self.net_input, self.targets],
                                          outputs=grads_climin)

        # define the gradient descent updates
        updates = []
        for (layer, (grad_W, grad_b)) in zip(self.layers, grads):
            updates.append((layer.W, layer.W - 0.1 * grad_W))
            updates.append((layer.b, layer.b - 0.1 * grad_b))

        self.train_step = theano.function(
            inputs=[self.net_input, self.targets],
            outputs=missclass_rate,
            updates=updates
        )

    def error_rate(self, X, y):
        return self._error_rate(X, y)

    def train(self, train_x, train_y, valid_x, valid_y, test_x, test_y, init_learning_rate=0.9, batch_size=20,
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
        while epoch_counter < NeuralNetClassifier.MAX_EPOCHS:
            mini_batch_x = train_x[counter * batch_size: (counter + 1) * batch_size]
            mini_batch_y = train_y[counter * batch_size: (counter + 1) * batch_size]
            batch_err = batch_err + self.train_step(mini_batch_x, mini_batch_y)
            counter += 1
            # one training epoch has passed
            if counter == validation_threshold:
                if no_better_since >= stop_threshold:
                    break

                epoch_counter += 1
                print("Epoch {} passed. Validating model:").format(epoch_counter)
                new_err = self.error_rate(valid_x, valid_y)
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
                test_errors.append(self.error_rate(test_x, test_y))

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

        print("Final error rate: {}").format(self.error_rate(valid_x, valid_y))

    def train_climin(self, train_x, train_y, valid_x, valid_y, test_x, test_y):
        """
        Trains the logistic regression model using the climin optimizer
        """
        print("Training model with climin gradient descent optimization")

        def unpack_params(parameters, model):
            current = 0
            for layer in model.layers:
                W = parameters[current:current + layer.in_dim * layer.out_dim].reshape(layer.in_dim, layer.out_dim)
                b = parameters[
                    current + layer.in_dim * layer.out_dim:current + layer.in_dim * layer.out_dim + layer.out_dim]
                layer.W.set_value(W)
                layer.b.set_value(b)
                # increment index for the next layer
                current = current + layer.in_dim * layer.out_dim + layer.out_dim

        def grad_func(parameters, X, y, model):
            unpack_params(parameters, model)
            grads = model._flat_grad(X, y)
            # flatten all the parameters. The order of the layers is preserved, W follows before b always
            return np.concatenate(map(lambda grad: grad.flatten(), grads))

        # initialize a flat parameters placeholder for climin
        # randomize the parameters, but keep variance small enough (0.01) to promote faster learning
        size = 0
        for layer in self.layers:
            size = size + layer.in_dim * layer.out_dim + layer.out_dim
        params = 0.01 * random.randn(size)

        # pass self during the training to update the parameters of the neural network object itself
        # and reuse the theano definitions from the constructor
        opt = climin.GradientDescent(params, grad_func, step_rate=0.1, momentum=0.9,
                                     args=itertools.repeat(([train_x, train_y, self], {})))

        print("Initial error rate: {}").format(self.error_rate(valid_x, valid_y))
        for iter_info in opt:
            print("Epoch {}").format(iter_info['n_iter'])
            print("Error rate {}").format(self.error_rate(valid_x, valid_y))
            if iter_info['n_iter'] >= 300:
                break
        print("Final error rate: {}").format(self.error_rate(valid_x, valid_y))


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
nnet = NeuralNetClassifier({
    "out_dim": 10,
    "in_dim": train_x.shape[1],
    "layer_specs": [
        {"in_dim": train_x.shape[1], "out_dim": 300, "activation": T.tanh}
    ]
})

"""
Train
"""
nnet.train_climin(train_x, train_y, valid_x, valid_y, test_x, test_y)

"""
Perform on test data
"""
test_error = nnet.error_rate(test_x, test_y)
print("Achieved error rate on test set: {}").format(test_error)

"""
Receptive fields
"""
for i in range(10):
    plt.imshow(nnet.layers[0].W.eval()[:, i].reshape((28, 28)), cmap=cm.Greys_r)
    plt.show()
