import climin
import theano
import numpy as np
import pickle
import gzip
import timeit
import PIL.Image as Image

from theano import tensor as T
from theano import shared
from numpy import random
from climin import util as cli_util

import matplotlib.pyplot as plt

"""
Implementation of a fully connected feed forward neural network based on Theano.
I implemented the neural network myself. Again, I tried to figure out if there was something that might come as an improvement
from the Theano tutorial. I really liked the idea of modelling the network layer-wise, with one matrix representing the weights of
each layer, and that's how I did it too.

The network below is not restricted to 2 layes, since it uses a "network specification" parameter that describes it layers.
The only hardcoded thing is the SoftMax layer, which is my last layer. This one cannot be changed without tinkering with the code a little
bit.

I reused the exact same idea as in the Logistic regression for the optimization with climin: I overwrite the protected variable
_train with a function for the respective optimizer, either mine (by default) or one of the climin ones. It works nicely.
Also, on my machine my network actually runs faster than the theano tutorial one, might be a coincidence.
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
        self.W = shared(np.asarray(0.1 * random.randn(in_dim, out_dim), dtype=theano.config.floatX), name='W',
                        borrow=True)
        self.b = shared(np.zeros(out_dim, dtype=theano.config.floatX), name="b", borrow=True)

        net = T.dot(inputs, self.W) + self.b
        self.outputs = activation(net)


class NeuralNetClassifier:
    def __init__(self, network_spec, batch_size=600, max_epochs=150, learning_rate=0.1):
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

        """
        Public variables
        """
        self.network_spec = network_spec
        self.batch_size = batch_size
        self.layers = []
        self.max_epochs = shared(max_epochs, name="max_epochs", borrow=True)
        self.learning_rate = shared(learning_rate, name="learning_rate", borrow=True)

        """
        Private theano configuration
        """
        self._minibatch_index = shared(0)
        self._current_epoch = shared(0)
        self._inc_epoch = theano.function(inputs=[], updates=[(self._current_epoch, self._current_epoch + 1)])
        self._inc_minibatch = theano.function(inputs=[], updates=[(self._minibatch_index, self._minibatch_index + 1)])

        # create a theano variable for the input
        self._net_input = T.matrix('init_input')
        self._targets = T.ivector('targets')

        # construct all the hidden layers
        prev_out = self._net_input
        for layer_spec in network_spec["layer_specs"]:
            next_layer = NeuralLayer(prev_out, layer_spec["in_dim"], layer_spec["out_dim"], layer_spec["activation"])
            prev_out = next_layer.outputs
            self.layers.append(next_layer)

        # Create a softmax classification output layer
        self.output_layer = NeuralLayer(prev_out, self.layers[-1].out_dim, network_spec["out_dim"], T.nnet.softmax)
        self.layers.append(self.output_layer)

        # Define the cost function
        # NLL is sum of only the log(prob) for the CORRECT class per instance
        # we can select them by indexing the probs matrix (like in logregression)
        filtered = self.output_layer.outputs[
            T.arange(self._targets.shape[
                         0]), self._targets]  # broadcasted indexes, the result is a vector of prob. for each instance

        # L1 Regularization
        L1 = T.mean([abs(layer.W).mean() for layer in self.layers])

        # now we form the NLL. Mean so that it is not dependent on the number of samples used
        self._loss = -T.mean(T.log(filtered)) + L1

        # define the error rate
        predictions = T.argmax(self.output_layer.outputs, axis=1)  # map the softmax probabilities to a digit (0 to 9)
        self._missclass_rate = T.neq(predictions, self._targets).mean()

        # Build a list of gradient tuples per layer: First gradient is grad_W (matrix) and
        # second gradient is grad_b (bias)
        self._grads = [(T.grad(self._loss, layer.W), T.grad(self._loss, layer.b)) for layer in self.layers]
        grads_climin = []
        for grad_W, grad_b in self._grads:
            grads_climin.append(grad_W)
            grads_climin.append(grad_b)

        """
        Callable functions (APPLY nodes) to access some of the internal states
        """
        # define the objective function
        self._objective = theano.function(inputs=[self._net_input, self._targets], outputs=self._loss)
        self._error_rate = theano.function(inputs=[self._net_input, self._targets], outputs=self._missclass_rate)
        self._climin_grads = theano.function(inputs=[self._net_input, self._targets],
                                             outputs=grads_climin)

    def error_rate(self, X, y):
        return self._error_rate(X, y)

    """
    TODO: define the momentum
    """

    def use_default(self, X, y):
        # define a decaying learning rate
        new_rate = self.learning_rate * (
            (self.max_epochs - self._current_epoch) / T.cast(self.max_epochs, T.config.floatX))

        # define the update to be used in the momentum calculation
        momentums_W = [shared(np.zeros(layer.W.get_value().shape)) for layer in self.layers]
        momentums_b = [shared(np.zeros(layer.b.get_value().shape)) for layer in self.layers]


        # create share variables for the data to operate upon
        shared_X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
        float_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)
        shared_y = T.cast(float_y, 'int32')

        # define the gradient descent updates
        updates = []
        for (layer, (grad_W, grad_b), momentum_W, momentum_b) in zip(self.layers, self._grads, momentums_W, momentums_b):
            updates.append((layer.W, layer.W - (new_rate*grad_W + (1 - new_rate)*momentum_W)))
            updates.append((layer.b, layer.b - (new_rate*grad_b + (1 - new_rate)*momentum_b)))
            updates.append((momentum_W, new_rate*grad_W + (1 - new_rate)*momentum_W))
            updates.append((momentum_b, new_rate*grad_b + (1 - new_rate)*momentum_b))

        self._train = theano.function(
            inputs=[],
            outputs=self._missclass_rate,
            updates=updates,
            givens={self._net_input: shared_X[self._minibatch_index * self.batch_size:(
                                                                                          self._minibatch_index + 1) * self.batch_size],
                    self._targets: shared_y[self._minibatch_index * self.batch_size:(
                                                                                        self._minibatch_index + 1) * self.batch_size]}
        )

    def use_sgd(self, X, y):
        params, args = self._climin_setup(X, y)
        _, grad_func = self._climin_funcs()
        # Standard gradient descent: 7.44 % validation, 7.74 % test at epcch 150
        opt = climin.GradientDescent(params, grad_func, args=args)

        # # Gradient descent with 0.9 momentum: 7.01 % validation, 7.30 % test set at epoch 150
        # opt = climin.GradientDescent(params, grad_func, momentum=0.9, args=args)

        # # Gradient descent with nesterov momentum: 6.98 % validation, 7.30 % test set at epoch 150
        # opt = climin.GradientDescent(params, grad_func, momentum=0.9, momentum_type="nesterov", args=args)

        self._train = self._climin_train_step(opt)

    def use_rprop(self, X, y):
        params, args = self._climin_setup(X, y)
        _, grad_func = self._climin_funcs()

        # Rprop: http://arxiv.org/pdf/1509.04612.pdf: 7.00 % validation at epoch 30, but then went to hell
        opt = climin.Rprop(params, grad_func, step_shrink=0.1, step_grow=1.01, max_step=5, min_step=0.001, args=args)

        self._train = self._climin_train_step(opt)

    def use_rmsprop(self, X, y):
        params, args = self._climin_setup(X, y)
        _, grad_func = self._climin_funcs()

        # # Rmsprop with a small learning rate and a medium momentum: 6.75% validation set, 7.1 % test set
        opt = climin.RmsProp(params, grad_func, step_rate=0.001, momentum=0.5, args=args)

        self._train = self._climin_train_step(opt)

    def use_adadelta(self, X, y):
        params, args = self._climin_setup(X, y)
        _, grad_func = self._climin_funcs()

        # Adadelta: large step rate (1), no momentum - 7% validation set, 7.5 % test set error
        opt = climin.Adadelta(params, grad_func, args=args)

        # Adadelta with smaller learning rate (0.1) and medium momentum: same as above
        # Adadelta: big learning rate, momentum: same
        # opt = climin.Adadelta(params, grad_func, step_rate=1, momentum=0.5, args = args)

        self._train = self._climin_train_step(opt)

    def use_adam(self, X, y):
        params, args = self._climin_setup(X, y)
        _, grad_func = self._climin_funcs()

        # Adam: the convergence looks really stable here, with a small learning rate: 6.93 % validation, 7.22 % test set
        # opt = climin.Adam(params, grad_func, args=args)

        # Adam with a slightly bigger learning rate: 6.75 % validation, 7.1 % test
        # convergence didn't really change with or without a momentum
        opt = climin.Adam(params, grad_func, step_rate=0.003, args=args)

        self._train = self._climin_train_step(opt)

    def use_lbfgs(self, X, y):
        params, args = self._climin_setup(X, y)
        obj_func, grad_func = self._climin_funcs()

        # Bfgs failed for me. The error was: 'Bfgs' object has no attribute 'logfunc' at line 169, and indeed there isn't such an attr in the file.

        # # Note: as expected, the pseudo-Neuton methods take much more time
        # # LBfgs with mini-batches: 7.64 % validation, 7.76 % test, epoch 150
        # # LBfgs with bigger batch data: 7.17 % validation, 7.5 % test, epoch 150
        opt = climin.Lbfgs(params, obj_func, grad_func, args=args)

        self._train = self._climin_train_step(opt)

    def _climin_funcs(self):
        """

        :return: returns two help functions for the climin optimizer
                    objective_func: callable function, returns the loss
                    grad_func: callable function, returns the flattened gradient
        """

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
            grads = model._climin_grads(X, y)
            # flatten all the parameters. The order of the layers is preserved, W follows before b always
            return np.concatenate(map(lambda grad: grad.flatten(), grads))

        def objective_func(parameters, X, y, model):
            unpack_params(parameters, model)
            return model._objective(X, y)

        return objective_func, grad_func

    def _climin_setup(self, train_x, train_y):
        """
        The functions sets up the climin minibatches protocol given the data to train on.

        :param train_x: the data to train on
        :param train_y: the labels for the data
        :return: params: a flat placeholder for the model parameters
                 args: a list of arguments passed to the climin optimizer at each training step
        """
        # initialize a flat parameters placeholder for climin
        # randomize the parameters, but keep variance small enough (0.01) to promote faster learning
        size = 0
        for layer in self.layers:
            size = size + layer.in_dim * layer.out_dim + layer.out_dim
        params = 0.01 * random.randn(size)

        # define the additional arguments for each train iteration
        minibatches = cli_util.iter_minibatches([train_x, train_y], self.batch_size, [0, 0])
        # as climin arguments, pass the mini_batch_x, mini_batch_y and the model itself
        args = (([minibatch[0], minibatch[1], self], {}) for minibatch in minibatches)

        return params, args

    def _climin_train_step(self, opt):
        model = self
        iterator = iter(opt)

        def train():
            next = iterator.next()
            # extract the minibatch and compute it's error rate
            return model.error_rate(next['args'][0], next['args'][1])

        return train

    def train(self, train_x, train_y, valid_x, valid_y, test_x, test_y, stop_threshold=10, optimizer=None):
        """
        Trains the logistic regression model with stochastic gradient descent.
        :param train_x: the training feature vectors
        :param train_y: the training labels
        :param valid_x: the validation feature vectors
        :param valid_y: the validation labels
        :param test_x:
        :param test_y:
        :param stop_threshold: how many epochs to wait for after the error stagnates
        :param optimizer: specifies which gradient optimizer to use
        """

        print("Selecting gradient optimizer ...")
        if optimizer == None:
            self.use_default(train_x, train_y)
        elif optimizer == "sgd":
            self.use_sgd(train_x, train_y)
        elif optimizer == "rprop":
            self.use_rprop(train_x, train_y)
        elif optimizer == "rmsprop":
            self.use_rmsprop(train_x, train_y)
        elif optimizer == "adadelta":
            self.use_adadelta(train_x, train_y)
        elif optimizer == "adam":
            self.use_adam(train_x, train_y)
        elif optimizer == "lbfgs":
            self.use_lbfgs(train_x, train_y)

        print("Training ...")
        # now let's define the early stopping when the validation error starts to increase
        final_params = [(layer.W.get_value(), layer.b.get_value()) for layer in self.layers]
        best_err = np.inf
        train_errors = []
        valid_errors = []
        test_errors = []
        batch_err = 0
        no_better_since = 0
        self._minibatch_index.set_value(0)
        self._current_epoch.set_value(0)

        start_time = timeit.default_timer()
        validation_threshold = train_x.shape[0] // self.batch_size
        while self._current_epoch.eval() < self.max_epochs.eval():
            batch_err = batch_err + self._train()
            self._inc_minibatch()

            # one training epoch has passed
            if self._minibatch_index.eval() == validation_threshold:
                if no_better_since >= stop_threshold:
                    break;
                self._inc_epoch()
                print("Epoch {} passed. Validating model:".format(self._current_epoch.eval()))

                new_err = self.error_rate(valid_x, valid_y)
                if new_err < best_err:
                    no_better_since = 0
                    best_err = new_err
                    print("New best error %f" % best_err)
                    # save the best model
                    final_params = [(layer.W.get_value(), layer.b.get_value()) for layer in self.layers]
                else:
                    no_better_since += 1
                print("Error rate: {}".format(new_err))

                # append errors to plotting data
                valid_errors.append(new_err)
                train_errors.append(batch_err / validation_threshold)
                test_errors.append(self.error_rate(test_x, test_y))

                # reset the mini batch index
                self._minibatch_index.set_value(0)
                batch_err = 0

        end_time = timeit.default_timer()
        print("Training done ... The training took %i seconds" % (end_time - start_time))

        plt.ylabel('error rate')
        valid_legend, = plt.plot(valid_errors, label="validation")
        train_legend, = plt.plot(train_errors, label="training")
        test_legend, = plt.plot(test_errors, label="test")
        plt.legend(handles=[valid_legend, train_legend, test_legend])
        # plt.savefig('relu_errors.png')
        plt.show()
        # set W and b to the values before the validation error started rising
        for (layer, (final_W, final_b)) in zip(self.layers, final_params):
            layer.W.set_value(final_W)
            layer.b.set_value(final_b)

        print("Final error rate: {}".format(self.error_rate(valid_x, valid_y)))


def scale_to_unit_interval(ndar, eps=1e-8):
    """ Scales all values in the ndarray ndar to be between 0 and 1 """
    ndar = ndar.copy()
    ndar -= ndar.min()
    ndar *= 1.0 / (ndar.max() + eps)
    return ndar


"""
Help function for visualization of receptive fields. Copied from one of the Theano tutorials.
"""


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
    """
    Transform an array with one flattened image per row, into an array in
    which images are reshaped and layed out like tiles on a floor.

    This function is useful for visualizing datasets whose rows are images,
    and also columns of matrices for transforming those rows
    (such as the first layer of a neural net).

    :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
    be 2-D ndarrays or None;
    :param X: a 2-D array in which every row is a flattened image.

    :type img_shape: tuple; (height, width)
    :param img_shape: the original shape of each image

    :type tile_shape: tuple; (rows, cols)
    :param tile_shape: the number of images to tile (rows, cols)

    :param output_pixel_vals: if output should be pixel values (i.e. int8
    values) or floats

    :param scale_rows_to_unit_interval: if the values need to be scaled before
    being plotted to [0,1] or not


    :returns: array suitable for viewing as an image.
    (See:`Image.fromarray`.)
    :rtype: a 2-d array with same dtype as X.

    """

    assert len(img_shape) == 2
    assert len(tile_shape) == 2
    assert len(tile_spacing) == 2

    # The expression below can be re-written in a more C style as
    # follows :
    #
    # out_shape    = [0,0]
    # out_shape[0] = (img_shape[0]+tile_spacing[0])*tile_shape[0] -
    #                tile_spacing[0]
    # out_shape[1] = (img_shape[1]+tile_spacing[1])*tile_shape[1] -
    #                tile_spacing[1]
    out_shape = [
        (ishp + tsp) * tshp - tsp
        for ishp, tshp, tsp in zip(img_shape, tile_shape, tile_spacing)
        ]

    if isinstance(X, tuple):
        assert len(X) == 4
        # Create an output numpy ndarray to store the image
        if output_pixel_vals:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype='uint8')
        else:
            out_array = np.zeros((out_shape[0], out_shape[1], 4),
                                 dtype=X.dtype)

        # colors default to 0, alpha defaults to 1 (opaque)
        if output_pixel_vals:
            channel_defaults = [0, 0, 0, 255]
        else:
            channel_defaults = [0., 0., 0., 1.]

        for i in range(4):
            if X[i] is None:
                # if channel is None, fill it with zeros of the correct
                # dtype
                dt = out_array.dtype
                if output_pixel_vals:
                    dt = 'uint8'
                out_array[:, :, i] = np.zeros(
                    out_shape,
                    dtype=dt
                ) + channel_defaults[i]
            else:
                # use a recurrent call to compute the channel and store it
                # in the output
                out_array[:, :, i] = tile_raster_images(
                    X[i], img_shape, tile_shape, tile_spacing,
                    scale_rows_to_unit_interval, output_pixel_vals)
        return out_array

    else:
        # if we are dealing with only one channel
        H, W = img_shape
        Hs, Ws = tile_spacing

        # generate a matrix to store the output
        dt = X.dtype
        if output_pixel_vals:
            dt = 'uint8'
        out_array = np.zeros(out_shape, dtype=dt)

        for tile_row in range(tile_shape[0]):
            for tile_col in range(tile_shape[1]):
                if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                    this_x = X[tile_row * tile_shape[1] + tile_col]
                    if scale_rows_to_unit_interval:
                        # if we should scale values to be between 0 and 1
                        # do this by calling the `scale_to_unit_interval`
                        # function
                        this_img = scale_to_unit_interval(
                            this_x.reshape(img_shape))
                    else:
                        this_img = this_x.reshape(img_shape)
                    # add the slice to the corresponding position in the
                    # output array
                    c = 1
                    if output_pixel_vals:
                        c = 255
                    out_array[
                    tile_row * (H + Hs): tile_row * (H + Hs) + H,
                    tile_col * (W + Ws): tile_col * (W + Ws) + W
                    ] = this_img * c
        return out_array


"""
Slightly modified from the Theano tutorial
"""


def get_data():
    print("mnist.pkl.gz must be directly in this directory, besides neuralnet.py")
    dataset = 'mnist.pkl.gz'
    import os

    # Download the MNIST dataset if it is not present
    data_dir, data_file = os.path.split(dataset)
    if not os.path.isfile(dataset):
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading MNIST from %s' % origin)
        urllib.request.urlretrieve(origin, dataset)

if __name__ == '__main__':
    get_data()
    """
    Load data
    """
    print("Loading data ...")
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        train_set, valid_set, test_set = pickle.load(f)
        train_x, train_y = train_set
        valid_x, valid_y = valid_set
        test_x, test_y = test_set

        # convert to types we can easily handle
        train_x = np.asarray(train_x, dtype='float32')
        valid_x = np.asarray(valid_x, dtype='float32')
        test_x = np.asarray(test_x, dtype='float32')
        train_y = np.asarray(train_y, dtype='int32')
        valid_y = np.asarray(valid_y, dtype='int32')
        test_y = np.asarray(test_y, dtype='int32')

    """
    Init
    """
    nnet = NeuralNetClassifier({
        "out_dim": 10,
        "in_dim": train_x.shape[1],
        "layer_specs": [
            {"in_dim": train_x.shape[1], "out_dim": 300, "activation": T.nnet.sigmoid}
        ]
    })

    """
    Train
    """
    # nnet.train(train_x, train_y, valid_x, valid_y, test_x, test_y)
    nnet.train(train_x, train_y, valid_x, valid_y, test_x, test_y, optimizer='rmsprop')

    """
    Perform on test data
    """
    test_error = nnet.error_rate(test_x, test_y)
    print("Achieved error rate on test set: {}".format(test_error))

    """
    Receptive fields
    """
    image = Image.fromarray(tile_raster_images(
        X=nnet.layers[0].W.eval().T[:100],
        img_shape=(28, 28), tile_shape=(10, 10),
        tile_spacing=(1, 1)))
    image.show()
    # image.save('relu_receptive.png')
