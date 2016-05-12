import theano
import numpy as np
import pickle
import gzip
import timeit
import climin
import PIL.Image as Image

from climin import util as cli_util
from theano import tensor as T
from theano import shared

import matplotlib.pyplot as plt

"""
This is my implementation of logistic regression. I implemented the class on my own and to my own liking. Then I
had a look at the theano tutorial and compared to see what I could do better. For instance, I figured out that if I am
not careful with the dtypes of the matrices I use, the execution becomes quite a bit slower. E.g. setting the types
to theano.config.floatX made my "default" optimization below run 3 times faster ... interesting.

Here is a little overview of the implementation:
    - the model class LRegression contains all the theano variable definitions in its constructor. It prepares the model
    to be trained.
    - I have a couple of private functions of the type "_use_<some_optimization>", which are called when the training
    starts. The user can pass a string, specifying which optimizer to use. Then the internal "_train" function is overwritten
    with the training step from the specific optimizer.
    - I have my own personal implementation of Stochastic Gradient Descent, with a decaying learning rate and a momentum.
    NOTE: since my implementation is entirely written in theano, it runs faster than the climin optimizers. However, it is not
    as refined.
    - The climin optimizers I wired as well as I could to the shared variables the model contains. The idea is that whenever
    climin iterates, the internal state variables in the LRegression class get updates.

    This pretty much wraps it up. To use one of the optimizers, simply call:

    lregression.train( <data_sets>, .., optimizer='<some_specific_optimizer>')


I used stochastic gradient descent with mini batches. The reason to not use the Newton method
proposed in Bishop's book is that the length of the features per image is rather big -> computing the
hessian matrix might actually be more inefficient than doing randomized gradient descent. Looking at the performance
of LBFGS, it is clear that computing the Hessian takes quite some time.

Check out the README.md for the execution results.
"""

"""
TODO: try to do this with rotated / shifted images maybe?
"""


class LRegression:
    def __init__(self, feature_dim, output_dim, batch_size=600, max_epochs=150, learning_rate=0.1):
        """

        :param feature_dim: the length of the feature vectors
        :param output_dim: the number of classes
        :param batch_size: the size of the minibatches used
        :param max_epochs: the number of maximum epochs the model is allowed to train for
        :param learning_rate: the learning rate for the update step
        """
        print("Initializing logistic regression model")
        """
        Publicly accessible states
        """
        self.output_dim = output_dim
        self.feature_dim = feature_dim
        self.batch_size = batch_size

        # Parameter placeholders
        # sample W with smaller variance: 0.01 instead of 1
        # 0.1 * random.randn(feature_dim, output_dim)
        self.W = shared(np.zeros([feature_dim, output_dim], dtype=theano.config.floatX), name='W', borrow=True)
        self.b = shared(np.zeros(output_dim, dtype=theano.config.floatX), name="b", borrow=True)
        self.max_epochs = shared(max_epochs, name="max_epochs", borrow=True)
        self.learning_rate = shared(learning_rate, name="learning_rate", borrow=True)

        """
        Internal representation of the model. Theano variables.
        """
        # input Data placeholders
        self._X = T.matrix('X')
        self._y = T.ivector('y')
        self._minibatch_index = shared(0)
        self._current_epoch = shared(0)
        self._inc_epoch = theano.function(inputs=[], updates=[(self._current_epoch, self._current_epoch + 1)])
        self._inc_minibatch = theano.function(inputs=[], updates=[(self._minibatch_index, self._minibatch_index + 1)])

        # training loss definition
        logit = T.dot(self._X,
                      self.W) + self.b  # returns a matrix, with N rows and 10 numbers per instance (per class), b is broadcast
        probs = T.nnet.softmax(
            logit)  # rowwise softmax for each instance, we get N x 10 matrix with probs for each digit per instance
        # NLL is sum of only the log(prob) for the CORRECT class per instance
        # we can select them by indexing the probs matrix
        filtered = probs[
            T.arange(
                self._y.shape[0]), self._y]  # broadcasted indexes, the result is a vector of prob. for each instance
        # now we take the logs and sum them up (with a minus)
        self._loss = T.mean(-T.log(filtered))
        # compute the gradients
        self._grad_W = T.grad(self._loss, self.W)
        self._grad_b = T.grad(self._loss, self.b)
        self._predictions = T.argmax(probs, axis=1)  # map the softmax probabilities to a digit (0 to 9)
        self._missclass_rate = T.mean(T.neq(self._predictions, self._y))

        """
        Callable functions (APPLY nodes) to access some of the internal states
        """
        # define the objective function
        self._objective = theano.function(inputs=[self._X, self._y], outputs=self._loss)

        # used by climin
        self._flat_grad = theano.function(inputs=[self._X, self._y],
                                          outputs=[self._grad_W, self._grad_b])
        # define the error rate function
        self._error_rate = theano.function(inputs=[self._X, self._y], outputs=self._missclass_rate)

        # define the predict function. Note: it only needs the data points, no labels as inputs of course.
        self._predict = theano.function(inputs=[self._X], outputs=[self._predictions])

    def error_rate(self, X, y):
        """
        The function evaluates the model for the given X and returns the error rate with respect to the real labels y.
        :param X: the design matrix containing the data points
        :param y: the corresponding labels
        :return: the error rate when predicting on this model
        """
        return self._error_rate(X, y)

    def use_default(self, X, y):
        # define a theano function for the learning rate
        new_rate = self.learning_rate * (
        (self.max_epochs - self._current_epoch) / T.cast(self.max_epochs, T.config.floatX))

        # define the update to be used in the momentum calculation
        update_W = shared(np.zeros(self.W.get_value().shape))
        update_b = shared(np.zeros(self.b.get_value().shape))

        # create share variables for the data to operate upon
        shared_X = theano.shared(np.asarray(X, dtype=theano.config.floatX), borrow=True)
        float_y = theano.shared(np.asarray(y, dtype=theano.config.floatX), borrow=True)
        shared_y = T.cast(float_y, 'int32')

        # now define the train function
        # we minimize the LOSS, hence we follow the negative gradient
        self._train = theano.function(inputs=[],
                                      outputs=self._missclass_rate,
                                      updates=[(self.W,
                                                # momentum
                                                self.W - (new_rate * self._grad_W + (1 - new_rate) * update_W)),
                                               (self.b,
                                                # momentum
                                                self.b - (new_rate * self._grad_b + (1 - new_rate) * update_b)),
                                               (update_W, new_rate * self._grad_W + (1 - new_rate) * update_W),
                                               (update_b, new_rate * self._grad_b + (1 - new_rate) * update_b)],
                                      givens={
                                          self._X: shared_X[self._minibatch_index * self.batch_size:(
                                                                                                        self._minibatch_index + 1) * self.batch_size],
                                          self._y: shared_y[self._minibatch_index * self.batch_size:(
                                                                                                        self._minibatch_index + 1) * self.batch_size]
                                      })

    def use_sgd(self, X, y):
        params, args = self._climin_setup(X, y)
        _, grad_func = self._climin_funcs()
        # Standard gradient descent: 7.44 % validation, 7.74 % test at epcch 150
        # opt = climin.GradientDescent(params, grad_func, args=args)

        # # Gradient descent with 0.9 momentum: 7.01 % validation, 7.30 % test set at epoch 150
        # opt = climin.GradientDescent(params, grad_func, momentum=0.9, args=args)

        # # Gradient descent with nesterov momentum: 6.98 % validation, 7.30 % test set at epoch 150
        opt = climin.GradientDescent(params, grad_func, momentum=0.9, momentum_type="nesterov", args=args)

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
            W = parameters[:model.feature_dim * model.output_dim].reshape(model.feature_dim, model.output_dim)
            b = parameters[model.feature_dim * model.output_dim:]
            model.W.set_value(W)
            model.b.set_value(b)

        def objective_func(parameters, X, y, model):
            unpack_params(parameters, model)
            return model._objective(X, y)

        def grad_func(parameters, X, y, model):
            unpack_params(parameters, model)
            grad_w, grad_b = model._flat_grad(X, y)
            return np.concatenate((grad_w.flatten(), grad_b))

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
        params = np.zeros(
            self.feature_dim * self.output_dim + self.output_dim)  # 0.01 * random.randn(self.feature_dim * self.output_dim + self.output_dim)

        # pass self during the training to update the parameters of the linear regression model object itself
        # and reuse the theano definitions from the constructor
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

    def train(self, train_x, train_y, valid_x, valid_y, test_x, test_y,
              stop_threshold=10, optimizer=None):
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
        final_W = self.W.get_value()
        final_b = self.b.get_value()
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
                    final_W = self.W.get_value()
                    final_b = self.b.get_value()
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
        plt.show()
        # set W and b to the values before the validation error started rising
        self.W.set_value(final_W)
        self.b.set_value(final_b)

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
    print("mnist.pkl.gz must be directly in this directory, besides logregression.py")
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
    """
    Load data
    """
    get_data()
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
    Init model
    """
    lregression = LRegression(train_x.shape[1], 10, learning_rate=0.13)

    """
    Train model
    """
    lregression.train(train_x, train_y, valid_x, valid_y, test_x, test_y, stop_threshold=20)
    #lregression.train(train_x, train_y, valid_x, valid_y, test_x, test_y, stop_threshold=20, optimizer='rprop')

    """
    Perform on test data
    """
    test_error = lregression.error_rate(test_x, test_y)
    print("Achieved error rate on test set: {}".format(test_error))

    """
    Receptive fields
    """
    image = Image.fromarray(tile_raster_images(
        X=lregression.W.eval().T,
        img_shape=(28, 28), tile_shape=(5, 2),
        tile_spacing=(1, 1)))
    image.show()
