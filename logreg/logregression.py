"""
I will use stochastic gradient descent with mini batches. The reason to not use the Newton method
proposed in Bishop's book is that the length of the features per image is rather big -> computing the
hessian matrix might actually be more inefficient than doing randomized gradient descent.
"""
import theano
import numpy as np
import pickle
import gzip

import climin
from climin import util as cli_util
import itertools

from theano import tensor as T
from theano import shared
from numpy import random

import matplotlib.cm as cm
import matplotlib.pyplot as plt

"""
TODO: try to do this with rotated / shifted images maybe?
"""
class LRegression:
    def __init__(self, feature_dim, output_dim, batch_size=600, max_epochs=150, learning_rate=0.1):
        """

        :param feature_dim:
        :param output_dim:
        :param batch_size:
        :param max_epochs:
        :param learning_rate:
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
        self.W = shared(np.zeros([feature_dim, output_dim]), name='W', borrow=True)
        self.b = shared(np.zeros(output_dim), name="b", borrow=True)
        self.max_epochs = shared(max_epochs, name="max_epochs", borrow=True)
        self.learning_rate = shared(learning_rate, name="learning_rate", borrow=True)

        """
        Internal representation of the model. Theano variables.
        """
        # input Data placeholders
        self._X = T.fmatrix('X')
        self._y = T.lvector('y')
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
        self._loss = -T.log(filtered).mean()

        # compute the gradients
        self._grad_W = T.grad(self._loss, self.W)
        self._grad_b = T.grad(self._loss, self.b)

        self._predictions = T.argmax(probs, axis=1)  # map the softmax probabilities to a digit (0 to 9)
        self._missclass_rate = T.neq(self._predictions, self._y).mean()

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
        new_rate = self.learning_rate * ((self.max_epochs - self._current_epoch) / self.max_epochs)

        # define the update to be used in the momentum calculation
        update_W = shared(np.zeros(self.W.get_value().shape))
        update_b = shared(np.zeros(self.b.get_value().shape))

        # create share variables for the data to operate upon
        shared_X = theano.shared(X, borrow=True)
        shared_y = theano.shared(y, borrow=True)

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
        mini_batch_count = train_x.shape[0] // self.batch_size

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

    def train(self, train_x, train_y, valid_x, valid_y, test_x, test_y, learning_rate=0.9,
              stop_threshold=10, optimizer=None):
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

        validation_threshold = train_x.shape[0] // self.batch_size
        while self._current_epoch.eval() < self.max_epochs.eval():
            batch_err = batch_err + self._train()
            self._inc_minibatch()

            # one training epoch has passed
            if self._minibatch_index.eval() == validation_threshold:
                if no_better_since >= stop_threshold:
                    break;
                self._inc_epoch()
                print("Epoch {} passed. Validating model:").format(self._current_epoch.eval())

                new_err = self.error_rate(valid_x, valid_y)
                if new_err < best_err:
                    no_better_since = 0
                    best_err = new_err
                    # save the best model
                    final_W = self.W.get_value()
                    final_b = self.b.get_value()
                else:
                    no_better_since += 1
                print("Error rate: {}").format(new_err)

                # append errors to plotting data

                valid_errors.append(new_err)
                train_errors.append(batch_err / validation_threshold)
                test_errors.append(self.error_rate(test_x, test_y))

                # reset the mini batch index
                self._minibatch_index.set_value(0)
                batch_err = 0

        print("Training done ...")

        plt.ylabel('error rate')
        valid_legend, = plt.plot(valid_errors, label="validation")
        train_legend, = plt.plot(train_errors, label="training")
        test_legend, = plt.plot(test_errors, label="test")
        plt.legend(handles=[valid_legend, train_legend, test_legend])
        plt.show()
        # set W and b to the values before the validation error started rising
        self.W.set_value(final_W)
        self.b.set_value(final_b)

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
lregression = LRegression(train_x.shape[1], 10, learning_rate=0.9)

"""
Train
"""
lregression.train(train_x, train_y, valid_x, valid_y, test_x, test_y, stop_threshold=20)

"""
Perform on test data
"""
test_error = lregression.error_rate(test_x, test_y)
print("Achieved error rate on test set: {}").format(test_error)

"""
Receptive fields
"""
for i in range(10):
    plt.imshow(lregression.W.eval()[:, i].reshape((28, 28)), cmap=cm.Greys_r)
    plt.show()
