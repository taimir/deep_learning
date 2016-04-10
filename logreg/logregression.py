"""
I will use stochastic gradient descent with mini batches. The reason to not use the Newton method
proposed in Bishop's book is that the length of the features per image is rather big -> computing the
hessian matrix might actually be more inefficient than doing randomized gradient descent.
"""

import theano
from theano import tensor as T
from theano import shared
import numpy as np
from numpy import random
import pickle
import gzip
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class LRegression:
    """
    Constant, number of maximum training epochs
    """
    MAX_EPOCHS = 150.0

    def __init__(self, feature_dim, output_dim):
        print("Initializing logistic regression model")
        # Data placeholders
        X = T.fmatrix('X')
        y = T.lvector('y')

        # Parameter placeholders
        # sample W with smaller variance: 0.01 instead of 1
        # 0.1 * random.randn(feature_dim, output_dim)
        self.W = shared(np.zeros([feature_dim, output_dim]), name='W', borrow=True)
        self.b = shared(np.zeros(output_dim), name="b", borrow=True)

        # training functions
        logit = T.dot(X,
                      self.W) + self.b  # returns a matrix, with N rows and 10 numbers per instance (per class), b is broadcast

        probs = T.nnet.softmax(
            logit)  # rowwise softmax for each instance, we get N x 10 matrix with probs for each digit per instance

        # NLL is sum of only the log(prob) for the CORRECT class per instance
        # we can select them by indexing the probs matrix
        filtered = probs[
            T.arange(y.shape[0]), y]  # broadcasted indexes, the result is a vector of prob. for each instance

        # now we take the logs and sum them up (with a minus)
        loss = -T.log(filtered).mean()

        # compute the gradients
        grad_W = T.grad(loss, self.W)
        grad_b = T.grad(loss, self.b)

        predictions = T.argmax(probs, axis=1)  # map the softmax probabilities to a digit (0 to 9)
        missclass_rate = T.neq(predictions, y).mean()

        # define a theano function for the learning rate
        max_epochs = shared(LRegression.MAX_EPOCHS)
        ep = T.iscalar('ep')
        init = T.dscalar('init')
        new_rate = init * ((max_epochs - ep) / max_epochs)
        self._learning_rate = theano.function(inputs=[ep, init], outputs=[new_rate])

        update_W = shared(np.zeros(self.W.get_value().shape))
        update_b = shared(np.zeros(self.b.get_value().shape))
        # now define the train function
        # we minimize the LOSS, hence we follow the negative gradient
        self._train = theano.function(inputs=[ep, init, X, y],
                                      outputs=missclass_rate,
                                      updates=[(self.W,
                                                # momentum
                                                self.W - (new_rate * grad_W + (1 - new_rate) * update_W)),
                                               (self.b,
                                                # momentum
                                                self.b - (new_rate * grad_b + (1 - new_rate) * update_b)),
                                               (update_W, new_rate * grad_W + (1 - new_rate) * update_W),
                                               (update_b, new_rate * grad_b + (1 - new_rate) * update_b)])

        # define the error rate function
        self._error_rate = theano.function(inputs=[X, y], outputs=missclass_rate)

        # define the predict function. Note: it only needs the data points, no labels as inputs of course.
        self._predict = theano.function(inputs=[X], outputs=[predictions])

    def error_rate(self, X, y):
        """
        The function evaluates the model for the given X and returns the error rate with respect to the real labels y.
        :param X: the design matrix containing the data points
        :param y: the corresponding labels
        :return: the error rate when predicting on this model
        """
        return self._error_rate(X, y)

    def learning_rate(self, epoch, initial_rate):
        """
        :param epoch: the current training epoch
        :param initial_rate: the initial learning rate (we're decreasing it with a negative exponent. factor)
        :return: the current learning rate
        """
        return self._learning_rate(epoch, initial_rate)

    def train(self, train_x, train_y, valid_x, valid_y, test_x, test_y, init_learning_rate=0.9, batch_size=600,
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
        final_W = self.W.get_value()
        final_b = self.b.get_value()
        best_err = np.inf
        train_errors = []
        valid_errors = []
        test_errors = []
        batch_err = 0
        no_better_since = 0

        counter = 0
        epoch_counter = 0
        validation_threshold = train_x.shape[0] // batch_size
        while epoch_counter < LRegression.MAX_EPOCHS:
            indices = random.choice(train_x.shape[0], batch_size)
            mini_batch_x = train_x[indices]
            # counter * batch_size: (counter + 1) * batch_size
            mini_batch_y = train_y[indices]
            batch_err = batch_err + self._train(epoch_counter, init_learning_rate, mini_batch_x, mini_batch_y)
            counter += 1
            # one training epoch has passed
            if counter == validation_threshold:
                if no_better_since >= stop_threshold:
                    break;

                epoch_counter += 1
                print("Epoch {} passed. Validating model:").format(epoch_counter)
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

                counter = 0
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
lregression = LRegression(train_x.shape[1], 10)

"""
Train
"""
lregression.train(train_x, train_y, valid_x, valid_y, test_x, test_y)

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
