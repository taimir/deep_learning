# Multi-layer perceptron
A lot of aspects of this implementation are very similar to my logistic regression. For example, the way the training functions are called and structured, or how the early stopping is organized. Please refer to my explanations there as well.

## Problem 14 - implementing the network
You can find the network in `neuralnet.py` in this directory. It is structured precisely the same way as the linear regression file. Moreover, in the implementation one can exchange the different gradient optimizers like in the logistic regression file - simply specify "optimizer=" as string in the `train()` function call. You can see examples for this in the code.

Again, I decided to implement the neural network myself and not follow the theano tutorial. The net actually trains a tad faster than the Theano example network, which I tried out. Of course, whenever I saw something useful (such as how to connect the layers easily), I integrated it into my solution.

From the bonus tasks: I did a momentum implementation (you can see it in the definition for the default training step, `use_default()`), a learning rate which drops linearly as the epochs progress, early stopping using the same heuristic as I did for logistic regression, and L1 regularization of the W matrices in the layers.

## Problem 15 - Self implementation and first evaluation
With 300 units and `tanh` activation in the hidden layer, with standard stochastic gradient descent (the one I implemented), the network actually did reasonably well.
It trained for 246 seconds (52 epochs) and it got really close to the target mark of 2 % test error:
```
Performance:
2.05 % misclassification error on the validation set
2.11 % misclassification error on the test set`
```

From this point onward, I started using the `rmsprop` optimizer from climin as suggested, and which proved to be really good for MNIST in the logistic regression task.

## Problem 16 - Evaluation of different units
On the topic of weight initialization: as the text suggested, I played around with the initial weights. First of all - leaving the `W` weights to be 0 with `tanh` activation units proved to be disastrous. Naturally so, because `tanh(0) = 0` after all. I also (out of curiosity) tried setting the initial `W` weights to higher values. This proved to be detrimental to how the model trained in the cases of `tanh` and `sigmoid` activations. My argumentation here would be that the derivative of those functions is steep around 0, and if the `sigma(W)` actually is too far to the left or to the right on the axis (relatively speaking), it will be hard for the units to learn anything. Even sampling fro the standard normal for the `W` weights proved to be worse than sampling from a normal distribution around 0 with standard deviation of 0.1. This proved to be the optimal setting for me. This is how I covered the "test out different starting weights" part of the assignment.

Now for the actual evaluation of the different units: this was really simple, I just passed different functions (`T.tanh`, `T.nnet.sigmoid` or `T.nnet.relu`) to the constructor of my `NeuralNetwokClassifier`.

### tanh
I ran the network with 300 hidden units (as required). It trained for 416 seconds (57 epochs) on my machine. It performed well (based on the goal of 2% test error):
```
Performance:
1.87 % error on the validation set
1.94 % error on the test set
```

Next I ran the network with hidden sigmoids:

### sigmoid
Again, 1 hidden layer with 300 hidden units. The model trained for 657 seconds (69 epochs) and did reasonably well. Note that it did slightly worse because I did not reinitialize the `W` matrices with different values this time. As also stated in the Theano tutorial, the optimal starting weights for `sigmoid` and `tanh` differ by factor 4. Therefore, I could have (theoretically) done slightly better here. That's why this particular execution took more time. But since 2 % is already a good error rate, I'll continue with the rectified unit.
```
Performance:
1.93 % error on the validation set
2.04 % error on the test set
```

### relu
The linear rectified units were the third activation I evaluated. With 300 hidden units, the model trained for 578 seconds (57 epochs). It achieved:
```
Performance:
2.01 % error rate on the validation set
2.12 % error rate on the test set
```

Overall, the optimization with `tanh` units was faster and slightly better than the other two alternatives. There were no significant differences between the three activations, however.

## Problem 17 - Error curves
Just like in logistic regression, each error curve contains one line for the: training, validation and test errors. Here it is again visible that the training error keeps going down, while the test and validation error gradually stop decreasing. I intentionally delay the point I stop at and not check how significant the error improvement is, simply because I wanted to squeeze the most out of the models. Still, the early stopping I've implemented does not let the model over-fit in any way. This is also easier because of the L1 regularization that I have introduced.

You can find the error curved under: `error_tanh.png`, `error_sigmoid.png` and `error_relu.png` respectively.

## Problem 18 - Receptive fields
I've plotted the receptive fields for 100 out of the 300 hidden units. There is one file for each of the 3 different activations: `repflds_tanh.png`, `repflds_sigmoid.png`, `repflds_relu.png`. They all look slightly different, but one can clearly see that the different fields represent some parts / segments / regions of the different digits (or sometimes a combination of a couple of them). Those are the hidden, latent aspects of the digits based on which the network has learned to differentiate them.

## Problem 19 - Reaching 2 % misclassification rate
With climin's `rmsprop` and combined with my simplistic early stopping technique and the L1 regularization I used, all of the above experiments came really close to the 2 % misclassification rate on the test set. So my network managed to accomplish this task without the need for translating or rotating the input images.
