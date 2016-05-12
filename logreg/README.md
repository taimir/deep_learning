# Logistic regression

## Problem 8
My implementation is in the file `logregression.py`. Check the big comment in the beginning of the file for more remarks on how I actually implemented it.
I expect mnist.pkl.gz to be directly in this directory. If it is missing, my program will attempt to download it.

I implemented rather straightforward stochastic gradient descent. I do early stopping: whenever the model finds a "best-error" on the validation set, I wait for a while longer (an adjustable parameter). If nothing better is found by then, I stop. I decrease the learning rate linearly as the epochs of training progress. I additionally have a momentum integrated into my learning.

## Problem 9 and 10
### Evaluation of different optimizations
#### My implementation
Training with my personal implementation is rather fast: on my machine it took 25 seconds (61 epochs) to stop training. The model stopped 20 epochs after finding a "best" error rate. Since the validation error didn't decrease again for 20 epochs, the model stopped training and used the optimal parameters that it had found so far.
```
Performance:
6.84 % misclassification on validation set
7.61 % misclassification on test set
```
#### Other optimizers
Training with climin's SGD:
I found out that introducing a momentum helps a lot. Changing between standard and 'nesterov' momentum didn't change that much. Again, the stop point was 20 epochs before the end of the line graphs.
The model trained for 64 seconds (70 epochs)
```
Performance (with momentum):
6.92 % error on validation set
7.61 % error on test set
```

Training with RPROP:
Didn't perform that well. I used 1.01 step grow, 0.1 step shrink.
The model trained for 43 seconds (48 epochs)

```
Performance:
7.04 % on validation set
7.91 % on test set
```


Training with rmsprop:
This model was second best after adam. I think it is due to the fact that the receptive fields are more edgy, i.e. it recognizes the contours slightly better. Nevertheless, I used a relatively small learning rate (0.001) and an average momentum (0.5). Training with a bigger momentum made things worse. The model trained for 52 seconds (56 epochs).
```
Performance:
6.74 % on validation set
7.3 % on test set
```

I am just going to list the results for the other optimizers below, the used parameters for the training can be seen in my code.
Adadelta:
trained for 55 seconds, 57 epochs. The error curve was really noisy (unstable).
```
Performance:
7.02 % error on validation set
7.88 % error on test set
```

ADAM:
This model produced the best results for me. RMSPROP and ADAM showed really nice training behavior, dropping to the low error percentages pretty fast.
Trained for 47 second (48 epochs)
```
Performance:
6.68 % on training set
7.1 % on test set
```

LBFGS
For some reason the pseudo-newton approach was really inefficient. Not only did it take 316 seconds (55 epochs), which goes to show how slow the computation of the Hessian is, but it performed significantly worse than the other models and the error curves were quite jittery. It might be that I misconfigured the hyper parameters, but my intuition says that it makes little sense to mix up a second-degree derivative approach with mini-batch learning.
```
Performance:
7.96 % on validation set
8.26 % on test set
```

## Problem 10
### Receptive fields
One particular thing I noticed about the receptive fields is that when you produce them with a model that has trained on the whole data, they appear `much smoother` than the once produced with mini batch learning. I guess this is normal, since the whole idea of mini-batch learning is to be stochastic (random).

As required, I will commit the file `repflds.png`, which contains the receptive fields of my personal implementation.
I am also uploading the receptive fields for RMSPROP for comparison purposes. You can see how much different they seem, especially on the "edges".

## Problem 11
## Error curves
I always stopped training as soon as the validation error didn't go down for a certain period of time (which is a sign that the model is not actually improving). As one can clearly see on the picture `error.png`, which comes from my personal implementation, the training error does not stop to decrease. But the validation error started to go up slightly at about epoch 40. This is always the time point at which I stopped training: when the training error still decreases, but test and validation error do not decrease and even start to increase. This is a clear sign that the model begins to overfit. My early stopping heuristic was rather simple, but it worked out nice enough (and the one in the theano tutorial didn't seem much more reasonable to me, hence why I picked mine).

I'm also uploading the error curve for LBFGS, to show how unstable it was, and the one of rms_prop, which was maybe the most stable one of them all.

## Problem 13
## Fine tuning and best model
I managed to get the best results out of the ADAM optimizer, with a smaller learning rate of 0.003 and default settings for the other parameters. RMSPROP performed really well as well.
With those two models I reached the 7.1 % mark for the test error rate pretty quickly, and decided that it wasn't necessary to shift / rotate the images any more. It turned out fine even with the default settings. I guess the early stopping helped as well.

I'd argue that problem 13 is not that good of a practice because it is all about exploiting very specific properties of the MNIST dataset. Trying to achieve a really small error rate by tweaking the images, or really manually searching for the very best hyper parameter setup is of little benefit to other scientists. Such experiments are rather unreliable, not that easily reproducible, not to mention that the improvement in the error rate is so small that it might have to do with chance.
