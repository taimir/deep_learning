# PCA and sparse autoencoder

## Problem 20 PCA implementation
The PCA implementation is in `pca.py` in this directory. For it to function, it needs `mnist.pkl.gz` to be in the directory as well, and also `cifar-10-batches-py` to be a subdirectory, containing all the batches of CIFAR.
In my implementation I first convert CIFAR to grayscale for convenience (that was written in the assigments sheet).
The actual computation of the PCA for all of the pairs (to form the big scatterplots) might take a couple of minutes to complete. Please, be patient (or reduce the number of data points being used in the code).

Some remarks:
 - I optimized the actual application of PCA on a given input in Theano
 - the computation of the eigenvectors and eigenvalues is done by numpy (scipy). This was also the advice of Prof. Smagdt on the Piazza forum. It is more robust that way, and actually quite efficient, judging by how fast the covariance is computed.
 - When the transformations are done, the program produces two (rather large) files with the pairwise scatterplots of the PCA-transformed CIFAR and MNIST datasets. They are to be found in this directory.

## Problem 21 PCA scatterplots
The scatterplots are called `scatterplotMNIST.png` and `scatterplotCIFAR.png`. They are plots between every pair of classes in the transformed space. The files are intentionally large, so that people can zoom in to observe the points. Simple start the program to generate the plots once again.

From the plots it becomes obvious that the MNIST pairs of classes are easily linearly separable. The CIFAR-10 ones, on the other hand, are not that easy to separate. Still, there are some relative boundries between the points from the different classes.

I also tried plotting all of MNIST into 1 diagram (the 10 classes). Then the separation between the classes is not as clear as in the pair-wise plot.

## Problem 22 Autoencoder implementation
I implemented the Autoencoder entirely basing it on my NN imlementation from the previous task. That way I could reuse a lot of the optimizations I did. The only thing I had to change was the way the output layer was wired. I also removed the `network_spec`, so that the autoencoder has exactly 2 layers: 1 hidden, 1 output, in all cases.

I implemented the L1 regularization and added it to the loss function. An operator `spasity_coef` is multiplied by the L1 term and specifies how sparse the parameters will be.

For training, usually 15 epochs have proved to be quite enough, achieving decent error rates.

Note that I currenlty do not corrupt the input, even though the function is there (took it over from the Theano tutorial). We do have the L1 regularization, and I did not want to mix both of the options.

## Problem 23 Autoencoder with sparsity constraint

## Problem 24 Reconstructions of MNIST with the autoencoder

## Problem 25 Receptive fields of the autoencoder

## Problem 26 Sparse encoding of MNIST
