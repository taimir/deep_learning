# PCA and sparse autoencoder

Check out file `pca.py` for the PCA implementation, file `autoenc.py` for the autoencoder implementation.

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
I implemented the Autoencoder entirely basing it on my NN imlementation from the previous task. That way I could reuse a lot of the optimizations I did. The only thing I had to change was the way the output layer was wired, the L1 normalization and also the cost function. I also removed the `network_spec`, so that the autoencoder has exactly 2 layers: 1 hidden, 1 output, in all cases.

For training, usually 15 epochs have proved to be quite enough, achieving decent error rates.

Note that I currently do not corrupt the input, even though the function is there (took it over from the Theano tutorial). We do have the L1 regularization, and I did not want to mix both of the options.

## Problem 23 Autoencoder with sparsity constraint
I implemented the L1 regularization and added it to the loss function - line 99 of `autoenc.py`. An operator `spasity_coef` is multiplied by the L1 term and specifies how sparse the parameters will be. The is `lambda` from the problem statement. Later I produced plots for the different lambdas to answer the questions.

In both of the below tasks, I used 300 hidden relu units.
Also you can check the `originals.png` when comparing the reconstructions.

## Problem 24 Reconstructions of MNIST with the autoencoder
As asked for in the task, I have produced a file with the digit reconstructions called `autoencoderrec.png`. Additionally, I constructed two more files for comparison purposes: `autoencoderrec_lower_sparsity.png` and `autoencoderrec_higher_sparsity.png`, where I adjusted the `lambda` values respectively: 0.001 for lower, 0.05 for the default `autoencoderrec.png` and 0.2 for the higher sparsity.

As you can see, higher `lambda` values read to unclear digit reconstructions (and as we will see to a bigger number of weak hidden units, i.e. sparsity). Lower `lambda` values increase the accuracy during reconstruction, but the receptive fields become more dense.

## Problem 25 Receptive fields of the autoencoder
Similarly to the above task, I constructed a file `autoencoderfilter.png` with the receptive fields of the autoencoder. You can see that even though the receptive fields are very sparse in this plot, the reconstruction isn't even that inaccurate. Again, `autoencoderfilter_lower_sparsity.png` and `autoencodrfilter_higher_sparsity.png` are for different `lambda` values (the same as above).
Here we see again what I mentioned already: with a lower sparsity (lower `lambda` value) we see more dense receptive fields. On the contrary, increasing the `lambda` leads to sparser receptive fields (but bigger reconstruction errors). But it is obvious that the MNIST dataset is indeed inherently rather sparse (has a lot of structure).

## Problem 26 Sparse encoding of MNIST
To conclude, I can say that MNIST can be reconstructed reasonably well even with a rather sparse encoding ( a lot of inactive units). This is because parts of the digits are actually correlated, so less information is required to describe the whole "world" of handwritten digits and sparse encodings are easily possible. If the digits were actually samples from a normal distribution (with high entropy), this wouldn't be the case. But since they are handwritten digit, we can identify interesting structures.
