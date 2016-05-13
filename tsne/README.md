# t-SNE
The code I used is in `bhtsne.py`. It has a dependency on the compiled C++ binary `bh_tsne.exe`.

## Problem 27 Understanding t-SNE
t-SNE is a technique for visualization of higher dimensional data into lower subspaces. It is mightier than PCA, as it does not just stick to linear projections. With the Barnes-Hut algorithm, it is also really fast and memory efficient (`O(nlogn)` runtime). SNE stands for "Stochastic Neighbor Embedding". The rough idea is, that in the embedded space and in the original space two joint distributions of points are considered. The probability of two points is high if they are similar, low if they are dissimilar, and this must hold in both the embedded space and the original space. Thus, the goal is to pretty much minimize the KL divergence of the distributions (one embedded, one in the original space). t-SNE is just one of the existing variants (optimizations) of this idea.

## Problem 28 Evaluating t-SNE
### IMPORTANT: The python code of the bh-tsne is just a wrapper around a c++ binary. You will need to execute the command below first, before you can use the python scripts.

Compile the C++ source using the following command:
```
g++ sptree.cpp tsne.cpp -o bh_tsne -O2
```

Note: I had to adapt the original python script to make it run on Windows. If you have trouble running it on Linux, please download the original python script from the t-SNE website and put my changes in it.

I evaluated `bh-tsne` on a couple of different training sets, including `MNIST, CIFAR-10, diabetes, olivetti_faces, covtype`.
http://scikit-learn.org/stable/datasets
You can find the resulting images placed directly in this directory.

The last three are from the standard `sklearn` training sets, BTW. From those five, only MNIST, CIFAR-10 and covtype are rather big. This is why also their embeddings make much more sense on the first sight. When I ran t-SNE on the smaller datasets (diabetes, olivetti_faces) it performed O.K., but not great (maybe because of the lack of data points?).

## Problem 29 t-SNE for MNIST
As asked I reproduced figure 5 from the paper on `bh-tsne`, it is called `figure5_reproduction.png`. The only difference is that I did not plot the images themselves, but I plotted the colors of each class (otherwise the image would be huge in size, and I would have trouble uploading it to github). But the main idea is still visible: bh-tsne manages to reflect the different classes perfectly in the 2D embedding. Clearly, running a classification algorithm on such a clear cut between the classes will work great. MNIST is easily separable through t-SNE.
