# Basic setup and first tutorials

Here I have documented my progress through the tutorials on Theano and setting up the installation environment. It is my first time working with Theano, so I had to spend some time getting familiar with the library.

## Problem 1
Anaconda was really easy to install under Windows. Unfortunately, I had python-3.5 installed before I began and I forgot to uninstall it first, which caused me some trouble. Reinstalling the whole python stack fixed everything and I was able to continue working with the other libraries.

## Problem 2
JetBrains is really superb! All of the code I developed entirely in this IDE.

## Problem 3
I used bleeding edge Theano, and I installed it just as is described in the Assignments sheet. There was no trouble with the theano installation, the only thing I forgot to do is actually to install a GNU c++ compiler prior to trying to use Theano. So I installed MingW on Windows, and once g++ was there everything ran smoothly.

## Problem 4
Unfortunately I do not have an NVIDIA graphics card. I wasn't able to test out the real speed of theano when it runs on the GPU. I tried to do my best and really use theano.config.floatX as much as I could throughout my code so that there's a better chance some part of it will run on the GPU. But I cannot guarantee this, because I didn't get a chance to test it out.

## Problem 5
When doing the tutorials, I basically went through them one by one. You can find the snippets I assembled in this directory. Here and there, I added some comments that were supposed to help my understanding of the different Theano aspects I encountered. I doubt it is of much use if I go into a lot of detail of what I found interesting and what not. Once I understood the whole theory behind the graphs (OP, APPLY, etc. nodes), using theano became much easier.

## Problem 6
Climin caused me some trouble. I wasn't able to get it to run at first, apparently because the numpy distribution packaged with Anaconda (Python 2.7, 64-bit) didn't contain some of the necessary DLL's to run climin. Looking at the climin code, it seems that those DLLs are only necessary because of some issue with keyboard interrupts. Anyways, I downloaded a numpy wheel package from [here](http://www.lfd.uci.edu/~gohlke/pythonlibs/) and installed it with pip. Once the missing DLLs were there, it was all O.K.

## Problem 7
Not much to say here: I downloaded the datasets and used `matplotlib` to have a look at a couple of the images. I checked out the dimensions of the datasets, and easy ways to turn them into `np.arrays`. Reading the Theano tutorials helped a lot, because they contained examples of how to `unpickle` the MNIST dataset, for example.
