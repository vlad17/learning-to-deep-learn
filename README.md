# TensorFlow-Learn

## Goal

Repo for me to learn the TensorFlow API. Contains some useful abstracted TensorFlow common-usage classes, if only just "useful" in the informative sense.

## Ready-made classes

* Softmax-based regression
* Gaussian Mixture Model (diagonal covariance only)

## Dependencies

`jupyter tensorflow numpy tabulate six sklearn matplotlib`

Relies on python3. Tested on 3.3+.

## Prepared data

Though not required, you may unzip `data.tgz` to recover the pre-trained CNN used in `notebooks/mnist.ipynb`. This also contains the MNIST data.

## Binder

[![Binder](http://mybinder.org/badge.svg)](http://mybinder.org/repo/vlad17/TensorFlow-Learn)

## TODOs

  * Vectorize TFGMM.marginalize() for a series of observations of the same indices.
  * At creation time for TFGMM save dtype, then convert all data input arrays to that type.
  * Refactor helper functions in em_gmm.py to have leading underscore.
  * Run pylint on repo (maybe set up a pre-commit hook?)
  * Highlight in `gmm.ipynb` where the argmax prediction is unstable.
  