"""
Lightweight container class for supervised learning.
"""

import numpy as np
from itertools import *
import tensorflow as tf

class DataSet(object):
  def __init__(self, x, y):
    """Construct a DataSet for examples x mapping to y. """
    assert x.shape[0] == y.shape[0], 'x.shape {} y.shape {}'.format(
        images.shape, labels.shape)
    self._size = x.shape[0]
    self._x = x
    self._y = y
    self._epoch_lock = False
    self._validation_lock = False
    self._validation_range = (0, 0)
  @property
  def x(self):
    return self._x
  @property
  def y(self):
    return self._y
  @property
  def size(self):
    return self._size

  def cross_validation(self, nfolds):
    """Creates a generator with validation folds compatible with
    new_epoch, which only returns batches from non-validation examples."""
    assert not self._validation_lock
    assert nfolds > 0
    assert nfolds <= self._size    
    try:
      self._validation_lock = True
      vsize = self._size // nfolds
      rounded_size = self._size - self._size % vsize
      for i in range(0, rounded_size, vsize):
        self._validation_range = (i, i + vsize)
        vx = self._x[self._validation_range[0]:self._validation_range[1]]
        vy = self._y[self._validation_range[0]:self._validation_range[1]]
        yield DataSet(vx, vy)
    finally:
      self._validation_range = (0, 0)
      self._validation_lock = False
    
  def new_epoch(self, batch_size):
    """Creates a generator of batches for a new epoch"""
    assert not self._epoch_lock
    assert batch_size > 0
    try:
      self._epoch_lock = True

      lo, hi = self._validation_range
      validation_size = hi - lo
      train_size = self._size - validation_size

      # It's possible to make a zero-copy epoch
      # if we do a Knuth shuffle on the epoch with a permutation
      # that keeps the validation range invariant.
      #
      # One needs to un-shuffle before the next fold, though.
      #
      # This would create draws from the training set WITHOUT
      # replacement.
      #
      # However, a batch-sized copy is probably not worth the trouble.
      for i in range(train_size // batch_size):
        assert (lo, hi) == self._validation_range
        batch = np.random.randint(train_size, size=batch_size)
        batch += (lo <= batch) * validation_size
        yield self._x[batch], self._y[batch]
    finally:
      self._epoch_lock = False

  def multiclass_error(self, x, predicted_y, actual_y,
                       feed_dict={}, session=None):
    """Returns the average error for the entire data set for predicting
    a mulit-class label (compares most likely classes only). Feeds
    a dictionary to TensorFlow session with this dataset's inputs as
    'x' and outputs as 'actual_y'."""
    is_correct = tf.equal(tf.argmax(predicted_y, 1), tf.argmax(actual_y, 1))
    accuracy = tf.reduce_mean(tf.cast(is_correct, "float"))
    feed_dict[x] = self._x
    feed_dict[actual_y] = self._y
    return 1 - accuracy.eval(feed_dict=feed_dict, session=session)


    


