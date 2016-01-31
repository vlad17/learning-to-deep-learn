"""
Lightweight container class for supervised learning.
"""

import numpy as np
from itertools import *

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
    new_epoch, which only returns batches from non-validation examples"""
    assert not self._validation_lock
    assert nfolds > 0
    assert nfolds < self._size
    try:
      self._validation_lock = True
      vsize = self._size // nfolds
      rounded_size = self._size - self._size % vsize
      # Validation is stored from [-self._validation_size, -1]
      for i in range(0, rounded_size, vsize):
        self._validation_range = (i, i + vsize)
        # TODO return views, not slices
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

        # TODO does this make copies or a slice?
        for i in range(train_size // batch_size):
            assert (lo, hi) == self._validation_range
            batch = np.random.randint(train_size, size=batch_size)
            batch += (lo <= batch) * validation_size
            # TODO return a view again
            yield self._x[batch], self._y[batch]
    finally:
        self._epoch_lock = False



