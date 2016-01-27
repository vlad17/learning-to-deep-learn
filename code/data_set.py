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
    # TODO also needs break-capable reentrance
    assert not self._validation_lock
    assert nfolds > 0
    assert nfolds < self._size
    self._validation_lock == True

    vsize = self._size // nfolds
    rounded_size = self._size - self._size % vsize

    # Validation is stored from [-self._validation_size, -1]
    for i in range(0, rounded_size, vsize):
        train = list(chain(range(0, i), range(i + vsize, self._size)))
        validate = range(i, i + vsize)
        assert not self._epoch_lock
        yield (DataSet(self._x[train], self._y[train]),
               DataSet(self._x[validate], self._y[validate]))
    
  def new_epoch(self, batch_size):
    """Creates a generator of batches for a new epoch"""
    assert not self._epoch_lock
    assert batch_size > 0
    # TODO is there a non-reentrant decorator? One that handles breaks
    self._epoch_lock = True
    
    perm = np.arange(self._size)
    np.random.shuffle(perm)
    self._x = self._x[perm]
    self._y = self._y[perm]

    # TODO does this make copies or a slice?
    rounded_size = self._size - self._size % batch_size
    for i in range(0, rounded_size, batch_size):
        j = i + batch_size
        assert not self._validation_lock
        yield self._x[i:j], self._y[i:j]
    
    self._epoch_lock = False



