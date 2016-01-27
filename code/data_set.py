"""
Lightweight container class for supervised learning.
"""

import numpy as np

class DataSet(object):
  def __init__(self, x, y):
    """Construct a DataSet for examples x mapping to y. """
    assert x.shape[0] == y.shape[0], 'x.shape {} y.shape {}'.format(
        images.shape, labels.shape)
    self._size = x.shape[0]
    self._x = x
    self._y = y
    self._epoch_lock = False
  @property
  def x(self):
    return self._x
  @property
  def y(self):
    return self._y
  @property
  def size(self):
    return self._size
  def new_epoch(self, batch_size):
    """Creates a generator of batches for a new epoch"""
    assert not self._epoch_lock # TODO is there a non-reentrant decorator?
    self._epoch_lock = True
    
    perm = np.arange(self._size)
    np.random.shuffle(perm)
    self._x = self._x[perm]
    self._y = self._y[perm]

    # TODO does this make copies or a slice?
    rounded_size = self._size - self._size % batch_size
    for i in range(0, rounded_size, batch_size):
        j = i + batch_size
        yield self._x[i:j], self._y[i:j]

    self._epoch_lock = False

