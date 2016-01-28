"""
Classes for regression.
"""

import tensorflow as tf

# From stack overflow, to prevent needless variables
class cached_property(object):
  def __init__(self, factory):
    self._factory = factory
  def __get__(self, instance, owner):
    attr = self._factory(instance)
    setattr(instance, self._factory.__name__, attr)
    return attr

class SoftMax(object):
  def __init__(self, x, y):
    """Constructs a (batch) SoftMax layer for a 2D-tensor x of dimensions (a, b)
    for an a-sized minibatch of b-length vectors, resulting in 2-D tensor
    of multiclass outputs (a, c).pp

    (a, c) should be the dimensions of the vectorized 2D tensor y with
    which the cross entropy of the output is calculated.

    If the cross entropy is not required, y may be an integer indicating
    how many classes the softmax should predict.

    Uses random N(0, 1) initialization for weights and bias. Note that this
    introduces some variables, which must be initialized."""
    assert len(x.get_shape()) == 2
    classes = 0
    if type(y) == int:
      classes = y
      self._y = None
    else:
      classes = y.get_shape()[1].value
      self._y = y
      assert x.dtype == y.dtype
      def eq_or_none(a, b): return not a or not b or a == b
      assert eq_or_none(y.get_shape()[0].value, x.get_shape()[0].value)
    assert classes > 1
    W = tf.Variable(tf.random_normal([x.get_shape()[1].value, classes]))
    b = tf.Variable(tf.random_normal([classes]))
    self._logit = tf.matmul(x, W) + b
  @cached_property
  def y(self): return tf.nn.softmax(self._logit)
  @cached_property
  def cross_entropy(self):
    assert self._y
    # Loss function direct from unscaled logits
    return tf.nn.softmax_cross_entropy_with_logits(self._logit, self._y)
