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
  def __init__(self, x, y, magnitude=0.1):
    """Constructs a (batch) SoftMax layer for a 2D-tensor x of dimensions (a, b)
    for an a-sized minibatch of b-length vectors, resulting in 2-D tensor
    of multiclass outputs (a, c).

    (a, c) should be the dimensions of the vectorized 2D tensor y with
    which the cross entropy of the output is calculated.

    Either of x or y may have its first dimensions set to None (which allows
    for a runtime-selected batch size). However, if using this option, user
    must enforce consistency between input batch sizes and their associated
    label batch size.

    If the cross entropy is not required, y may be an integer indicating
    how many classes the softmax should predict.

    Uses random [-1, 1]-truncated N(0, M) initialization for weights and
    constant M for bias, where M is set by the optional magnitude argument.
    Note that this introduces some variables, which must be initialized.

    Adds trainable variables and resulting processed tensors to the
    default graph."""
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
    W = tf.Variable(tf.truncated_normal([x.get_shape()[1].value, classes],
                                        stddev=magnitude))
    b = tf.Variable(tf.constant(magnitude, shape=[classes]))
    self._logit = tf.matmul(x, W) + b
  
  @cached_property
  def y(self): return tf.nn.softmax(self._logit)
  
  @cached_property
  def cross_entropy(self):
    assert self._y is not None
    # Loss function direct from unscaled logits
    return tf.nn.softmax_cross_entropy_with_logits(self._logit, self._y)
