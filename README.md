# TensorFlow-Learn
Scratch repo for me to learn the TensorFlow API

## future ideas

fast cross validation:
    # Invert the permutation
    inv = np.zeros(self._size)
    for i, x in enumerate(perm): inv[x] = i
    perm = inv
    self._x = self._x[perm]
    self._y = self._y[perm]
