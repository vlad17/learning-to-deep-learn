# TensorFlow-Learn

## Goal

Repo for me to learn the TensorFlow API. Contains some useful abstracted TensorFlow common-usage classes, if only just "useful" in the informative sense.

## Dependencies

`tensorflow numpy tabulate six`

## Overarching TODOs:

Softmax with simulated annealing

fast cross validation:

```
    # Invert the permutation
    inv = np.zeros(self._size)
    for i, x in enumerate(perm): inv[x] = i
    perm = inv
    self._x = self._x[perm]
    self._y = self._y[perm]
```

Conv net (try some asymmetry later)

