# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np

try:
    import tensorflow as tf
except:
    # TF is only needed for generating examples not for testing.
    pass

"""
This file contains the op implementations, one for tensorflow and one
for numpy. We compose tests using the generic interface then use TF to
export and numpy to test.
"""


# Numpy implementation of each primitive.
class NumpyImpl:
    def constant(self, values):
        return np.array(values)

    def transpose(self, input, permute):
        return np.transpose(input, permute)

    def bmm(self, x, y):
        return np.matmul(x, y)

    def range(self, start, stop, step):
        return np.arange(start, stop, step)

    def reshape(self, x, shape):
        return x.reshape(shape)

    def split(self, x, num_splits, axis):
        return np.split(x, num_splits, axis)


# The keras implementation of each primitive.
class TFImpl:
    def constant(self, values):
        return tf.constant(np.array(values))

    def transpose(self, input, permute):
        return tf.transpose(input, permute)

    def bmm(self, x, y):
        return tf.linalg.matmul(x, y)

    def range(self, start, stop, step):
        return tf.range(start, stop, step)

    def reshape(self, x, shape):
        return tf.reshape(x, shape)

    def split(self, x, num_splits, axis):
        return tf.split(x, num_splits, axis)
