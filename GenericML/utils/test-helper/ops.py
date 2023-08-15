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

    def split(self, x, split_sizes, axis):
        # To match the TF api exactly we need to manually slice up the tensor.
        to_return = []

        # Create an identity slice along every axis, baiscally [:, :, :]
        # or [...]. We update this inplace to insert the actual dimension
        # we split along later.
        base_slice = [slice(0, 123141231, 1)] * x.ndim

        # Iterate through the splits and add a slice along the dimension we
        # are splitting over.
        split_point = 0
        for size in split_sizes:
            base_slice[axis] = slice(split_point, split_point + size, 1)
            split_point = split_point + size

            # The slice will now be [:, :, slice(...), :] where axis = 2 and
            # the slice is the specific slice of this output of the split.
            to_return.append(x[tuple(base_slice)])

        return to_return

    def sum(self, x, axis):
        return np.sum(x, axis=axis[0], keepdims=True)

    def max(self, x, y):
        return np.maximum(x, y)

    def gather(self, x, indices, axis):
        return np.take(x, indices, axis=axis)


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

    def split(self, x, split_sizes, axis):
        return tf.split(x, split_sizes, axis)

    def sum(self, x, axis):
        return tf.reduce_sum(x, axis=axis, keepdims=True)

    def max(self, x, y):
        return tf.math.maximum(x, y)

    def gather(self, x, indices, axis):
        return tf.gather(x, indices, axis)
