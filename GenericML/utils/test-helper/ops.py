# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import numpy as np
import math


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


class CommonImpls:
    def add(self, x, y):
        return x + y

    def sub(self, x, y):
        return x - y

    def mul(self, x, y):
        return x * y

    def lt(self, x, y):
        return x < y

    def lte(self, x, y):
        return x <= y

    def gt(self, x, y):
        return x > y

    def gte(self, x, y):
        return x >= y

    def eq(self, x, y):
        return x == y


# Numpy implementation of each primitive.
class NumpyImpl(CommonImpls):
    def constant(self, values, dtype=None):
        if dtype:
            return np.array(values, dtype=dtype)

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
        return np.sum(x, axis=axis, keepdims=True)

    def max(self, x, y):
        return np.maximum(x, y)

    def gather(self, x, indices, axis):
        return np.take(x, indices, axis=axis)

    def strided_slice(self, x, start, stop, step):
        slice_to_apply = []
        for i in range(len(start)):
            slice_to_apply.append(slice(start[i], stop[i], step[i]))
        return x[tuple(slice_to_apply)]

    def abs(self, x):
        return np.abs(x)

    def erf(self, x):
        apply = lambda t: math.erf(t)
        out = np.copy(x)
        np.vectorize(apply)(out)
        return out

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def max(self, x, limit):
        return np.maximum(x, limit)

    def min(self, x, limit):
        return np.minimum(x, limit)

    def select(self, cond, x, y):
        return np.where(cond, x, y)

    def sin(self, x):
        return np.sin(x)

    def cos(self, x):
        return np.cos(x)

    def tanh(self, x):
        return np.tanh(x)


# The keras implementation of each primitive.
class TFImpl(CommonImpls):
    def constant(self, values, dtype=None):
        if dtype:
            return tf.constant(np.array(values, dtype=dtype))
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
        return tf.gather(x, indices, axis=axis)

    def strided_slice(self, x, start, stop, step):
        return tf.strided_slice(x, start, stop, step)

    def abs(self, x):
        return tf.abs(x)

    def erf(self, x):
        return tf.math.erf(x)

    def exp(self, x):
        return tf.exp(x)

    def log(self, x):
        return tf.math.log(x)

    def max(self, x, limit):
        return tf.math.maximum(x, limit)

    def min(self, x, limit):
        return tf.math.minimum(x, limit)

    def select(self, cond, x, y):
        return tf.raw_ops.Select(condition=cond, x=x, y=y)

    def sin(self, x):
        return tf.sin(x)

    def cos(self, x):
        return tf.cos(x)

    def tanh(self, x):
        return tf.tanh(x)
