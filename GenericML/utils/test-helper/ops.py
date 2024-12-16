# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import math

import numpy as np

"""
This file contains the op implementations for numpy.
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

    def neq(self, x, y):
        return x != y


# Numpy implementation of each primitive.
class NumpyImpl(CommonImpls):
    def constant(self, values, dtype=None):
        if dtype:
            return np.array(values, dtype=dtype)

        return np.array(values)

    def ceil(self, x):
        return np.ceil(x)

    def floor(self, x):
        return np.floor(x)

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
        return np.vectorize(apply)(x)

    def exp(self, x):
        return np.exp(x)

    def log(self, x):
        return np.log(x)

    def log1p(self, x):
        return np.log1p(x)

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

    def is_nan(self, x):
        return np.isnan(x)

    def is_inf(self, x):
        return np.isinf(x)

    def logical_and(self, x, y):
        return np.logical_and(x, y)

    def logical_or(self, x, y):
        return np.logical_or(x, y)

    def logical_xor(self, x, y):
        return np.logical_xor(x, y)

    def isqrt(self, x):
        return 1.0 / np.sqrt(x)
