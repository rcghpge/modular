# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

import sys, inspect
import numpy as np

"""
This file contains the test implementations. Each test class added to this file
can then be picked up by the tools just using its name. For instance:

python export_mo_graph.py -n BasicTranspose

Will pick up the class named `BasicTranspose` in this file.
"""

# Inherit from this to get some sensible defaults.
class TestBase:
    @staticmethod
    def compare_results(ours, theirs):
        return np.allclose(ours, theirs, rtol=1e-05, atol=1e-08, equal_nan=True)

    @staticmethod
    def num_runs():
        return 25


# Basic transpose test case.
class BasicTranspose(TestBase):
    @staticmethod
    def input_sizes_to_test():
        return [([4, 2, 1, 6], [4])]

    @staticmethod
    def get_input_specs():
        # Create a test of rank 3 with and a permute vector of rank 1.
        return [
            ((None, None, None, None), np.float32),
            ((4,), np.int32),
        ]

    @staticmethod
    def generate_inputs(shapes):
        permute = np.arange(len(shapes[0])).astype(np.int32)
        permute = np.random.permutation(permute)
        return [
            np.random.uniform(-2.0, 2.0, shapes[0]).astype(np.float32),
            permute,
        ]

    @staticmethod
    def run(inputs, impl):
        return impl.transpose(inputs[0], inputs[1])


# A normal batch matmul.
class BatchMatmul(TestBase):
    @staticmethod
    def input_sizes_to_test():
        return [([5, 11, 6], [1, 6, 23])]

    @staticmethod
    def get_input_specs():
        return [
            ((None, None, None), np.float32),
            ((None, None, None), np.float32),
        ]

    @staticmethod
    def generate_inputs(shapes):
        return [
            np.random.uniform(-2.0, 2.0, shapes[0]).astype(np.float32),
            np.random.uniform(-2.0, 2.0, shapes[1]).astype(np.float32),
        ]

    @staticmethod
    def run(inputs, impl):
        return impl.bmm(inputs[0], inputs[1])


# A normal batch matmul.
class BatchMatmulTrans(BatchMatmul):
    @staticmethod
    def input_sizes_to_test():
        return [([5, 11, 6], [1, 23, 6])]

    @staticmethod
    def run(inputs, impl):
        permute = impl.constant((0, 2, 1))
        trans = impl.transpose(inputs[1], permute)
        return impl.bmm(inputs[0], trans)


# Test for range
class Range(TestBase):
    @staticmethod
    def compare_results(ours, theirs):
        return np.array_equal(ours, theirs)

    @staticmethod
    def num_runs():
        return 100

    @staticmethod
    def input_sizes_to_test():
        return [(3,)]

    @staticmethod
    def get_input_specs():
        return [((3,), np.int32)]

    @staticmethod
    def generate_inputs(shapes):
        start = np.random.randint(-64, 64)
        stop = np.random.randint(-64, 64)
        diff = stop - start
        if diff == 0:
            start, stop, step = (-3, 3, 1)
        elif diff > 0:
            step = np.random.randint(1, diff) if diff != 1 else 1
        else:
            step = np.random.randint(diff, -1) if diff != -1 else -1

        return [np.array([start, stop, step], dtype=np.int32)]

    @staticmethod
    def run(inputs, impl):
        # Until `mt` supports the input type we have to slice them out #18187.
        start, stop, step = impl.split(inputs[0], [1, 1, 1], 0)
        start = impl.reshape(start, shape=())
        stop = impl.reshape(stop, shape=())
        step = impl.reshape(step, shape=())
        return impl.range(start, stop, step)


class Split(TestBase):
    @staticmethod
    def compare_results(ours, theirs):
        return np.array_equal(ours, theirs)

    @staticmethod
    def num_runs():
        return 300

    @staticmethod
    def input_sizes_to_test():
        return [([20, 12, 12], [6], [1])]

    @staticmethod
    def get_input_specs():
        return [
            ((None, None, None), np.int32),
            ((6,), np.int32),
            ((1,), np.int32),
        ]

    @staticmethod
    def generate_inputs(shapes):
        start = np.random.randint(-128, 128, size=shapes[0]).astype(np.int32)
        axis = np.random.randint(-start.ndim, start.ndim - 1, size=(1,)).astype(
            np.int32
        )

        dim_splitting_along = start.shape[axis[0]]
        split_sizes = []
        splits_left_to_make = shapes[1][0]

        # Create the vector representing the size along each dimension to split
        # ensuring we are splitting to at least 1 at each point.
        for i in range(shapes[1][0] - 1):
            assert dim_splitting_along > 0
            assert dim_splitting_along > splits_left_to_make
            assert splits_left_to_make > 0

            split_size = np.random.randint(
                1, dim_splitting_along - splits_left_to_make
            )
            split_sizes.append(split_size)

            dim_splitting_along -= split_size
            splits_left_to_make -= 1

        # Last dim is however many are left
        split_sizes.append(dim_splitting_along)

        # If the first number is big we might end up with all ones so shuffle
        # a bit so it's not always the first one.
        np.random.shuffle(split_sizes)

        split_sizes = np.array(split_sizes).astype(np.int32)

        return [start, split_sizes, axis]

    @staticmethod
    def run(inputs, impl):
        axis = impl.reshape(inputs[2], shape=())
        return impl.split(inputs[0], inputs[1], axis)


def get_test_case(target_name):
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if name == target_name:
                return obj
    raise Exception(f"No object of name {name} found")
