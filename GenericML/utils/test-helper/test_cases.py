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


def get_test_case(target_name):
    for name, obj in inspect.getmembers(sys.modules[__name__]):
        if inspect.isclass(obj):
            if name == target_name:
                return obj
    raise Exception(f"No object of name {name} found")
