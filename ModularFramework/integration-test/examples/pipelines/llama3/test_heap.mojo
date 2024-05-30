# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: mojo -I %sdk_examples_dir "%s"
#
# Unit tests for MaxHeap

from testing import *

from pipelines.llama3.tokenizer.max_heap import MaxHeap, MinHeap


def test_simple():
    heap = MaxHeap[Int]()
    for i in range(10):
        heap.push(i)

    for i in range(10):
        assert_equal(9 - i, heap.pop())


def test_min_heap_simple():
    heap = MinHeap[Int]()
    for i in range(10):
        heap.push(i)

    for i in range(10):
        assert_equal(i, heap.pop())


def main():
    test_simple()
    test_min_heap_simple()
