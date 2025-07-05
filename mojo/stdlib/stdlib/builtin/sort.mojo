# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements the built-in `sort` function.

These are Mojo built-ins, so you don't need to import them.
"""

from math import ceil
from sys import bitwidthof

from bit import count_leading_zeros
from memory import Span

# ===-----------------------------------------------------------------------===#
# sort
# ===-----------------------------------------------------------------------===#

alias insertion_sort_threshold = 32


@fieldwise_init("implicit")
struct _SortWrapper[T: Copyable & Movable](Copyable, Movable):
    var data: T

    fn __init__(out self, *, other: Self):
        self.data = other.data


@always_inline
fn _insertion_sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin]):
    """Sort the array[start:end] slice"""
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)

    for i in range(1, size):
        var value = array[i]
        var j = i

        # Find the placement of the value in the array, shifting as we try to
        # find the position. Throughout, we assume array[start:i] has already
        # been sorted.
        while j > 0 and cmp_fn(value, array[j - 1]):
            array[j] = array[j - 1]
            j -= 1

        array[j] = value


# put everything thats "<" to the left of pivot
@always_inline
fn _quicksort_partition_right[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin]) -> Int:
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)

    var left = 1
    var right = size - 1
    var pivot_value = array[0]

    while True:
        # no need for left < right since quick sort pick median of 3 as pivot
        while cmp_fn(array[left], pivot_value):
            left += 1
        while left < right and not cmp_fn(array[right], pivot_value):
            right -= 1
        if left >= right:
            var pivot_pos = left - 1
            swap(array[pivot_pos], array[0])
            return pivot_pos
        swap(array[left], array[right])
        left += 1
        right -= 1


# put everything thats "<=" to the left of pivot
@always_inline
fn _quicksort_partition_left[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin]) -> Int:
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)

    var left = 1
    var right = size - 1
    var pivot_value = array[0]

    while True:
        while left < right and not cmp_fn(pivot_value, array[left]):
            left += 1
        while cmp_fn(pivot_value, array[right]):
            right -= 1
        if left >= right:
            var pivot_pos = left - 1
            swap(array[pivot_pos], array[0])
            return pivot_pos
        swap(array[left], array[right])
        left += 1
        right -= 1


fn _heap_sort_fix_down[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin], idx: Int):
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)
    var i = idx
    var j = i * 2 + 1
    while j < size:  # has left child
        # if right child exist and has higher value, swap with right
        if i * 2 + 2 < size and cmp_fn(array[j], array[i * 2 + 2]):
            j = i * 2 + 2
        if not cmp_fn(array[i], array[j]):
            return
        swap(array[j], array[i])
        i = j
        j = i * 2 + 1


@always_inline
fn _heap_sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin]):
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)
    # heapify
    for i in range(size // 2 - 1, -1, -1):
        _heap_sort_fix_down[cmp_fn](span, i)
    # sort
    while size > 1:
        size -= 1
        swap(array[0], array[size])
        _heap_sort_fix_down[cmp_fn](span, 0)


@always_inline
fn _estimate_initial_height(size: Int) -> Int:
    # Compute the log2 of the size rounded upward.
    var log2 = Int(
        (bitwidthof[DType.index]() - 1) ^ count_leading_zeros(size | 1)
    )
    # The number 1.3 was chosen by experimenting the max stack size for random
    # input. This also depends on insertion_sort_threshold
    return max(2, Int(ceil(1.3 * log2)))


@always_inline
fn _delegate_small_sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin]):
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)
    if size == 2:
        _small_sort[2, T, cmp_fn](array)

        return
    if size == 3:
        _small_sort[3, T, cmp_fn](array)
        return

    if size == 4:
        _small_sort[4, T, cmp_fn](array)
        return

    if size == 5:
        _small_sort[5, T, cmp_fn](array)
        return


# FIXME (MSTDL-808): Using _Pair over Span results in 1-3% improvement
# struct _Pair[T: AnyType]:
#     var ptr: UnsafePointer[T]
#     var len: Int


@always_inline
fn _quicksort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin]):
    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var size = len(span)
    if size == 0:
        return

    # Work with an immutable span so we don't run into exclusivity problems with
    # the List[Span].
    alias ImmSpan = span.Immutable

    var stack = List[ImmSpan](capacity=_estimate_initial_height(size))
    stack.append(span)
    while len(stack) > 0:
        var imm_interval = stack.pop()
        var imm_ptr = imm_interval.unsafe_ptr()
        var mut_ptr = imm_ptr.origin_cast[mut=True, origin=MutableAnyOrigin]()
        var len = len(imm_interval)
        var interval = Span[T, MutableAnyOrigin](ptr=mut_ptr, length=len)

        if len <= 5:
            _delegate_small_sort[cmp_fn](interval)
            continue

        if len < insertion_sort_threshold:
            _insertion_sort[cmp_fn](interval)
            continue

        # pick median of 3 as pivot
        _sort3[T, cmp_fn](mut_ptr, len >> 1, 0, len - 1)

        # if ptr[-1] == pivot_value, then everything in between will
        # be the same, so no need to recurse that interval
        # already have array[-1] <= array[0]
        if mut_ptr > array and not cmp_fn(imm_ptr[-1], imm_ptr[0]):
            var pivot = _quicksort_partition_left[cmp_fn](interval)
            if len > pivot + 2:
                stack.append(
                    ImmSpan(ptr=imm_ptr + pivot + 1, length=len - pivot - 1)
                )
            continue

        var pivot = _quicksort_partition_right[cmp_fn](interval)

        if len > pivot + 2:
            stack.append(
                ImmSpan(ptr=imm_ptr + pivot + 1, length=len - pivot - 1)
            )

        if pivot > 1:
            stack.append(ImmSpan(ptr=imm_ptr, length=pivot))


# ===-----------------------------------------------------------------------===#
# stable sort
# ===-----------------------------------------------------------------------===#


fn _merge[
    T: Copyable & Movable,
    span_origin: ImmutableOrigin,
    result_origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](
    span1: Span[T, span_origin],
    span2: Span[T, span_origin],
    result: Span[T, result_origin],
):
    """Merge span1 and span2 into result using the given cmp_fn. The function
    will crash if result is not large enough to hold both span1 and span2.

    Note that if result contains data previously, its destructor will not be called.

    Parameters:
        T: Type of the spans.
        span_origin: Origin of the input spans.
        result_origin: Origin of the result Span.
        cmp_fn: Comparison functor of (type, type) capturing [_] -> Bool type.

    Args:
        span1: The first span to be merged.
        span2: The second span to be merged.
        result: The output span.
    """
    var span1_size = len(span1)
    var span2_size = len(span2)
    var res_ptr = result.unsafe_ptr()

    debug_assert(
        span1_size + span2_size <= len(result),
        "The merge result does not fit in the span provided",
    )
    var i = 0
    var j = 0
    var k = 0
    while i < span1_size:
        if j == span2_size:
            while i < span1_size:
                (res_ptr + k).init_pointee_copy(span1[i])
                k += 1
                i += 1
            return
        if cmp_fn(span2[j], span1[i]):
            (res_ptr + k).init_pointee_copy(span2[j])
            j += 1
        else:
            (res_ptr + k).init_pointee_copy(span1[i])
            i += 1
        k += 1

    while j < span2_size:
        (res_ptr + k).init_pointee_copy(span2[j])
        k += 1
        j += 1


fn _stable_sort_impl[
    T: Copyable & Movable,
    span_life: MutableOrigin,
    tmp_life: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, span_life], temp_buff: Span[T, tmp_life]):
    var size = len(span)
    if size <= 1:
        return
    var i = 0
    while i < size:
        _insertion_sort[cmp_fn](
            span[i : min(i + insertion_sort_threshold, size)]
        )
        i += insertion_sort_threshold
    var merge_size = insertion_sort_threshold
    while merge_size < size:
        var j = 0
        while j + merge_size < size:
            var span1 = span[j : j + merge_size]
            var span2 = span[j + merge_size : min(size, j + 2 * merge_size)]
            _merge[cmp_fn](span1, span2, temp_buff)
            for i in range(merge_size + len(span2)):
                span[j + i] = temp_buff[i]
            j += 2 * merge_size
        merge_size *= 2


fn _stable_sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin]):
    var temp_buff = UnsafePointer[T].alloc(len(span))
    var temp_buff_span = Span(ptr=temp_buff, length=len(span))
    _stable_sort_impl[cmp_fn](span, temp_buff_span)
    temp_buff.free()


# ===-----------------------------------------------------------------------===#
# partition
# ===-----------------------------------------------------------------------===#


@always_inline
fn _partition[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](span: Span[T, origin]) -> Int:
    var size = len(span)
    if size <= 1:
        return 0

    var array = span.unsafe_ptr().origin_cast[origin=MutableAnyOrigin]()
    var pivot = size // 2

    var pivot_value = array[pivot]

    var left = 0
    var right = size - 2

    swap(array[pivot], array[size - 1])

    while left < right:
        if cmp_fn(array[left], pivot_value):
            left += 1
        elif not cmp_fn(array[right], pivot_value):
            right -= 1
        else:
            swap(array[left], array[right])

    if cmp_fn(array[right], pivot_value):
        right += 1
    swap(array[size - 1], array[right])
    return right


fn _partition[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](var span: Span[T, origin], var k: Int):
    while True:
        var pivot = _partition[cmp_fn](span)
        if pivot == k:
            return
        elif k < pivot:
            span._len = pivot
            span = span[:pivot]
        else:
            span._data += pivot + 1
            span._len -= pivot + 1
            k -= pivot + 1


fn partition[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
](span: Span[T, origin], k: Int):
    """Partition the input buffer inplace such that first k elements are the
    largest (or smallest if cmp_fn is < operator) elements.
    The ordering of the first k elements is undefined.

    Parameters:
        T: Type of the underlying data.
        origin: Origin of span.
        cmp_fn: Comparison functor of (T, T) capturing [_] -> Bool type.

    Args:
        span: Input buffer.
        k: Index of the partition element.
    """

    @parameter
    fn _cmp_fn(lhs: _SortWrapper[T], rhs: _SortWrapper[T]) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _partition[_cmp_fn](span, k)


# ===-----------------------------------------------------------------------===#
# sort
# ===-----------------------------------------------------------------------===#


# Junction from public to private API
fn _sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
    *,
    stable: Bool = False,
](span: Span[T, origin]):
    if len(span) <= 5:
        _delegate_small_sort[cmp_fn](span)
        return

    if len(span) < insertion_sort_threshold:
        _insertion_sort[cmp_fn](span)
        return

    @parameter
    if stable:
        _stable_sort[cmp_fn](span)
    else:
        _quicksort[cmp_fn](span)


# TODO (MSTDL-766): The Int and Scalar[T] overload should be remove
# (same for partition)
# Eventually we want a sort that takes a Span and one that takes a List with
# optional cmp_fn.
fn sort[
    T: Copyable & Movable,
    origin: MutableOrigin, //,
    cmp_fn: fn (T, T) capturing [_] -> Bool,
    *,
    stable: Bool = False,
](span: Span[T, origin]):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        T: Copyable & Movable type of the underlying data.
        origin: Origin of span.
        cmp_fn: The comparison function.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
    """

    @parameter
    fn _cmp_fn(lhs: _SortWrapper[T], rhs: _SortWrapper[T]) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _sort[_cmp_fn, stable=stable](span)


fn sort[
    origin: MutableOrigin, //,
    cmp_fn: fn (Int, Int) capturing [_] -> Bool,
    *,
    stable: Bool = False,
](span: Span[Int, origin]):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        origin: Origin of span.
        cmp_fn: The comparison function.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
    """

    @parameter
    fn _cmp_fn(lhs: _SortWrapper[Int], rhs: _SortWrapper[Int]) -> Bool:
        return cmp_fn(lhs.data, rhs.data)

    _sort[_cmp_fn, stable=stable](span)


fn sort[
    origin: MutableOrigin, //,
    *,
    stable: Bool = False,
](span: Span[Int, origin]):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        origin: Origin of span.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
    """

    @parameter
    fn _cmp_fn(lhs: Int, rhs: Int) -> Bool:
        return lhs < rhs

    sort[_cmp_fn, stable=stable](span)


fn sort[
    dtype: DType,
    origin: MutableOrigin, //,
    *,
    stable: Bool = False,
](span: Span[Scalar[dtype], origin]):
    """Sort the list inplace.
    The function doesn't return anything, the list is updated inplace.

    Parameters:
        dtype: Copyable & Movable type of the underlying data.
        origin: Origin of span.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
    """

    @parameter
    fn _cmp_fn(lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Bool:
        return lhs < rhs

    sort[_cmp_fn, stable=stable](span)


fn sort[
    T: Copyable & Movable & Comparable,
    origin: MutableOrigin, //,
    *,
    stable: Bool = False,
](span: Span[T, origin]):
    """Sort list of the order comparable elements in-place.

    Parameters:
        T: The order comparable collection element type.
        origin: Origin of span.
        stable: Whether the sort should be stable.

    Args:
        span: The span to be sorted.
    """

    @parameter
    fn _cmp_fn(a: T, b: T) -> Bool:
        return a < b

    sort[_cmp_fn, stable=stable](span)


# ===-----------------------------------------------------------------------===#
# sort networks
# ===-----------------------------------------------------------------------===#


@always_inline
fn _sort2[
    T: Copyable & Movable,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](
    array: UnsafePointer[
        T, address_space = AddressSpace.GENERIC, mut=True, **_
    ],
    offset0: Int,
    offset1: Int,
):
    var a = array[offset0]
    var b = array[offset1]
    if not cmp_fn(a, b):
        array[offset0] = b
        array[offset1] = a


@always_inline
fn _sort3[
    T: Copyable & Movable,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](
    array: UnsafePointer[
        T, address_space = AddressSpace.GENERIC, mut=True, **_
    ],
    offset0: Int,
    offset1: Int,
    offset2: Int,
):
    _sort2[T, cmp_fn](array, offset0, offset1)
    _sort2[T, cmp_fn](array, offset1, offset2)
    _sort2[T, cmp_fn](array, offset0, offset1)


@always_inline
fn _sort_partial_3[
    T: Copyable & Movable,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](
    array: UnsafePointer[
        T, address_space = AddressSpace.GENERIC, mut=True, **_
    ],
    offset0: Int,
    offset1: Int,
    offset2: Int,
):
    var a = array[offset0]
    var b = array[offset1]
    var c = array[offset2]
    var r = cmp_fn(c, a)
    var t = c if r else a
    if r:
        array[offset2] = a
    if cmp_fn(b, t):
        array[offset0] = b
        array[offset1] = t
    elif r:
        array[offset0] = t


@always_inline
fn _small_sort[
    n: Int,
    T: Copyable & Movable,
    cmp_fn: fn (_SortWrapper[T], _SortWrapper[T]) capturing [_] -> Bool,
](array: UnsafePointer[T, address_space = AddressSpace.GENERIC, mut=True, **_]):
    @parameter
    if n == 2:
        _sort2[T, cmp_fn](array, 0, 1)
        return

    @parameter
    if n == 3:
        _sort2[T, cmp_fn](array, 1, 2)
        _sort_partial_3[T, cmp_fn](array, 0, 1, 2)
        return

    @parameter
    if n == 4:
        _sort2[T, cmp_fn](array, 0, 2)
        _sort2[T, cmp_fn](array, 1, 3)
        _sort2[T, cmp_fn](array, 0, 1)
        _sort2[T, cmp_fn](array, 2, 3)
        _sort2[T, cmp_fn](array, 1, 2)
        return

    @parameter
    if n == 5:
        _sort2[T, cmp_fn](array, 0, 1)
        _sort2[T, cmp_fn](array, 3, 4)
        _sort_partial_3[T, cmp_fn](array, 2, 3, 4)
        _sort2[T, cmp_fn](array, 1, 4)
        _sort_partial_3[T, cmp_fn](array, 0, 2, 3)
        _sort_partial_3[T, cmp_fn](array, 1, 2, 3)
        return
