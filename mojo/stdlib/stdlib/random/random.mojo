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
"""Provides functions for random numbers.

You can import these APIs from the `random` package. For example:

```mojo
from random import seed
```
"""

import math
from math import floor
from sys import external_call
from time import perf_counter_ns


fn _get_random_state() -> OpaquePointer:
    return external_call[
        "KGEN_CompilerRT_GetRandomState",
        OpaquePointer,
    ]()


fn seed():
    """Seeds the random number generator using the current time."""
    seed(Int(perf_counter_ns()))


fn seed(a: Int):
    """Seeds the random number generator using the value provided.

    Args:
        a: The seed value.
    """
    external_call["KGEN_CompilerRT_SetRandomStateSeed", NoneType](
        _get_random_state(), a
    )


fn random_float64(min: Float64 = 0, max: Float64 = 1) -> Float64:
    """Returns a random `Float64` number from the given range.

    Args:
        min: The minimum number in the range (default is 0.0).
        max: The maximum number in the range (default is 1.0).

    Returns:
        A random number from the specified range.
    """
    return external_call["KGEN_CompilerRT_RandomDouble", Float64](min, max)


fn random_si64(min: Int64, max: Int64) -> Int64:
    """Returns a random `Int64` number from the given range.

    Args:
        min: The minimum number in the range.
        max: The maximum number in the range.

    Returns:
        A random number from the specified range.
    """
    return external_call["KGEN_CompilerRT_RandomSInt64", Int64](min, max)


fn random_ui64(min: UInt64, max: UInt64) -> UInt64:
    """Returns a random `UInt64` number from the given range.

    Args:
        min: The minimum number in the range.
        max: The maximum number in the range.

    Returns:
        A random number from the specified range.
    """
    return external_call["KGEN_CompilerRT_RandomUInt64", UInt64](min, max)


fn randint[
    dtype: DType
](ptr: UnsafePointer[Scalar[dtype]], size: Int, low: Int, high: Int):
    """Fills memory with uniform random in range [low, high].

    Constraints:
        The type should be integral.

    Parameters:
        dtype: The dtype of the pointer.

    Args:
        ptr: The pointer to the memory area to fill.
        size: The number of elements to fill.
        low: The minimal value for random.
        high: The maximal value for random.
    """
    constrained[dtype.is_integral(), "dtype must be integral"]()

    @parameter
    if dtype.is_signed():
        for si in range(size):
            ptr[si] = random_si64(low, high).cast[dtype]()
    else:
        for ui in range(size):
            ptr[ui] = random_ui64(low, high).cast[dtype]()


fn rand[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], mut=True, **_],
    size: Int,
    /,
    *,
    min: Float64 = 0.0,
    max: Float64 = 1.0,
    int_scale: Optional[Int] = None,
):
    """Fills memory with random values from a uniform distribution.

    Parameters:
        dtype: The dtype of the pointer.

    Args:
        ptr: The pointer to the memory area to fill.
        size: The number of elements to fill.
        min: The minimum value for random.
        max: The maximum value for random.
        int_scale: The scale for error checking (float type only).
    """
    var scale_val = int_scale.or_else(-1)

    @parameter
    if dtype.is_floating_point():
        if scale_val >= 0:
            var scale_double: Float64 = 1 << scale_val
            for i in range(size):
                var rnd = random_float64(min, max)
                ptr[i] = (floor(rnd * scale_double) / scale_double).cast[
                    dtype
                ]()
        else:
            for i in range(size):
                var rnd = random_float64(min, max)
                ptr[i] = rnd.cast[dtype]()

        return

    @parameter
    if dtype.is_signed():
        var min_ = math.max(
            Scalar[dtype].MIN.cast[DType.int64](), min.cast[DType.int64]()
        )
        var max_ = math.min(
            max.cast[DType.int64](), Scalar[dtype].MAX.cast[DType.int64]()
        )
        for i in range(size):
            ptr[i] = random_si64(min_, max_).cast[dtype]()
        return

    @parameter
    if dtype is DType.bool or dtype.is_unsigned():
        var min_ = math.max(0, min.cast[DType.uint64]())
        var max_ = math.min(
            max.cast[DType.uint64](), Scalar[dtype].MAX.cast[DType.uint64]()
        )
        for i in range(size):
            ptr[i] = random_ui64(min_, max_).cast[dtype]()
        return


fn randn_float64(
    mean: Float64 = 0.0, standard_deviation: Float64 = 1.0
) -> Float64:
    """Returns a random double sampled from a Normal(mean, standard_deviation) distribution.

    Args:
        mean: Normal distribution mean.
        standard_deviation: Normal distribution standard deviation.

    Returns:
        A random float64 sampled from Normal(mean, standard_deviation).
    """
    return external_call["KGEN_CompilerRT_NormalDouble", Float64](
        mean, standard_deviation
    )


fn randn[
    dtype: DType
](
    ptr: UnsafePointer[Scalar[dtype], mut=True, **_],
    size: Int,
    mean: Float64 = 0.0,
    standard_deviation: Float64 = 1.0,
):
    """Fills memory with random values from a Normal(mean, standard_deviation) distribution.

    Constraints:
        The type should be floating point.

    Parameters:
        dtype: The dtype of the pointer.

    Args:
        ptr: The pointer to the memory area to fill.
        size: The number of elements to fill.
        mean: Normal distribution mean.
        standard_deviation: Normal distribution standard deviation.
    """

    for i in range(size):
        ptr[i] = randn_float64(mean, standard_deviation).cast[dtype]()
    return


fn shuffle[T: Copyable & Movable, //](mut list: List[T]):
    """Shuffles the elements of the list randomly.

    Performs an in-place Fisher-Yates shuffle on the provided list.

    Args:
        list: The list to modify.

    Parameters:
        T: The type of element in the List.
    """
    for i in reversed(range(len(list))):
        var j = Int(random_ui64(0, i))
        list.swap_elements(i, j)
