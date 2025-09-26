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

"""
Random number generation for GPU kernels.

This module implements a high-performance random number generator using the Philox algorithm,
which is designed for parallel and GPU computing. The Philox algorithm is a counter-based
random number generator that provides high-quality random numbers with excellent statistical
properties.

The main class is Random which generates both uniform random numbers and raw 32-bit integers.
It supports:
- Seeding for reproducible sequences
- Multiple independent subsequences
- Configurable number of rounds for quality vs performance tradeoff
- Vectorized operations for efficiency

Example:

```mojo
from gpu.random import Random
    rng = Random(seed=42)
    uniform_values = rng.step_uniform()  # Returns 4 random floats in [0,1)
    raw_values = rng.step()  # Returns 4 raw 32-bit integers
```
"""

from math import cos, log, sin, sqrt

from memory import bitcast

from .intrinsics import mulwide


fn _mulhilow(a: UInt32, b: UInt32) -> SIMD[DType.uint32, 2]:
    var res = mulwide(a, b)
    return bitcast[DType.uint32, 2](res)


struct Random[rounds: Int = 6]:
    """A high-performance random number generator using the Philox algorithm.

    The Philox algorithm is a counter-based random number generator designed for parallel
    and GPU computing. It provides high-quality random numbers with excellent statistical properties.

    Parameters:
        rounds: Number of mixing rounds to perform. Higher values provide better statistical
               quality at the cost of performance. Default is 6.
    """

    var _key: SIMD[DType.uint32, 2]
    var _counter: SIMD[DType.uint32, 4]

    fn __init__(
        out self,
        *,
        seed: UInt64 = 0,
        subsequence: UInt64 = 0,
        offset: UInt64 = 0,
    ):
        """Initialize the random number generator.

        Args:
            seed: Initial seed value for reproducible sequences. Default is 0.
            subsequence: Subsequence number for generating independent streams. Default is 0.
            offset: Starting offset in the sequence. Default is 0.
        """
        self._key = bitcast[DType.uint32, 2](seed)
        self._counter = bitcast[DType.uint32, 4](
            SIMD[DType.uint64, 2](offset, subsequence)
        )

    @always_inline
    fn step(mut self) -> SIMD[DType.uint32, 4]:
        """Generate 4 random 32-bit unsigned integers.

        Returns:
            SIMD vector containing 4 random 32-bit unsigned integers.
        """
        alias K_PHILOX_10 = SIMD[DType.uint32, 2](0x9E3779B9, 0xBB67AE85)

        @parameter
        for i in range(rounds):
            self._counter = self._single_round(self._counter, self._key)
            self._key += K_PHILOX_10
        return self._single_round(self._counter, self._key)

    @always_inline
    fn step_uniform(mut self) -> SIMD[DType.float32, 4]:
        """Generate 4 random floating point numbers uniformly distributed in [0,1).

        Returns:
            SIMD vector containing 4 random float32 values in range [0,1).
        """
        # The inverse of 2^32
        alias INV_2_32 = 2.3283064e-10
        return self.step().cast[DType.float32]() * INV_2_32

    @always_inline
    fn _incrn(mut self, n: Int64):
        """Increment the internal counter by n.

        Args:
            n: Amount to increment the counter by.
        """
        var hilo = bitcast[DType.uint32, 2](n)
        var hi = hilo[0]
        var lo = hilo[1]

        self._counter[0] += lo
        if self._counter[0] < lo:
            hi += 1
        self._counter[1] += hi
        if hi <= self._counter[1]:
            return
        self._counter[2] += 1
        if self._counter[2]:
            return
        self._counter[3] += 1

    @always_inline
    @staticmethod
    fn _single_round(
        counter: SIMD[DType.uint32, 4], key: SIMD[DType.uint32, 2]
    ) -> SIMD[DType.uint32, 4]:
        """Perform a single round of the Philox mixing function.

        Args:
            counter: Current counter state as 4 32-bit values.
            key: Current key state as 2 32-bit values.

        Returns:
            Mixed output as 4 32-bit values.
        """
        alias K_PHILOX_SA = 0xD2511F53
        alias K_PHILOX_SB = 0xCD9E8D57

        var res0 = _mulhilow(K_PHILOX_SA, counter[0])
        var res1 = _mulhilow(K_PHILOX_SB, counter[3])
        return SIMD[DType.uint32, 4](
            res1[1] ^ counter[1] ^ key[0],
            res1[0],
            res0[1] ^ counter[3] ^ key[1],
            res0[0],
        )


struct NormalRandom[rounds: Int = 6]:
    """A high-performance random number generator using the Box-Muller transform.

    The Box-Muller transform is a method for generating pairs of independent standard normal random variables.

    Parameters:
        rounds: Number of mixing rounds to perform for the underlying random uniform generator that serves as
               input to the Box-Muller transform. Higher values provide better statistical quality at the cost of
               performance. Default is 6.
    """

    var _rng: Random[rounds]

    fn __init__(
        out self,
        *,
        seed: UInt64 = 0,
        subsequence: UInt64 = 0,
        offset: UInt64 = 0,
    ):
        self._rng = Random[rounds](
            seed=seed, subsequence=subsequence, offset=offset
        )

    fn step_normal(
        mut self, mean: Float32 = 0.0, stddev: Float32 = 1.0
    ) -> SIMD[DType.float32, 8]:
        """Generate 8 normally distributed random numbers using Box-Muller transform.

        Returns:
            SIMD vector containing 8 random float32 values from a normal distribution with mean `mean` and standard deviation `stddev`.
        """
        var u1 = self._rng.step_uniform()
        var u2 = self._rng.step_uniform()

        var epsilon = SIMD[DType.float32, 4](1e-7)
        var safe_u1 = max(u1, epsilon)

        var r = sqrt(-2.0 * log(safe_u1))
        var theta = 2.0 * math.pi * u2
        var z0 = r * cos(theta)
        var z1 = r * sin(theta)

        # Scale and shift both sets.
        z0 = z0 * stddev + mean
        z1 = z1 * stddev + mean

        # Combine z0 and z1 into a single SIMD[DType.float32, 8]
        return z0.join(z1)
