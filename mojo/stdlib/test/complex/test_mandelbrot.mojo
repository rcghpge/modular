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
#
# This uses mandelbrot as an example to test how the entire stdlib works
# together.
#
# ===----------------------------------------------------------------------=== #

from complex import ComplexScalar, ComplexFloat32, ComplexSIMD
from testing import assert_equal

# NOTE: This is commented out because TestSuite is part of `test_utils` which
# is not packaged with the stdlib.
# from test_utils import TestSuite


fn mandelbrot_iter(row: Int, col: Int) -> Int:
    alias height = 375
    alias width = 500

    var xRange: Float32 = 2.0
    var yRange: Float32 = 1.5
    var minX = 0.5 - xRange
    var maxY = -0.5 + yRange

    var c = ComplexFloat32(
        minX + col * xRange / width, maxY - row * yRange / height
    )

    var z = c

    var iter: Int = 0
    for _i in range(10):
        iter += 1
        z = z * z + c
        if z.squared_norm() > 4:
            return iter
    return iter


def test_mandelbrot_iter():
    assert_equal(mandelbrot_iter(0, 0), 1)
    assert_equal(mandelbrot_iter(0, 1), 1)
    assert_equal(mandelbrot_iter(50, 50), 2)
    assert_equal(mandelbrot_iter(100, 100), 3)

    var z = ComplexScalar[DType.int32](re=Int32(3), im=Int32(4))
    assert_equal(z.squared_norm(), 25)


def main():
    test_mandelbrot_iter()

    # NOTE: We need to print this for the SDK self test.
    print("Mandelbrot passed")
