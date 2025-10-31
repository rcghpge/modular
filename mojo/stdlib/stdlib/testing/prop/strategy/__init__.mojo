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

from testing.prop.random import Rng
from .types import *


trait Strategy(Movable):
    """A type used to produce random inputs for property tests.

    Strategies are a core building block of property testing. They are used to
    produce the random input values for the properties being tested.
    """

    # TODO: `Value` should require `Representable/Stringable` once conditional
    # conformance is supported.
    alias Value: Copyable & Movable
    """The type the strategy produces."""

    fn value(mut self, mut rng: Rng) raises -> Self.Value:
        """Produces a random value using this strategy.

        Args:
            rng: The random number generator to use.

        Returns:
            A random value.

        Raises:
            If the underlying `Rng` raises an error.
        """
        ...
