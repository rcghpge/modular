# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Test eager tensors basic behavior."""

import numpy as np
import pytest
from hypothesis import given, settings, Phase
from hypothesis.extra import numpy as nst

from max.experimental.eager import Tensor


@given(
    na=nst.arrays(dtype=np.float32, shape=(3, 1)),
    nb=nst.arrays(dtype=np.float32, shape=(3, 1)),
    nc=nst.arrays(dtype=np.float32, shape=(1, 3)),
    nd=nst.arrays(dtype=np.float32, shape=(1, 3)),
)
# currently slooooow
@settings(max_examples=5, phases=[Phase.generate], deadline=None)
@pytest.mark.asyncio
async def test_basic_add(na, nb, nc, nd):
    a, b, c, d = (Tensor.from_numpy(n) for n in (na, nb, nc, nd))
    expected = (na + nb) + (nc + nd)
    value = await ((a + b) + (c + d))
    result = value.to_numpy()
    np.testing.assert_allclose(result, expected)


@given(
    na=nst.arrays(dtype=np.float32, shape=(3, 2)),
    nb=nst.arrays(dtype=np.float32, shape=(2, 3)),
)
def test_eager_shape_failure(na, nb):
    a, b = Tensor.from_numpy(na), Tensor.from_numpy(nb)
    with pytest.raises(ValueError):
        _ = a + b
