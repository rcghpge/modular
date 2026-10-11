# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

"""Pytest plugin that names the build-time eager warm store for this process.

Runs as the plugin loads, before any test imports ``max._interpreter_ops``,
whose families load from the store the moment they first need a model.
"""

from __future__ import annotations

import os

from python.runfiles import runfiles


def _store_directories(rlocations: list[str]) -> list[str]:
    resolver = runfiles.Create()
    if resolver is None:
        raise RuntimeError("the eager warm store needs runfiles")
    directories = []
    for rlocation in rlocations:
        resolved = resolver.Rlocation(rlocation)
        if resolved is None or not os.path.isdir(resolved):
            raise RuntimeError(f"eager warm store missing at {rlocation!r}")
        directories.append(resolved)
    return directories


_RLOCATIONS = os.environ.get("XARCH_WARM_RLOCATIONS", "").split()
if _RLOCATIONS:
    os.environ["MAX_EAGER_WARM_STORE"] = os.pathsep.join(
        _store_directories(_RLOCATIONS)
    )
