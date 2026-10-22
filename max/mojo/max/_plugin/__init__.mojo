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
"""Compile-time hooks for target-specific MAX behavior.

`CurrentMaxPlugin` is the `MaxPluginHooks` of the current compilation
target, and `MaxPluginForTarget[target]` that of any other. A target is
matched to its plugin by its `opaque_plugin` payload first, then by the name
of its stdlib plugin; see `_impl.mojo` for the rule.
"""

from ._impl import CurrentMaxPlugin, MaxPluginForTarget
from ._trait import DefaultMaxPlugin, MaxPlugin, MaxPluginHooks
