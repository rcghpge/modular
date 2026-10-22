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
"""Selects the active `MaxPluginHooks` for a target.

A target's `opaque_plugin` payload is consulted first: the first type in it
that conforms to `MaxPlugin` is the target's MAX plugin, so a plugin that
ships with its own target table needs no registry entry here. A target whose
payload names no MAX plugin falls back to `MAX_PLUGINS`, the name-keyed list
in `_overlay.mojo`, matched by the name of the stdlib plugin selected for
the target.

The name list is transitional. It exists only because the builtin target
tables live in the stdlib, which cannot name MAX plugin types, so the
NVIDIA, AMD and Apple plugins reach their targets by name instead. The
intended end state is that every target, builtin or not, carries its MAX
plugin in its payload and the name list goes away.
"""

from ._overlay import MAX_PLUGINS
from ._trait import DefaultMaxPluginHooks, MaxPlugin, MaxPluginHooks

from std.collections.string.string_span import _get_kgen_string
from std.sys.info import CompilationTarget


struct _NoMaxPlugin(MaxPlugin):
    """Stands for a payload that names no MAX plugin.

    `DefaultMaxPlugin` can't play this role: a payload that names it keeps it,
    as a payload that names `DefaultPlugin` does in the stdlib.
    """

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["<none>"]()
    comptime HookFor[target: CompilationTarget] = DefaultMaxPluginHooks[target]


comptime _payload_plugin[target: CompilationTarget]: MaxPlugin = (
    target._first_conforming_plugin[MaxPlugin, default=_NoMaxPlugin]
)
"""The MAX plugin named in `target`'s payload, or `_NoMaxPlugin`."""

comptime _max_plugin[target: CompilationTarget]: MaxPlugin = (
    MAX_PLUGINS.for_target[target._mlir_value] if _payload_plugin[target]
    == _NoMaxPlugin else _payload_plugin[target]
)
"""The MAX plugin selected for `target`, before specialization."""

comptime MaxPluginForTarget[target: CompilationTarget] = _max_plugin[
    target
].HookFor[target]
"""The MAX plugin selected for the specified target."""

comptime CurrentMaxPlugin: MaxPluginHooks = MaxPluginForTarget[
    CompilationTarget.current()
]
"""The active MAX plugin."""
