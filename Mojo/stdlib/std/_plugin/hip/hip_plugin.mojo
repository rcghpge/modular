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

from std._plugin._trait import (
    PluginHooks,
    StdPlugin,
    _UnaryMathPluginHookFnType,
)
from std.collections import OptionalReg
from std.sys.info import CompilationTarget
from std.collections.string.string_span import _get_kgen_string

from .hip_math import _amdgcn_unary_math, _has_amdgcn_op


struct HIPPlugin(StdPlugin):
    """`PluginHooks` implementation for AMD HIP backends."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["hip"]()
    comptime HookFor[target: CompilationTarget] = HIPPluginHooks[target]


struct HIPPluginHooks[target: CompilationTarget](PluginHooks):
    comptime unary_math_fn[
        name: StaticString, dtype: DType, width: Int
    ]: OptionalReg[_UnaryMathPluginHookFnType] = (
        _amdgcn_unary_math[name=name] if _has_amdgcn_op[name, dtype]() else None
    )
