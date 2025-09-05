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
"""This module includes the inlined_assembly function."""

from collections.string.string_slice import _get_kgen_string

from .intrinsics import _mlirtype_is_eq


@always_inline("nodebug")
fn inlined_assembly[
    asm: StaticString,
    result_type: AnyTrivialRegType,
    *types: AnyType,
    constraints: StaticString,
    has_side_effect: Bool = True,
](*args: *types) -> result_type:
    """Generates assembly via inline assembly."""
    var loaded_pack = args.get_loaded_kgen_pack()

    alias asm_kgen_string = _get_kgen_string[asm]()
    alias constraints_kgen_string = _get_kgen_string[constraints]()

    @parameter
    if _mlirtype_is_eq[result_type, NoneType]():
        __mlir_op.`pop.inline_asm`[
            _type=None,
            assembly=asm_kgen_string,
            constraints=constraints_kgen_string,
            hasSideEffects = has_side_effect._mlir_value,
        ](loaded_pack)
        return rebind[result_type](None)
    else:
        return __mlir_op.`pop.inline_asm`[
            _type=result_type,
            assembly=asm_kgen_string,
            constraints=constraints_kgen_string,
            hasSideEffects = has_side_effect._mlir_value,
        ](loaded_pack)
