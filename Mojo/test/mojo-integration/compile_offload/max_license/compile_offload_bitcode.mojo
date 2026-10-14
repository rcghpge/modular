# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the MAX License:
# https://modular.com/legal/max-license
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug -D TMP_FILE="%t.bc" %s
# RUN: llvm-dis %t.bc -o - | FileCheck %s

from std.collections.string.string_span import StaticString, _get_kgen_string
from max.gpu.host.info import A100
from std.sys.info import CompilationTarget
from std.sys import get_defined_string


@fieldwise_init
struct _Info(TrivialRegisterPassable):
    var kernel: __mlir_type.`!kgen.string`
    var name: __mlir_type.`!kgen.string`
    var num_captures: __mlir_type.index
    var capture_sizes: __mlir_type.`!kgen.pointer<ui64>`


@fieldwise_init
struct Info(TrivialRegisterPassable):
    var kernel: StaticString
    var name: StaticString
    var num_captures: Int

    @__allow_legacy_any_origin_fields
    var capture_sizes: Pointer[__mlir_type.ui64, ImmUnsafeAnyOrigin]


@always_inline
def _compile_info[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    /,
    emission_kind: Int = 0,
    target: CompilationTarget = A100.target,
    compile_options: StaticString = target.default_compile_options(),
]() -> Info:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=target._mlir_value,
        emission_kind=emission_kind.__mlir_index__(),
        emission_option=_get_kgen_string[compile_options](),
        emission_link_option=_get_kgen_string[""](),
        func=func,
        _type=_Info,
    ]()
    var num_captures_ptr_ui64 = Pointer[__mlir_type.ui64, ImmUnsafeAnyOrigin](
        _mlir_value=__mlir_op.`pop.pointer.bitcast`[
            _type=Pointer[__mlir_type.ui64, ImmUnsafeAnyOrigin]._mlir_type,
        ](info.capture_sizes)
    )

    return Info(
        StaticString(info.kernel),
        StaticString(info.name),
        Int(mlir_value=info.num_captures),
        num_captures_ptr_ui64,
    )


def hello():
    pass


def main() raises:
    comptime tmp_file = get_defined_string["TMP_FILE"]()
    # CHECK: target datalayout = "{{.*}}"
    # CHECK: target triple = "{{.*}}"
    var t6 = _compile_info[
        hello, emission_kind=4, target=CompilationTarget.current()
    ]()
    var f = open(tmp_file, "w")
    print(t6.kernel, sep="", end="", flush=True, file=RawFileDescriptor(f))
    f.close()
