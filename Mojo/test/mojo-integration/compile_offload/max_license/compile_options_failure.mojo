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

# RUN: not kgen -elaborate %s -o /dev/null 2>&1 | FileCheck %s

from std.collections.string.string_span import StaticString, _get_kgen_string
from max.gpu.host.info import A100
from std.sys.info import CompilationTarget


struct _Info(TrivialRegisterPassable):
    var kernel: __mlir_type.`!kgen.string`
    var name: __mlir_type.`!kgen.string`
    var num_captures: __mlir_type.index


@fieldwise_init
struct Info(TrivialRegisterPassable):
    var kernel: StaticString
    var name: StaticString
    var num_captures: Int


@always_inline
def _compile_info[
    func_type: TrivialRegisterPassable,
    //,
    func: func_type,
    /,
    emission_kind: Int = 0,
    target: CompilationTarget = A100.target,
    compile_options: StaticString = "i_am_a_bad_option=true",
]() -> Info:
    var info = __mlir_op.`kgen.compile_offload`[
        target_type=target._mlir_value,
        emission_kind=emission_kind.__mlir_index__(),
        emission_option=_get_kgen_string[compile_options](),
        emission_link_option=_get_kgen_string[""](),
        func=func,
        _type=_Info,
    ]()

    return Info(
        StaticString(info.kernel),
        StaticString(info.name),
        Int(mlir_value=info.num_captures),
    )


def hello():
    pass


def world():
    pass


def main():
    # CHECK: emission option "i_am_a_bad_option" is not found
    var t1 = _compile_info[hello]()
    print(t1.kernel)
