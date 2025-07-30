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
# RUN: %mojo-no-debug -D MLIRC_DYLIB=.graph_lib %s

# Test the MLIR rewriter C API

import _mlir
from _mlir import Context, Module, Operation, Rewriter
from testing import assert_equal


fn main() raises:
    with Context() as ctx:
        ctx.allow_unregistered_dialects()
        var loc = _mlir.Location.unknown(ctx)
        var module = Module(loc)
        assert_equal("module {\n}\n", String(module))

        var rewriter = Rewriter(ctx)
        rewriter.set_insertion_point_to_start(module.body())
        var new_op = rewriter.insert(Operation("d.new_op", loc))
        var new_module_str = """module {
  "d.new_op"() : () -> ()
}\n"""
        assert_equal(new_module_str, String(module))

        var replacing_op = rewriter.insert(Operation("d.replacing_op", loc))
        rewriter.replace_op_with(new_op, replacing_op)

        var new_module_str2 = """module {
  "d.replacing_op"() : () -> ()
}\n"""
        assert_equal(new_module_str2, String(module))

        # Right now the lifetime of `module` is poorly defined.
        # This `destroy()` is just a temp. workaround so
        # ASAN does not complain (and can therefore catch realer bugs)
        module.destroy()
