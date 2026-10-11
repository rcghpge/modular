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

# RUN: %parse-mojo-isolated %s | FileCheck %s

##===----------------------------------------------------------------------===##
# List pattern lvalue tests
##===----------------------------------------------------------------------===##


# CHECK-LABEL: lit.fn @"lists_lv
def lists_lv(x: List[Int]) raises ListLengthError:
    # CHECK: %b = lit.var.decl "b"
    var b: Int

    # CHECK: [[MIN0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: [[EXACT0:%.*]] = kgen.param.constant: !Bool = <{:scalar<bool> true}>
    # CHECK: lit.call {{.*}}@"check_list_length{{.*}}(%x, [[MIN0]], [[EXACT0]],
    # CHECK: [[C0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK: [[G0:%.*]] = lit.call {{.*}}@List::@"__getitem__{{.*}}(%x, [[C0]])
    # CHECK: [[G0R:%.*]] = kgen.rebind [[G0]]
    # CHECK: %a = lit.var.decl "a" var
    # CHECK: [[A:%.*]] = lit.ref.load [[G0R]]
    # CHECK: lit.ref.store [[A]], %a
    # CHECK: [[C1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK: [[G1:%.*]] = lit.call {{.*}}@List::@"__getitem__{{.*}}(%x, [[C1]])
    # CHECK: [[G1R:%.*]] = kgen.rebind [[G1]]
    # CHECK: [[B:%.*]] = lit.ref.load [[G1R]]
    # CHECK: lit.ref.store [[B]], %b
    # FIXME: Precedence requires parens here which is untidy.
    [(var a), b] = x

    # CHECK: [[MIN1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: [[EXACT1:%.*]] = kgen.param.constant: !Bool = <{:scalar<bool> true}>
    # CHECK: lit.call {{.*}}@"check_list_length{{.*}}(%x, [[MIN1]], [[EXACT1]],
    # CHECK: [[C0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK: [[G0:%.*]] = lit.call {{.*}}@List::@"__getitem__{{.*}}(%x, [[C0]])
    # CHECK: [[G0R:%.*]] = kgen.rebind [[G0]]
    # CHECK: %c = lit.var.decl "c" var
    # CHECK: [[C:%.*]] = lit.ref.load [[G0R]]
    # CHECK: lit.ref.store [[C]], %c
    # CHECK: [[C1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK: [[G1:%.*]] = lit.call {{.*}}@List::@"__getitem__{{.*}}(%x, [[C1]])
    # CHECK: [[G1R:%.*]] = kgen.rebind [[G1]]
    # CHECK: %d = lit.var.decl "d" var
    # CHECK: [[D:%.*]] = lit.ref.load [[G1R]]
    # CHECK: lit.ref.store [[D]], %d
    var [c, d] = x

    # CHECK: [[MIN2:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 2}>
    # CHECK-NEXT: [[EXACT2:%.*]] = kgen.param.constant: !Bool = <{:scalar<bool> true}>
    # CHECK: lit.call {{.*}}@"check_list_length{{.*}}(%x, [[MIN2]], [[EXACT2]],
    # CHECK: [[C0:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 0}>
    # CHECK: [[G0:%.*]] = lit.call {{.*}}@List::@"__getitem__{{.*}}(%x, [[C0]])
    # CHECK: [[G0R:%.*]] = kgen.rebind [[G0]]
    # CHECK: %e = lit.var.decl "e" ref
    # CHECK: lit.ref.store [[G0R]], %e
    # CHECK: [[C1:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK: [[G1:%.*]] = lit.call {{.*}}@List::@"__getitem__{{.*}}(%x, [[C1]])
    # CHECK: [[G1R:%.*]] = kgen.rebind [[G1]]
    # CHECK: %f = lit.var.decl "f" ref
    # CHECK: lit.ref.store [[G1R]], %f
    ref [e, f] = x


# CHECK-LABEL: lit.fn @"list_length_match
def list_length_match(x: List[Int]) -> Bool:
    # CHECK: [[MIN:%.*]] = kgen.param.constant: !Int = <{:scalar<index> 1}>
    # CHECK-NEXT: [[INEXACT:%.*]] = kgen.param.constant: !Bool = <{:scalar<bool> false}>
    # CHECK: lit.call {{.*}}@"list_length_matches{{.*}}(%x, [[MIN]], [[INEXACT]])
    return list_length_matches(x, 1, False)


# CHECK-LABEL: lit.fn @"array_length_match
def array_length_match(x: Array[Int, 2]) -> Bool:
    # CHECK: lit.call {{.*}}@"list_length_matches
    return list_length_matches(x, 1, False)
