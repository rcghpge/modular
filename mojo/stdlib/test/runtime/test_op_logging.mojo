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
# RUN: %mojo-no-debug -D LOGGING_LEVEL=trace %s 2>&1 | FileCheck %s


from runtime.tracing import Trace, TraceLevel
from collections.optional import OptionalReg


def test_op_logging[
    level: TraceLevel, target: Optional[StaticString] = None
](
    op_name: StaticString,
    detail: StaticString = "",
    task_id: OptionalReg[Int] = None,
):
    with Trace[level, target=target](op_name, detail, task_id=task_id):
        pass


# CHECK-LABEL: test_op_logging
def main():
    print("== test_op_logging")

    # CHECK-NOT: [OP] LAUNCH
    test_op_logging[TraceLevel.THREAD]("test_op_without_op_logging")

    # CHECK: [OP] LAUNCH test_op [id=0]
    # CHECK-NEXT: [OP] COMPLETE test_op [id=0]
    test_op_logging[TraceLevel.OP]("test_op")

    # Confirm that the next op as an id incremented by 1
    # CHECK: [OP] LAUNCH test_second_op [id=1]
    # CHECK-NEXT: [OP] COMPLETE test_second_op [id=1]
    test_op_logging[TraceLevel.OP]("test_second_op")

    # CHECK: [OP] LAUNCH test_op_with_target{{.*}} target=accelerator
    # CHECK: [OP] COMPLETE test_op_with_target{{.*}} target=accelerator
    test_op_logging[TraceLevel.OP, StaticString("accelerator")](
        "test_op_with_target"
    )

    # CHECK: [OP] LAUNCH test_op_with_target_and_id{{.*}} target=accelerator:42
    # CHECK: [OP] COMPLETE test_op_with_target_and_id{{.*}} target=accelerator:42
    test_op_logging[TraceLevel.OP, StaticString("accelerator")](
        "test_op_with_target_and_id", task_id=42
    )

    # CHECK: [OP] LAUNCH test_op_with_detail{{.*}} some detail
    # CHECK: [OP] COMPLETE test_op_with_detail{{.*}} some detail
    test_op_logging[TraceLevel.OP]("test_op_with_detail", "some detail")

    # CHECK: [OP] LAUNCH test_op_with_target_and_detail{{.*}} some detail;target=accelerator
    # CHECK: [OP] COMPLETE test_op_with_target_and_detail{{.*}} some detail;target=accelerator
    test_op_logging[TraceLevel.OP, StaticString("accelerator")](
        "test_op_with_target_and_detail", "some detail"
    )
