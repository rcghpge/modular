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

from pathlib import Path
from sys import DLHandle

from testing import assert_raises
from test_utils import TestSuite


def check_invalid_dlhandle():
    with assert_raises(contains="dlopen failed"):
        _ = DLHandle("/an/invalid/library")


def check_invalid_dlhandle_path():
    with assert_raises(contains="dlopen failed"):
        _ = DLHandle(Path("/an/invalid/library"))


def main():
    var suite = TestSuite()

    suite.test[check_invalid_dlhandle]()
    suite.test[check_invalid_dlhandle_path]()

    suite^.run()
