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

"""Entry point into _bgrec.recorder_main.

This exists because _bgrec itself can't be used as an entry point without
causing this warning:

    <frozen runpy>:128: RuntimeWarning: 'max.diagnostics.gpu._bgrec' found in
    sys.modules after import of package 'max.diagnostics.gpu', but prior to
    execution of 'max.diagnostics.gpu._bgrec'; this may result in unpredictable
    behaviour

This is because _bgrec is imported by __init__, but we need that, so this
module exists as an entry point that isn't imported by __init__.
"""

from ._bgrec import recorder_main

if __name__ == "__main__":
    recorder_main()
