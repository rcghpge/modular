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
#
# Cascade core types that every schema shares, mirroring the Python `core`
# package: the vocabulary pipelines, workers and runtimes are built from.

@0xb1d1f3a7c2e94d10;

# An async sequence of `T`, passed or returned by a method in place of a
# plain value. A local runtime special-cases this type and hands its own
# handle (e.g. the Rust runtime's queue) straight through, so no stream is
# ever serialized.
# TODO: define the remote transport (methods) once a runtime needs one.
interface ResultIter(T) {}
