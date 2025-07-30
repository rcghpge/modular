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


struct KwParamStruct[greeting: String = "Hello", name: String = "🔥mojo🔥"]:
    fn __init__(out self):
        print(greeting, name)


fn use_kw_params():
    var a = KwParamStruct[]()  # prints 'Hello 🔥mojo🔥'
    var b = KwParamStruct[name="World"]()  # prints 'Hello World'
    var c = KwParamStruct[greeting="Hola"]()  # prints 'Hola 🔥mojo🔥'


def main():
    use_kw_params()
