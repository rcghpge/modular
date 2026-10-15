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
"""The cascade generative-AI types, generated from ``gen_ai.capnp``.

pycapnp compiles the schema into a module at load time; that module replaces
this one, so ``from ...gen_ai_capnp import GenAIRequest`` yields the generated
struct classes.
"""

import copyreg
import sys
from pathlib import Path

import capnp

# pycapnp pickles readers and root builders, but a nested builder or a list
# (a request's `messages` or `text` options) cannot be written on its own, so
# those pickle as readers of a copy and as plain lists.
copyreg.pickle(
    capnp._DynamicStructBuilder,
    lambda struct: struct.as_reader().__reduce_ex__(4),
)
copyreg.pickle(capnp._DynamicListBuilder, lambda items: (list, (list(items),)))
copyreg.pickle(capnp._DynamicListReader, lambda items: (list, (list(items),)))

# Other schemas import this one by its path under the `max` package, and
# pycapnp aborts the process if one schema is loaded under two identities, so
# load it relative to that same import root.
_IMPORT_ROOT = Path(__file__).parents[4]
sys.modules[__name__] = capnp.load(
    str(Path(__file__).with_name("gen_ai.capnp")), imports=[str(_IMPORT_ROOT)]
)
