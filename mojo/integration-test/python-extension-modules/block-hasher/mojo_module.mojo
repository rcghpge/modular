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

from os import abort
from sys import sizeof

from python import Python, PythonObject
from python.bindings import PythonModuleBuilder
from python._cpython import PyObjectPtr


@export
fn PyInit_mojo_module() -> PythonObject:
    """Create a Python module with function bindings for `mojo_block_hasher`."""
    try:
        var b = PythonModuleBuilder("mojo_module")
        b.def_function[mojo_block_hasher](
            "mojo_block_hasher",
            docstring=(
                "Computes block hashes for a numpy array containing tokens"
            ),
        )
        return b.finalize()
    except e:
        return abort[PythonObject](
            String("failed to create Python module: ", e)
        )


@fieldwise_init
struct PyArrayObject[dtype: DType](Copyable, Movable):
    """
    Container for a numpy array.

    See: https://numpy.org/doc/2.1/reference/c-api/types-and-structures.html#c.PyArrayObject
    """

    var data: UnsafePointer[Scalar[dtype]]
    var nd: Int
    var dimensions: UnsafePointer[Int]
    var strides: UnsafePointer[Int]
    var base: PyObjectPtr
    var descr: PyObjectPtr
    var flags: Int
    var weakreflist: PyObjectPtr

    # version dependent private members are omitted
    # ...

    fn num_elts(self) -> Int:
        var num_elts = 1
        for i in range(self.nd):
            num_elts *= self.dimensions[i]
        return num_elts


@always_inline
fn _mojo_block_hasher[
    dtype: DType, //,
](
    py_array_object_ptr: UnsafePointer[PyArrayObject[dtype]],
    block_size: Int,
) -> PythonObject:
    # Compute number of hashes
    var num_elts: Int = py_array_object_ptr[].num_elts()
    var num_hashes: Int = num_elts // block_size

    var cpython = Python().cpython()

    # Create a list of NULL elements with the size needed to store the hash
    # results.
    var result_py_list = cpython.PyList_New(num_hashes)

    # Initial hash seed value
    alias initial_hash = hash("None")

    # Performing hashing
    var prev_hash = initial_hash
    var num_bytes = block_size * sizeof[dtype]()
    var hash_ptr_base = py_array_object_ptr[].data
    for block_idx in range(num_hashes):
        var hash_ptr_ints = hash_ptr_base.offset(block_idx * block_size)
        var hash_ptr_bytes = hash_ptr_ints.bitcast[Byte]()
        var token_hash = hash(hash_ptr_bytes, num_bytes)
        var pair_to_hash = SIMD[DType.uint64, 2](prev_hash, token_hash)
        var curr_hash = hash(pair_to_hash)
        # Convert the hash result to a Python object and store it in our
        # uninitialized list.
        var curr_hash_obj = cpython.PyLong_FromSsize_t(Int(curr_hash))
        _ = cpython.PyList_SetItem(result_py_list, block_idx, curr_hash_obj)

        prev_hash = curr_hash

    return PythonObject(from_owned_ptr=result_py_list)


@export
fn mojo_block_hasher(
    py_array_object: PythonObject,
    block_size_obj: PythonObject,
) raises -> PythonObject:
    # Parse np array tokens input
    var py_array_object_ptr = UnsafePointer[PyArrayObject[DType.int32], **_](
        unchecked_downcast_value=py_array_object
    )

    # Parse block size
    var block_size = Int(block_size_obj)

    # Performing hashing
    var results = _mojo_block_hasher(py_array_object_ptr, block_size)

    return results^
