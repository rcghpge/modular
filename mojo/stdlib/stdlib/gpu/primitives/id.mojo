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

"""This module provides GPU thread and block indexing functionality.

It defines aliases and functions for accessing GPU grid, block, thread and cluster
dimensions and indices. These are essential primitives for GPU programming that allow
code to determine its position and dimensions within the GPU execution hierarchy.

Most functionality is architecture-agnostic, with some NVIDIA-specific features clearly marked.
The module is designed to work seamlessly across different GPU architectures while providing
optimal performance through hardware-specific optimizations where applicable."""

import math
from sys import llvm_intrinsic
from sys.info import (
    CompilationTarget,
    _is_sm_9x_or_newer,
    is_amd_gpu,
    is_apple_gpu,
    is_gpu,
    is_nvidia_gpu,
)
from memory import AddressSpace

from ..globals import WARP_SIZE
from .warp import broadcast


# ===-----------------------------------------------------------------------===#
# Helper functions
# ===-----------------------------------------------------------------------===#


# Check that the dimension is either x, y, or z.
# TODO: Some day we should use typed string literals or 'requires' clauses to
#       enforce this at the type system level.
# https://github.com/modular/modular/issues/1278
fn _verify_xyz[dim: StaticString]():
    constrained[
        dim == "x" or dim == "y" or dim == "z",
        "the dimension must be x, y, or z",
    ]()


@always_inline
fn _get_gcn_idx[offset: Int, dtype: DType]() -> UInt:
    var ptr = llvm_intrinsic[
        "llvm.amdgcn.implicitarg.ptr",
        UnsafePointer[Scalar[dtype], address_space = AddressSpace.CONSTANT],
        has_side_effect=False,
    ]()
    return UInt(ptr.load[alignment=4](offset))


# ===-----------------------------------------------------------------------===#
# lane_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn lane_id() -> UInt:
    """Returns the lane ID of the current thread within its warp.

    The lane ID is a unique identifier for each thread within a warp, ranging from 0 to
    WARP_SIZE-1. This ID is commonly used for warp-level programming and thread
    synchronization within a warp.

    Returns:
        The lane ID (0 to WARP_SIZE-1) of the current thread.
    """
    constrained[is_gpu(), "This function only applies to GPUs."]()

    @parameter
    if is_nvidia_gpu():
        return UInt(
            Int(
                llvm_intrinsic[
                    "llvm.nvvm.read.ptx.sreg.laneid",
                    Int32,
                    has_side_effect=False,
                ]().cast[DType.uint32]()
            )
        )

    elif is_amd_gpu():
        alias none = Int32(-1)
        alias zero = Int32(0)
        var t = llvm_intrinsic[
            "llvm.amdgcn.mbcnt.lo", Int32, has_side_effect=False
        ](none, zero)
        return UInt(
            Int(
                llvm_intrinsic[
                    "llvm.amdgcn.mbcnt.hi", Int32, has_side_effect=False
                ](none, t).cast[DType.uint32]()
            )
        )

    elif is_apple_gpu():
        return UInt(
            Int(
                llvm_intrinsic[
                    "llvm.air.thread_index_in_simdgroup",
                    Int32,
                    has_side_effect=False,
                ]().cast[DType.uint32]()
            )
        )

    else:
        return CompilationTarget.unsupported_target_error[
            UInt,
            operation="lane_id",
        ]()


# ===-----------------------------------------------------------------------===#
# warp_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn warp_id() -> UInt:
    """Returns the warp ID of the current thread within its block.
    The warp ID is a unique identifier for each warp within a block, ranging
    from 0 to BLOCK_SIZE/WARP_SIZE-1. This ID is commonly used for warp-level
    programming and synchronization within a block.

    Returns:
        The warp ID (0 to BLOCK_SIZE/WARP_SIZE-1) of the current thread.
    """

    return thread_idx.x // UInt(WARP_SIZE)


# ===-----------------------------------------------------------------------===#
# sm_id
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
fn sm_id() -> UInt:
    """Returns the Streaming Multiprocessor (SM) ID of the current thread.

    The SM ID uniquely identifies which physical streaming multiprocessor the thread is
    executing on. This is useful for SM-level optimizations and understanding hardware
    utilization.

    If called on non-NVIDIA GPUs, this function aborts as this functionality
    is only supported on NVIDIA hardware.

    Returns:
        The SM ID of the current thread.
    """

    @parameter
    if is_nvidia_gpu():
        return broadcast(
            UInt(
                Int(
                    llvm_intrinsic[
                        "llvm.nvvm.read.ptx.sreg.smid",
                        Int32,
                        has_side_effect=False,
                    ]().cast[DType.uint32]()
                )
            )
        )
    else:
        return CompilationTarget.unsupported_target_error[
            UInt,
            operation="sm_id",
            note="sm_id() is only supported when targeting NVIDIA GPUs.",
        ]()


# ===-----------------------------------------------------------------------===#
# thread_idx
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct _ThreadIdx(Defaultable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a thread within a block."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StaticString:
        @parameter
        if is_nvidia_gpu():
            return "llvm.nvvm.read.ptx.sreg.tid." + dim
        elif is_amd_gpu():
            return "llvm.amdgcn.workitem.id." + dim
        elif is_apple_gpu():
            return "llvm.air.thread_position_in_threadgroup." + dim
        else:
            return CompilationTarget.unsupported_target_error[
                StaticString,
                operation="thread_idx field access",
            ]()

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` coordinates of a thread within a block.

        Returns:
            The `x`, `y`, or `z` coordinates of a thread within a block.
        """
        _verify_xyz[dim]()
        alias intrinsic_name = Self._get_intrinsic_name[dim]()
        return UInt(
            llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False]()
        )


alias thread_idx = _ThreadIdx()
"""Contains the thread index in the block, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# block_idx
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct _BlockIdx(Defaultable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a block within a grid."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StaticString:
        @parameter
        if is_nvidia_gpu():
            return "llvm.nvvm.read.ptx.sreg.ctaid." + dim
        elif is_amd_gpu():
            return "llvm.amdgcn.workgroup.id." + dim
        elif is_apple_gpu():
            return "llvm.air.threadgroup_position_in_grid." + dim
        else:
            return CompilationTarget.unsupported_target_error[
                StaticString,
                operation="block_idx field access",
            ]()

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` coordinates of a block within a grid.

        Returns:
            The `x`, `y`, or `z` coordinates of a block within a grid.
        """
        _verify_xyz[dim]()
        alias intrinsic_name = Self._get_intrinsic_name[dim]()
        return UInt(
            llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False]()
        )


alias block_idx = _BlockIdx()
"""Contains the block index in the grid, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# block_dim
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct _BlockDim(Defaultable):
    """Provides accessors for getting the `x`, `y`, and `z` dimensions of a
    block."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StaticString](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the block.

        Returns:
            The `x`, `y`, or `z` dimension of the block.
        """
        _verify_xyz[dim]()

        @parameter
        if is_nvidia_gpu():
            alias intrinsic_name = "llvm.nvvm.read.ptx.sreg.ntid." + dim
            return UInt(
                Int(
                    llvm_intrinsic[
                        intrinsic_name, Int32, has_side_effect=False
                    ]()
                )
            )
        elif is_apple_gpu():
            return UInt(
                Int(
                    llvm_intrinsic[
                        "llvm.air.threads_per_threadgroup." + dim,
                        Int32,
                        has_side_effect=False,
                    ]()
                )
            )
        elif is_amd_gpu():

            @parameter
            fn _get_offset() -> Int:
                @parameter
                if dim == "x":
                    return 6
                elif dim == "y":
                    return 7
                else:
                    constrained[dim == "z"]()
                    return 8

            return _get_gcn_idx[_get_offset(), DType.uint16]()

        else:
            return CompilationTarget.unsupported_target_error[
                UInt,
                operation="block_dim field access",
            ]()


alias block_dim = _BlockDim()
"""Contains the dimensions of the block as `x`, `y`, and `z` values (for
example, `block_dim.y`)"""


# ===-----------------------------------------------------------------------===#
# grid_dim
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct _GridDim(Defaultable):
    """Provides accessors for getting the `x`, `y`, and `z` dimensions of a
    grid."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StaticString](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the grid.

        Returns:
            The `x`, `y`, or `z` dimension of the grid.
        """
        _verify_xyz[dim]()

        @parameter
        if is_nvidia_gpu():
            alias intrinsic_name = "llvm.nvvm.read.ptx.sreg.nctaid." + dim
            return UInt(
                Int(
                    llvm_intrinsic[
                        intrinsic_name, Int32, has_side_effect=False
                    ]()
                )
            )
        elif is_amd_gpu():

            @parameter
            fn _get_offset() -> Int:
                @parameter
                if dim == "x":
                    return 0
                elif dim == "y":
                    return 1
                else:
                    constrained[dim == "z"]()
                    return 2

            return _get_gcn_idx[_get_offset(), DType.uint32]()
        elif is_apple_gpu():
            alias intrinsic_name = "llvm.air.threads_per_grid." + dim
            var gridDim = UInt(
                Int(
                    llvm_intrinsic[
                        intrinsic_name, Int32, has_side_effect=False
                    ]()
                )
            )
            # Metal passes grid dimention as a gridDim.dim * blockDim.dim.
            # To make things compatible with NVidia and AMDGPU, divide result
            # by block_dim.dim
            return gridDim // block_dim.__getattr__[dim]()
        else:
            return CompilationTarget.unsupported_target_error[
                UInt,
                operation="grid_dim field access",
            ]()


alias grid_dim = _GridDim()
"""Provides accessors for getting the `x`, `y`, and `z`
dimensions of a grid."""


# ===-----------------------------------------------------------------------===#
# global_idx
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct _GlobalIdx(Defaultable):
    """Provides accessors for getting the `x`, `y`, and `z` global offset of
    the kernel launch."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the program.

        Returns:
            The `x`, `y`, or `z` dimension of the program.
        """
        _verify_xyz[dim]()
        var t_idx = thread_idx.__getattr__[dim]()
        var b_idx = block_idx.__getattr__[dim]()
        var b_dim = block_dim.__getattr__[dim]()

        return math.fma(b_idx, b_dim, t_idx)


alias global_idx = _GlobalIdx()
"""Contains the global offset of the kernel launch, as `x`, `y`, and `z`
values."""


# ===-----------------------------------------------------------------------===#
# cluster_dim
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct _ClusterDim(Defaultable):
    """Provides accessors for getting the `x`, `y`, and `z` dimensions of a
    cluster."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    fn __getattr__[dim: StaticString](self) -> UInt:
        """Gets the `x`, `y`, or `z` dimension of the cluster.

        Returns:
            The `x`, `y`, or `z` dimension of the cluster.
        """
        constrained[
            _is_sm_9x_or_newer(),
            "cluster_id is only supported on NVIDIA SM90+ GPUs",
        ]()
        _verify_xyz[dim]()

        alias intrinsic_name = "llvm.nvvm.read.ptx.sreg.cluster.nctaid." + dim
        return UInt(
            Int(llvm_intrinsic[intrinsic_name, Int32, has_side_effect=False]())
        )


alias cluster_dim = _ClusterDim()
"""Contains the dimensions of the cluster, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# cluster_idx
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct _ClusterIdx(Defaultable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a cluster within a grid."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StaticString:
        return "llvm.nvvm.read.ptx.sreg.clusterid." + dim

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` coordinates of a cluster within a grid.

        Returns:
            The `x`, `y`, or `z` coordinates of a cluster within a grid.
        """
        constrained[
            _is_sm_9x_or_newer(),
            "cluster_id is only supported on NVIDIA SM90+ GPUs",
        ]()
        _verify_xyz[dim]()
        alias intrinsic_name = Self._get_intrinsic_name[dim]()
        return UInt(
            Int(llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False]())
        )


alias cluster_idx = _ClusterIdx()
"""Contains the cluster index in the grid, as `x`, `y`, and `z` values."""


# ===-----------------------------------------------------------------------===#
# block_id_in_cluster
# ===-----------------------------------------------------------------------===#


@register_passable("trivial")
struct _ClusterBlockIdx(Defaultable):
    """Provides accessors for getting the `x`, `y`, and `z` coordinates of
    a threadblock within a cluster."""

    @always_inline("nodebug")
    fn __init__(out self):
        return

    @always_inline("nodebug")
    @staticmethod
    fn _get_intrinsic_name[dim: StringLiteral]() -> StaticString:
        return "llvm.nvvm.read.ptx.sreg.cluster.ctaid." + dim

    @always_inline("nodebug")
    fn __getattr__[dim: StringLiteral](self) -> UInt:
        """Gets the `x`, `y`, or `z` coordinates of a threadblock within a cluster.

        Returns:
            The `x`, `y`, or `z` coordinates of a threadblock within a cluster.
        """
        constrained[
            _is_sm_9x_or_newer(),
            "cluster_id is only supported on NVIDIA SM90+ GPUs",
        ]()
        _verify_xyz[dim]()
        alias intrinsic_name = Self._get_intrinsic_name[dim]()
        return UInt(
            Int(llvm_intrinsic[intrinsic_name, UInt32, has_side_effect=False]())
        )


alias block_id_in_cluster = _ClusterBlockIdx()
"""Contains the block id of the threadblock within a cluster, as `x`, `y`, and `z` values."""
