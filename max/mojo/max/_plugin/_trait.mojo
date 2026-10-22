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

from std.collections import OptionalReg
from std.collections.string.string_span import _get_kgen_string
from std.utils.index import IndexList
from std._gpu import lane_id
from std._plugin.selector import Plugin
from std.sys.info import CompilationTarget

from max.gpu import PDLLevel
from max.gpu.globals import WARP_SIZE
from max.gpu.host import DeviceContext
from max.gpu.memory.memory import (
    _AsyncCopyGroupPluginHookFnType,
    _AsyncCopyWaitGroupPluginHookFnType,
)
from max.gpu.primitives.warp import _FULL_MASK
from max.gpu.sync.sync import (
    _BarrierPluginHookFnType,
    _SyncwarpPluginHookFnType,
)


trait MaxPlugin(Plugin):
    comptime HookFor[target: CompilationTarget]: MaxPluginHooks


trait MaxPluginHooks:
    """Defines compile-time hooks for target-specific MAX behavior."""

    @staticmethod
    def elementwise_fn[
        rank: Int,
        simd_width: Int,
        *,
        pdl_level: PDLLevel = PDLLevel.ON,
    ](
        func: Some[
            def[
                width: Int, rank: Int, alignment: Int = 1
            ](IndexList[rank]) -> None
        ],
        shape: IndexList[rank, element_type=_],
        ctx: DeviceContext,
    ) raises:
        """Per-target plugin hook for `elementwise[...]`.

        Parameters:
            rank: The rank of the work domain.
            simd_width: The SIMD lane count for bulk invocations.
            pdl_level: PDL level for overlap control.

        Args:
            func: The body closure to invoke per index.
            shape: The shape of the work domain.
            ctx: The device context to dispatch on.

        Only invoked when `_handles_elementwise` is `True`; the default
        is never called and is a no-op.
        """
        pass

    comptime _handles_elementwise: Bool = False
    """Whether this plugin overrides MAX elementwise dispatch."""

    comptime barrier_fn: OptionalReg[_BarrierPluginHookFnType] = None
    """Lowers `barrier()`, which synchronizes the threads of a block and
    makes their shared-memory writes visible to each other.

    A GPU target that leaves this `None` rejects `barrier()` at compile time.
    """

    comptime syncwarp_fn: OptionalReg[_SyncwarpPluginHookFnType] = None
    """Lowers `syncwarp()`, which synchronizes the lanes of a warp.

    A GPU target that leaves this `None` rejects `syncwarp()` at compile time.

    Args:
        mask: The lanes taking part; a target without lane masks ignores it.
    """

    comptime async_copy_commit_group_fn: OptionalReg[
        _AsyncCopyGroupPluginHookFnType
    ] = None
    """Lowers `async_copy_commit_group()`.

    A target without asynchronous copy groups leaves this and the two wait
    hooks `None`: its `async_copy()` completes before returning, so there is
    nothing to commit or wait for, and the calls compile to nothing.
    """

    comptime async_copy_wait_group_fn: OptionalReg[
        _AsyncCopyWaitGroupPluginHookFnType
    ] = None
    """Lowers `async_copy_wait_group()`; see `async_copy_commit_group_fn`.

    Args:
        n: The number of committed groups that may still be pending.
    """

    comptime async_copy_wait_all_fn: OptionalReg[
        _AsyncCopyGroupPluginHookFnType
    ] = None
    """Lowers `async_copy_wait_all()`; see `async_copy_commit_group_fn`."""

    @staticmethod
    def shuffle_idx_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        """Lowers `shuffle_idx()`, which copies `val` from lane `offset`.

        This is the one shuffle a GPU plugin has to provide: the up, down and
        XOR hooks default to reading their source lane through it. A target
        that doesn't override it rejects every shuffle at compile time.

        Parameters:
            dtype: The element type of `val`.
            simd_width: The number of elements in `val`.

        Args:
            mask: The lanes taking part; a target without lane masks ignores it.
            val: The value each lane contributes.
            offset: The source lane.

        Returns:
            The value read from the source lane.
        """
        CompilationTarget.unsupported_target_error[operation="shuffle_idx"]()

    @inline(.always)
    @staticmethod
    def shuffle_up_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        """Lowers `shuffle_up()`, which copies `val` from the lane `offset`
        below the caller.

        The default reads that lane through `shuffle_idx_fn`; a lane whose
        source would fall below the first lane of the warp keeps its own
        value. A plugin with a native instruction overrides it.

        Parameters:
            dtype: The element type of `val`.
            simd_width: The number of elements in `val`.

        Args:
            mask: The lanes taking part; a target without lane masks ignores it.
            val: The value each lane contributes.
            offset: The number of lanes to look down.

        Returns:
            The value read from the source lane.
        """
        var lane = Int32(lane_id())
        var src_lane = lane - offset.cast[.int32]()
        return Self.shuffle_idx_fn(
            mask, val, UInt32(src_lane.lt(0).select(lane, src_lane))
        )

    @inline(.always)
    @staticmethod
    def shuffle_down_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        """Lowers `shuffle_down()`, which copies `val` from the lane `offset`
        above the caller.

        The default reads that lane through `shuffle_idx_fn`; a lane whose
        source would fall past the last lane of the warp keeps its own value.
        A plugin with a native instruction overrides it.

        Parameters:
            dtype: The element type of `val`.
            simd_width: The number of elements in `val`.

        Args:
            mask: The lanes taking part; a target without lane masks ignores it.
            val: The value each lane contributes.
            offset: The number of lanes to look up.

        Returns:
            The value read from the source lane.
        """
        var lane = UInt32(lane_id())
        return Self.shuffle_idx_fn(
            mask,
            val,
            (lane + offset).le(UInt32(WARP_SIZE - 1)).select(offset, 0) + lane,
        )

    @inline(.always)
    @staticmethod
    def shuffle_xor_fn[
        dtype: DType, simd_width: SIMDLength, //
    ](mask: UInt, val: SIMD[dtype, simd_width], offset: UInt32) -> SIMD[
        dtype, simd_width
    ]:
        """Lowers `shuffle_xor()`, which copies `val` from the lane whose ID is
        the caller's XOR `offset`.

        The default reads that lane through `shuffle_idx_fn`; a lane whose
        source would fall outside the warp keeps its own value. A plugin with
        a native instruction overrides it.

        Parameters:
            dtype: The element type of `val`.
            simd_width: The number of elements in `val`.

        Args:
            mask: The lanes taking part; a target without lane masks ignores it.
            val: The value each lane contributes.
            offset: The value XORed with the caller's lane ID.

        Returns:
            The value read from the source lane.
        """
        var lane = UInt32(lane_id())
        var src_lane = lane ^ offset
        return Self.shuffle_idx_fn(
            mask, val, src_lane.lt(UInt32(WARP_SIZE)).select(src_lane, lane)
        )

    @staticmethod
    def vote_fn[ret_type: DType](val: Bool) -> Scalar[ret_type]:
        """Lowers `vote()`, which returns the mask of warp lanes that voted
        `True`.

        A GPU plugin has to provide this. A target that doesn't override it
        rejects `vote()` at compile time.

        Parameters:
            ret_type: The mask type, `DType.uint32` or `DType.uint64`.

        Args:
            val: The calling lane's vote.

        Returns:
            The mask with bit `l` set for each lane `l` that voted `True`.
        """
        CompilationTarget.unsupported_target_error[operation="vote"]()

    @inline(.always)
    @staticmethod
    def match_any_fn[
        dtype: DType, //, mask_type: DType
    ](bits: Scalar[dtype]) -> Scalar[mask_type]:
        """Lowers `match_any()`, which returns the mask of warp lanes whose
        bits equal the caller's.

        The default needs only `shuffle_idx_fn`: each lane reads every lane's
        bits, which costs `WARP_SIZE` shuffles. A plugin with a match
        instruction, or a cheaper way to group equal lanes, overrides it.

        Parameters:
            dtype: The bits' type, `DType.uint32` or `DType.uint64`.
            mask_type: The mask type, `DType.uint32` or `DType.uint64`.

        Args:
            bits: The bits of the calling lane's value.

        Returns:
            The mask with bit `l` set for each lane `l` holding equal bits.
        """
        var result = Scalar[mask_type](0)
        comptime for l in range(WARP_SIZE):
            if Self.shuffle_idx_fn(_FULL_MASK, bits, UInt32(l)) == bits:
                result |= Scalar[mask_type](1) << Scalar[mask_type](l)
        return result

    @inline(.always)
    @staticmethod
    def match_all_fn[
        dtype: DType, //, mask_type: DType
    ](bits: Scalar[dtype]) -> Scalar[mask_type]:
        """Lowers `match_all()`, which returns the mask of warp lanes if they
        all hold the caller's bits, and 0 otherwise.

        The default needs only `shuffle_idx_fn`: each lane compares every
        lane's bits with the first lane's, which costs `WARP_SIZE` shuffles.
        A plugin with a match instruction, or a cheaper way to compare the
        lanes, overrides it.

        Parameters:
            dtype: The bits' type, `DType.uint32` or `DType.uint64`.
            mask_type: The mask type, `DType.uint32` or `DType.uint64`.

        Args:
            bits: The bits of the calling lane's value.

        Returns:
            The mask of all lanes if they hold equal bits, otherwise 0.
        """
        var chosen = Self.shuffle_idx_fn(_FULL_MASK, bits, 0)
        var all_same = True
        var active = Scalar[mask_type](0)
        comptime for l in range(WARP_SIZE):
            if Self.shuffle_idx_fn(_FULL_MASK, bits, UInt32(l)) != chosen:
                all_same = False
            active |= Scalar[mask_type](1) << Scalar[mask_type](l)
        return active if all_same else Scalar[mask_type](0)


struct DefaultMaxPlugin(MaxPlugin):
    """Provides the default MAX plugin behavior."""

    comptime name: __mlir_type.`!kgen.string` = _get_kgen_string["default"]()
    comptime HookFor[target: CompilationTarget] = DefaultMaxPluginHooks[target]


struct DefaultMaxPluginHooks[target: CompilationTarget](MaxPluginHooks):
    pass
