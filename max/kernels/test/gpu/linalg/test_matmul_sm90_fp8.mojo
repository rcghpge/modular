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

import linalg.vendor_blas
from gpu.host import DeviceContext
from internal_utils._utils import dynamic, static
from linalg.matmul_sm90_testbed import test_matmul_sm90
from linalg.matmul_tile_scheduler import MatmulSchedule
from utils.index import Index

# Helper to calculate block_tile_shape based on dtype and wgmma_n
alias block_tile_shape[wgmma_n: Int, a_dtype: DType] = Index(
    128, wgmma_n, 128
) if a_dtype is DType.float8_e4m3fn else Index(128, wgmma_n, 64)

# Helper to calculate wgmma_shape based on dtype and wgmma_n
alias wgmma_shape[wgmma_n: Int, a_dtype: DType] = Index(
    64, wgmma_n, 32
) if a_dtype is DType.float8_e4m3fn else Index(64, wgmma_n, 16)


fn main() raises:
    with DeviceContext() as ctx:
        # NOTE: please note that cublaslt handle should be used for fp8-e4m3fn and cublas handle for bfloat16
        # because cublas does not support float8-e4m3fn. Also, fp8 tests should be run first and then bfloat16 tests
        # otherwise we will get unhandled exception error.
        print("FP8-E4M3FN GEMM TESTS")
        test_matmul_sm90[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[80, DType.float8_e4m3fn],
            wgmma_shape[80, DType.float8_e4m3fn],
            num_consumer=2,
            num_pipeline_stages=6,
            partitioned_multicast=False,
            grid_shape = Index(32, 4),
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[512](),
            static[2560](),
            static[8192](),
        )

        test_matmul_sm90[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, DType.float8_e4m3fn],
            wgmma_shape[128, DType.float8_e4m3fn],
            num_consumer=2,
            partitioned_multicast=False,
            num_pipeline_stages=6,
            grid_shape = Index(10, 13),
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[8192](),
            static[2560](),
            static[8192](),
        )

        test_matmul_sm90[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, DType.float8_e4m3fn],
            wgmma_shape[128, DType.float8_e4m3fn],
            num_consumer=2,
            partitioned_multicast=False,
            num_pipeline_stages=6,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[4096](),
            static[2560](),
            static[8192](),
        )

        test_matmul_sm90[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, DType.float8_e4m3fn],
            wgmma_shape[128, DType.float8_e4m3fn],
            num_consumer=2,
            partitioned_multicast=False,
            num_pipeline_stages=6,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[512](),
            static[8192](),
            static[2048](),
        )

        test_matmul_sm90[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, DType.float8_e4m3fn],
            wgmma_shape[128, DType.float8_e4m3fn],
            num_consumer=2,
            num_pipeline_stages=6,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[512](),
            static[14336](),
            static[8192](),
        )

        test_matmul_sm90[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, DType.float8_e4m3fn],
            wgmma_shape[128, DType.float8_e4m3fn],
            num_consumer=2,
            num_pipeline_stages=6,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[4096](),
            static[8192](),
            static[7168](),
        )

        test_matmul_sm90[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            Index(2, 1, 1),
            block_tile_shape[128, DType.float8_e4m3fn],
            wgmma_shape[128, DType.float8_e4m3fn],
            num_consumer=2,
            num_pipeline_stages=6,
            partitioned_multicast=False,
            schedule = MatmulSchedule.TILE2D,
            measure_threshold=0.001,
        ](
            ctx,
            static[512](),
            static[8192](),
            static[7168](),
        )

        test_matmul_sm90[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            Index(1, 2, 1),
            block_tile_shape[128, DType.float8_e4m3fn],
            wgmma_shape[128, DType.float8_e4m3fn],
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(199),
            static[512](),
            static[1024](),
        )

        test_matmul_sm90[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            Index(1, 2, 1),
            block_tile_shape[128, DType.float8_e4m3fn],
            wgmma_shape[128, DType.float8_e4m3fn],
            partitioned_multicast=False,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(200),
            static[256](),
            static[256](),
        )

        test_matmul_sm90[
            DType.float8_e4m3fn,
            DType.float8_e4m3fn,
            DType.bfloat16,
            Index(1, 2, 1),
            block_tile_shape[128, DType.float8_e4m3fn],
            wgmma_shape[128, DType.float8_e4m3fn],
            num_consumer=2,
            partitioned_multicast=True,
            measure_threshold=0.001,
        ](
            ctx,
            dynamic(201),
            static[384](),
            static[256](),
        )
