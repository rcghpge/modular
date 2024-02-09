//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJO_PARAMETERS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJO_PARAMETERS_H

#include <llvm/ADT/StringRef.h>

namespace M::MOGG {

static constexpr llvm::StringLiteral PARAMETER_SINGLE_THREAD_BLOCKING_OVERRIDE =
    "single_thread_blocking_override";
static constexpr llvm::StringLiteral PARAMETER_LAMBDAS_HAVE_FUSION =
    "lambdas_have_fusion";

/// Matmul and BatchMatmul op
static constexpr llvm::StringLiteral PARAMETER_MATMUL_TRANSPOSED_WEIGHTS =
    "transpose_in_1";
static constexpr llvm::StringLiteral PARAMETER_MATMUL_PACKED_WEIGHTS =
    "packed_in_1";

/// Conv op
static constexpr llvm::StringLiteral PARAMETER_CONV_PACKED_FILTER =
    "filter_packed";
static constexpr llvm::StringLiteral PARAMETER_CONV_STATIC_STRIDES =
    "static_strides";
static constexpr llvm::StringLiteral PARAMETER_CONV_STATIC_DILATIONS =
    "static_dilations";
static constexpr llvm::StringLiteral PARAMETER_CONV_STATIC_PADDING =
    "static_padding";
static constexpr llvm::StringLiteral PARAMETER_CONV_INPUT_RANK = "input_rank";

/// ConvTranspose op
static constexpr llvm::StringLiteral PARAMETER_CONVTRANSPOSE_INPUT_RANK =
    "input_rank";

/// Gather ND op
static constexpr llvm::StringLiteral PARAMETER_GATHER_ND_BATCH_DIMENSIONS =
    "batch_dims";

/// Resize op
static constexpr llvm::StringLiteral
    PARAMETER_RESIZE_COORDINATE_TRANSFORM_MODE = "coordinate_transform_mode";
static constexpr llvm::StringLiteral PARAMETER_RESIZE_ROUNDING_MODE =
    "round_mode";
static constexpr llvm::StringLiteral PARAMETER_RESIZE_ANTIALIASING =
    "antialias";

/// Cum Sum op
static constexpr llvm::StringLiteral PARAMETER_CUM_SUM_EXCLUSIVE = "exclusive";

/// AvgPool op
static constexpr llvm::StringLiteral PARAMETER_AVG_POOL_COUNT_BOUNDARY =
    "count_boundary";

/// ROI Align op
static constexpr llvm::StringLiteral PARAMETER_ROI_ALIGNMENT = "aligned";
static constexpr llvm::StringLiteral PARAMETER_ROI_ALIGNMENT_MODE = "mode";

} // namespace M::MOGG

#endif
