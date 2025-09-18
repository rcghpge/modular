//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJO_PARAMETERS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJO_PARAMETERS_H

#include "llvm/ADT/StringRef.h"

namespace M::MOGG {

static constexpr llvm::StringLiteral PARAMETER_LAMBDAS_HAVE_FUSION =
    "lambdas_have_fusion";

/// Matmul and BatchMatmul op
static constexpr llvm::StringLiteral PARAMETER_MATMUL_TRANSPOSED_WEIGHTS =
    "transpose_b";
static constexpr llvm::StringLiteral PARAMETER_MATMUL_PACKED_WEIGHTS =
    "packed_b";

/// Conv op
static constexpr llvm::StringLiteral PARAMETER_CONV_PACKED_FILTER =
    "filter_packed";
static constexpr llvm::StringLiteral PARAMETER_CONV_INPUT_LAYOUT =
    "input_layout";
static constexpr llvm::StringLiteral PARAMETER_CONV_FILTER_LAYOUT =
    "filter_layout";
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

// TransposeOp
static constexpr llvm::StringLiteral PARAMETER_TRANSPOSE_STATIC_PERM =
    "static_permutations";

// SliceOp
static constexpr llvm::StringLiteral PARAMETER_SLICE_STATIC_STEPS =
    "static_steps";

// SliceDimOp
static constexpr llvm::StringLiteral PARAMETER_SLICE_DIM_STATIC_START =
    "static_start";
static constexpr llvm::StringLiteral PARAMETER_SLICE_DIM_STATIC_STEP =
    "static_step";

} // namespace M::MOGG

#endif
