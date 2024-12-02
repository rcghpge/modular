//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPH_COMPILER_MOGGSUPPORT_MOJOTARGETFUNCTIONS_H
#define GENERICML_GRAPH_COMPILER_MOGGSUPPORT_MOJOTARGETFUNCTIONS_H

#include "llvm/ADT/StringRef.h"

namespace M::MOGG::Utils {

// The KGEN function calls we will match based on its name in various graph MOGG
// passes (e.g., in/out lambda fusion).

constexpr llvm::StringLiteral DPSTensorForEach =
    "tensor_utils_internal::managed_tensor_slice::foreach";

constexpr llvm::StringLiteral MOGGElementwiseTarget = "elementwise";

constexpr llvm::StringLiteral MOGGSimdLoadTarget = "simd_load";

constexpr llvm::StringLiteral MOGGSimdStoreTarget = "simd_store";

constexpr llvm::StringLiteral LambdasHaveFusion = "lambdas_have_fusion";

constexpr llvm::StringLiteral DPSTensorCopy =
    "tensor_utils_internal::managed_tensor_slice::view_copy_impl";

constexpr llvm::StringLiteral DPSTensorGetTensorSpecWithoutLambdas =
    "tensor_utils_internal::managed_tensor_slice::_get_tensor_specs_without_"
    "lambdas";

constexpr llvm::StringLiteral DPSBuildTensorSpecs = "build_static_tensor_specs";

constexpr llvm::StringLiteral DPSRebuildTensorSpecsWithLambdas =
    "rebuild_static_tensor_specs_with_lambdas";

constexpr llvm::StringLiteral DPSRebuildTensorSpecsWithStrides =
    "rebuild_static_tensor_specs_with_strides";

} // namespace M::MOGG::Utils

#endif // GENERICML_GRAPH_COMPILER_MOGGSUPPORT_MOJOTARGETFUNCTIONS_H
