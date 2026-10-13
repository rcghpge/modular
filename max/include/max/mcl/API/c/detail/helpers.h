//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_LIB_API_C_DETAIL_HELPERS_H
#define GENERICML_LIB_API_C_DETAIL_HELPERS_H

#include "Support/Error.h"
#include "Support/ML/DType.h"
#include "max/c/types.h"

namespace M::Detail {

/// Set status to error with given message.
void setError(M_Status *status, Error &&error);

/// Set status to OK and clear message
void setOK(M_Status *status);

/// Maps the C API's notion of `M_Dtype` to the GraphCompiler `M::DType`.
/// Returns `DType::invalid` for unknown or unrepresentable values.
M::DType mapDType(M_Dtype dtype);

/// Maps the GraphCompiler `M::DType` to the C API's `M_Dtype`.
/// Returns `M_UNKNOWN` for unknown or unrepresentable values.
M_Dtype mapDType(M::DType dtype);

} // namespace M::Detail

#endif // GENERICML_LIB_API_C_DETAIL_HELPERS_H
