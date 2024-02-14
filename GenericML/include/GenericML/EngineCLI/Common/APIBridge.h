//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_ENGINECLI_COMMON_APIBRIDGE_H
#define GENERICML_ENGINECLI_COMMON_APIBRIDGE_H

#include "Support/ForwardDecls.h"
#include "max/cpp/aliases.h"
#include "max/cpp/pytorch/aliases.h"

namespace M {
class TensorSpec;
} // namespace M

namespace M::EngineCLI {

M_Dtype dtypeToCAPI(DType dty);
DType capiToDType(M_Dtype dty);

TensorSpecPtr tensorSpecToCAPI(const char *name, const TensorSpec &spec);
TorchInputSpecPtr tensorSpecToCAPITorchSpec(const TensorSpec &spec);
TensorSpec capiToTensorSpec(const M_TensorSpec *spec);

} // namespace M::EngineCLI

#endif // GENERICML_ENGINECLI_COMMON_APIBRIDGE_H
