//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_ENGINECLI_COMMON_APIBRIDGE_H
#define SDK_ENGINECLI_COMMON_APIBRIDGE_H

#include "Support/ForwardDecls.h"
#include "max/cpp/aliases.h"
#include "max/cpp/pytorch/aliases.h"

namespace M {
class TensorSpec;
} // namespace M

namespace M::EngineCLI {

TensorSpecPtr tensorSpecToCAPI(const char *name, const TensorSpec &spec);
ErrorOr<TorchInputSpecPtr> tensorSpecToCAPITorchSpec(const TensorSpec &spec);
TensorSpec capiToTensorSpec(const M_TensorSpec *spec);

} // namespace M::EngineCLI

#endif // SDK_ENGINECLI_COMMON_APIBRIDGE_H
