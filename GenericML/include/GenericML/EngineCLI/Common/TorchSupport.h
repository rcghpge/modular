//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_ENGINECLI_COMMON_TORCHSUPPORT_H
#define GENERICML_ENGINECLI_COMMON_TORCHSUPPORT_H

#include "Support/ForwardDecls.h"
#include "max/c/types.h"

namespace M::EngineCLI {

struct InputGenerationSchema;

ErrorOrSuccess setTorchInputSpecs(M_CompileConfig *config,
                                  const InputGenerationSchema &schema);

} // namespace M::EngineCLI

#endif // GENERICML_ENGINECLI_COMMON_TORCHSUPPORT_H
