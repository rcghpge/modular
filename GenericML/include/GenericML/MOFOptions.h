//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef SDK_MOF_OPTIONS_LOWERINGOPTIONS_H
#define SDK_MOF_OPTIONS_LOWERINGOPTIONS_H

#include "Support/Context.h"
#include "Support/ML/DebugPrint.h"

namespace M::MOF {

/// Options which control MOF lowering and execution.
/// See also MO::MoToMgpOptions for generic lowering options.
struct MOFLoweringOptions {
  DebugTensorPrintOptions debugTensorPrintOptions;

  /// Establish global context objects in runtime to capture (some of)
  /// these options.
  void setupContext(M::ContextRef context);

  void print(llvm::raw_ostream &os);
};

} // namespace M::MOF

#endif // SDK_MOF_OPTIONS_LOWERINGOPTIONS_H
