//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "Kernels/mcl/Tracing.h"
#include "llvm/ADT/StringRef.h"

using namespace M;

static constexpr int kResultSuccess = 0;

ErrorOr<MojoTrace> MojoTrace::create(StringRef message, StringRef color) {
  void *trace;
  if (create_trace(&trace, (void *)message.data(), message.size(),
                   (void *)color.data(), color.size()) == kResultSuccess)
    return MojoTrace(MojoTracePtr(trace, destroy_trace));

  return Error("unable to create Mojo trace");
}
