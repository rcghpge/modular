//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_UTILS_H
#define GENERICML_DRIVER_UTILS_H

#include "GenericML/Driver/DriverCLOptions.h"
#include "LLCL/Runtime/Runtime.h"
#include "LLCL/Runtime/Utils.h"
#include "LLCL/Support/GenericUniquePtrSet.h"
#include "Support/LLVMCompilerForwardDecls.h"

namespace M {

/// This is the type we expect to have when
/// `setupMLIRAndRunWithLeakCheckedRuntime` is called. The runtime reference
/// and MLIRContext pointer are the only two things that the body function
/// could not capture outside this call.
using BodyFn = std::function<LogicalResult(Runtime &, MLIRContext *)>;

/// Run the given body function with a leak checked runtime.
int setupMLIRAndRunWithLeakCheckedRuntime(
    const ExecutorDriverCLOptions &options, const char *testName,
    std::unique_ptr<llvm::MemoryBuffer> &&file, BodyFn &&bodyFn,
    RCRef<LLCL::SharedGenericUniquePtrSet> globalContextObjects = {}) {
  if (failed(options.configureMLIRContextAndExecute(
          std::move(file), [&](MLIRContext *ctx) {
            return LLCL::runWithLeakCheckedRuntime(
                options, testName,
                [&](Runtime &runtime) { return bodyFn(runtime, ctx); },
                std::move(globalContextObjects));
          })))
    return EXIT_FAILURE;
  return EXIT_SUCCESS;
}

} // namespace M

#endif // GENERICML_DRIVER_UTILS_H
