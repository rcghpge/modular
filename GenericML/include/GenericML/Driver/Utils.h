//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_DRIVER_UTILS_H
#define GENERICML_DRIVER_UTILS_H

#include "AsyncRT/ForwardDecls.h"
#include "GenericML/Driver/DriverInputFormat.h"
#include "Support/Buffer.h"
#include "Support/Context.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"

#include <functional>
#include <memory>
#include <string>

namespace M {

class CommonOptions;
class ExecutorDriverOptions;

/// This is the type we expect to have when
/// `setupMLIRAndRun` is called. The runtime reference
/// and MLIRContext pointer are the only two things that the body function
/// could not capture outside this call.
using BodyFn = std::function<LogicalResult(Runtime &, MLIRContext *)>;

/// Run the given body function with runtime from given context.
int setupMLIRAndRun(ContextRef context, const ExecutorDriverOptions &options,
                    const char *testName,
                    std::unique_ptr<llvm::MemoryBuffer> &&file,
                    BodyFn &&bodyFn);

// FIXME: We have to duplicate this here because of #1639 (RUNP-82). Remove
//        this when possible to do so.
ErrorOr<std::string> getFileContentIdentity(const BufferRef &file,
                                            FileFormat format);

/// Saves an intermediate file with a dump of `module` if --saved-temps is set.
void writeIntermediateFileIfRequested(const std::string &identifier,
                                      ModuleOp module,
                                      const CommonOptions *options,
                                      const BufferRef &file);

} // namespace M

#endif // GENERICML_DRIVER_UTILS_H
