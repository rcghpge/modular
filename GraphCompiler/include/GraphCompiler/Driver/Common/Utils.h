//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_COMMONDRIVER_UTILS_H
#define GENERICML_COMMONDRIVER_UTILS_H

#include "AsyncRT/ForwardDecls.h"
#include "GraphCompiler/Driver/Common/DriverInputFormat.h"
#include "Support/Buffer.h"
#include "Support/Context.h"
#include "Support/ForwardDecls.h"
#include "Support/LLVMCompilerForwardDecls.h"

#include <functional>
#include <memory>
#include <string>

namespace M {

class CommonOptions;

// FIXME(RUNP-82): Duplicated because of #1639.
ErrorOr<std::string> getFileContentIdentity(const BufferRef &file,
                                            FileFormat format);

/// Saves an intermediate file with a dump of `module` if --saved-temps is set.
/// `baseName` is used as the output file stem when `file` has no backing path.
void writeIntermediateFileIfRequested(const std::string &identifier,
                                      ModuleOp module,
                                      const CommonOptions *options,
                                      const BufferRef &file,
                                      llvm::StringRef baseName = "temp");

/// Returns the per-stage IR dump directory, or empty if dumping is disabled.
/// Config keys, in order of preference:
///   1. `max-debug.ir-output-dir` (set via `MODULAR_DEBUG=ir-output-dir=...`,
///      `[max-debug] ir-output-dir = ...` in `modular.cfg`, or
///      `InferenceSession.debug.ir_output_dir = ...`).
///   2. `max.temps_dir` (legacy; also set by `MODULAR_MAX_TEMPS_DIR`).
///
/// All IR-dump gating should go through this so one setting drives both.
std::string getMaxIRDumpDir();

} // namespace M

#endif // GENERICML_COMMONDRIVER_UTILS_H
