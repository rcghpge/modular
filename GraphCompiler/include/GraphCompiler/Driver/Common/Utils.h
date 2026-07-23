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

// FIXME: We have to duplicate this here because of #1639 (RUNP-82). Remove
//        this when possible to do so.
ErrorOr<std::string> getFileContentIdentity(const BufferRef &file,
                                            FileFormat format);

/// Saves an intermediate file with a dump of `module` if --saved-temps is set.
/// `baseName` is used as the output file stem when `file` has no backing path.
void writeIntermediateFileIfRequested(const std::string &identifier,
                                      ModuleOp module,
                                      const CommonOptions *options,
                                      const BufferRef &file,
                                      llvm::StringRef baseName = "temp");

/// Returns the directory where intermediate compiler IR should be dumped
/// (per-pipeline-stage MLIR files), or an empty string if dumping is disabled.
///
/// Consulted Config keys, in order of preference:
///   1. `max-debug.ir-output-dir` (set via `MODULAR_DEBUG=ir-output-dir=...`,
///      `[max-debug] ir-output-dir = ...` in `modular.cfg`, or
///      `InferenceSession.debug.ir_output_dir = ...`).
///   2. `max.temps_dir` (legacy, also driven by the `MODULAR_MAX_TEMPS_DIR`
///      env var via `Config`'s automatic env-var mapping).
///
/// All callers that gate for destination IR dumping should funnel through this
/// helper so a single user-facing setting drives both.
std::string getMaxIRDumpDir();

} // namespace M

#endif // GENERICML_COMMONDRIVER_UTILS_H
