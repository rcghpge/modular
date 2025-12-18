//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_COMMONDRIVER_UTILS_H
#define GENERICML_COMMONDRIVER_UTILS_H

#include "GenericML/CommonDriver/DriverInputFormat.h"
#include "MLRT/AsyncRT/ForwardDecls.h"
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
void writeIntermediateFileIfRequested(const std::string &identifier,
                                      ModuleOp module,
                                      const CommonOptions *options,
                                      const BufferRef &file);

} // namespace M

#endif // GENERICML_COMMONDRIVER_UTILS_H
