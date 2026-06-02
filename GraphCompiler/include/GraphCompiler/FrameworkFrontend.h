//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_FRAMEWORKFRONTEND_H
#define GENERICML_FRAMEWORKFRONTEND_H

#include "Support/Buffer.h"
#include "Support/Context.h"
#include "Support/SymbolExport.h"

#include <memory>

namespace M {

class DriverConfig;
class DriverOptions;
class ExecutorDriverOptions;
enum class FileFormat : uint8_t;
namespace MO {
struct GCPipelineOptions;
}

// TODO: It would be nice if we didn't have to export these... Unsure how else
// MT would get access to the symbols it needs though.
#define EXPORT_FOR_MT DRIVER_VISIBILITY_EXPORT

class MOFContext final {
public:
  explicit MOFContext(ContextRef);
  EXPORT_FOR_MT MOFContext(ContextRef, int &argc, char **&argv);
  EXPORT_FOR_MT ~MOFContext();

  // TODO: setOptions is used only by mt
  EXPORT_FOR_MT ErrorOrSuccess setOptions(ArrayRef<std::string> opts);

  // TODO: getExecutorDriverOptions is only used by mt
  EXPORT_FOR_MT const ExecutorDriverOptions *getExecutorDriverOptions() const;

  // TODO: setDriverOptions is only used by mt
  EXPORT_FOR_MT void setDriverOptions(const DriverOptions &options);

  // TODO: overrideLoweringOptions is only used by mt
  EXPORT_FOR_MT void overrideLoweringOptions(MO::GCPipelineOptions &);

  ErrorOrSuccess setConfigs(ArrayRef<std::unique_ptr<DriverConfig>> configs);

  /// Given an input buffer with a known format, try to compile it to a model
  /// binary. On success, this returns a BufferRef that contains the compiled
  /// model in MEF format.
  EXPORT_FOR_MT ErrorOr<M::BufferRef> compileToBinary(M::BufferRef file,
                                                      FileFormat format);

  /// Given an input with a known format, try to compile it to a model
  /// binary. The type of input is unknown at this stage, is transparently
  /// passed to framework driver. On success, this returns a BufferRef that
  /// contains the compiled model in MEF format.
  ErrorOr<M::BufferRef> compileToBinary(void *source, FileFormat format);

private:
  struct Impl;

  std::unique_ptr<Impl> impl;
};

#undef EXPORT_FOR_MT

} // namespace M

#endif // GENERICML_FRAMEWORKFRONTEND_H
