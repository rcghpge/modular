//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_API_C_DETAIL_MODELCONTEXT_H
#define GENERICML_API_C_DETAIL_MODELCONTEXT_H

#include "GraphCompiler/FrameworkFrontend.h"
#include "Support/Buffer.h"
#include "Support/Context.h"
#include "max/API/c/SourceModel.h"

#include <memory>

namespace M::Detail {
/// We need to separate framework loading and compilation into separate tasks
/// as some framework needs to be loaded on main thread. Some frameworks
/// require sniffing model format from source buffer to load correct framework
/// Having a context class that can hold both framework and the buffer helps us
/// avoid double reads from file.
struct ModelContext {
  ModelContext(M::BufferRef buffer, std::filesystem::path path,
               FileFormat format, std::unique_ptr<MOFContext> ctx)
      : buffer(std::move(buffer)), model(std::move(path), format),
        frameworkCtx(std::move(ctx)) {}

  M::BufferRef buffer;
  SourceModel model;
  std::unique_ptr<MOFContext> frameworkCtx;
};

/// Allocates a `ModelContext` and initializes it for the framework
/// corresponding to the model given by `modelPath`.
/// For TF, initializes the context on the main thread, otherwise adds a task
/// to perform the initialization.
ErrorOr<ModelContext> loadModelFramework(const std::filesystem::path &modelPath,
                                         M::ContextRef context);

} // namespace M::Detail

#endif
