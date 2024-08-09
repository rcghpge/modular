//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOLIBRARYANALYSIS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOLIBRARYANALYSIS_H

#include <string>

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"

#include "KGEN/CustomDialect/CustomDialect.h"
#include "KGEN/HLCFDialect/HLCFDialect.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITDialect.h"
#include "KGEN/MOGGPreElab/MOGGDecorators.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Pass/AnalysisManager.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::MOGG {

/// This analysis lazily loads the standard and extra Mojo libraries and exposes
/// an interface to inquire about their content.
class MojoLibraryAnalysis {
public:
  MojoLibraryAnalysis(MLIRContext *ctx, bool withDebuginfo);

  MojoLibraryAnalysis(const MojoLibraryAnalysis &other) = delete;

  MojoLibraryAnalysis(MojoLibraryAnalysis &&other) = delete;

  ~MojoLibraryAnalysis();

  /// This method must be called before we can use this object. The standard
  /// Mojo kernel library is directly embedded. Only the extra library paths
  /// must be provided.
  ///
  /// We have eager and lazy mode for the library loading. Lazy mode parses the
  /// mojo library on-demand and is triggered for the full materialization via
  /// `getMojoBindings()`. Eager mode fully materializes the library during
  /// `init()`. In general, eager mode has faster bulk materialization API than
  /// the lazy mode. Therefore, if you know you are going to have a full
  /// materialization, use eager mode. Otherwise, lazy mode is more efficient.
  ErrorOrSuccess init(ArrayRef<std::string> kernelLibraryPaths);

  /// Get the fully materialized module which contains all exported generators
  /// from mojo libraries.
  mlir::ModuleOp getMojoBindings() { return module; }

  /// Query the mojo kernel with the specified `name`. Return nullptr if not
  /// existed.
  Operation *lookup(StringRef name);

  /// Get the MLIR context.
  inline MLIRContext *getContext() { return context; }

  /// Get the iterator range of the mojo kernels.
  inline auto getRawMojoKernels() {
    return llvm::map_range(
        llvm::iterator_range(rawMojoKernel),
        [&](const llvm::StringMapEntry<KGEN::GeneratorOp> &entry) {
          return entry.getValue();
        });
  }

  /// Looks up a registered Mojo symbol using the name that was used for
  /// registration.
  inline KGEN::GeneratorOp getRegisteredMojoKernelFor(StringRef name) {
    return getRegisteredMojoFunctionFor(name, registeredMojoKernelSymbol);
  }

  /// Returns true if a registered Mojo symbol using the name exists.
  inline bool hasRegisteredMojoKernelFor(StringRef name) {
    return registeredMojoKernelSymbol.contains(name);
  }

  /// Return the mapping.
  const llvm::StringMap<std::string> &getRegisteredMojoKernelSymbol() {
    return registeredMojoKernelSymbol;
  }

  /// Looks up a registered Mojo shape function using the name that was used
  /// for registration.
  inline KGEN::GeneratorOp getRegisteredMojoShapeFunctionFor(StringRef name) {
    return getRegisteredMojoFunctionFor(name, registeredShapeFunctionSymbol);
  }

  /// Looks up and returns a mojo function as is.
  KGEN::GeneratorOp getMojoFunctionNamed(StringRef name);

  /// Return true if the provided kernel or shape function has the input
  /// parameter named `parameterName`.
  bool kernelHasInputParameter(KGEN::GeneratorOp kernel,
                               StringRef parameterName);

  /// Returns true if the kernel is an elementwise kernel.
  inline bool kernelIsElementwise(KGEN::GeneratorOp kernel) {
    return kernel->hasAttr(KGEN::MOGGPreElab::Decorators::ELEMENTWISE.attr);
  }

  /// Returns true if the kernel is expecting to take indices as last arg.
  inline bool kernelIsViewOp(KGEN::GeneratorOp kernel) {
    return kernel->hasAttr(KGEN::MOGGPreElab::IS_VIEW_ATTR);
  }

  /// Returns true if the kernel is a view operation.
  inline bool kernelTakesIndices(KGEN::GeneratorOp kernel) {
    return kernel->hasAttr(KGEN::MOGGPreElab::Decorators::TAKES_INDICES.attr);
  }

  /// Returns true if the kernel implementation is in the Tensor API.
  inline bool kernelUsesTensorAPI(KGEN::GeneratorOp kernel) {
    return kernel->hasAttr(KGEN::MOGGPreElab::ALLOCS_ATTR) ||
           kernel->hasAttr(KGEN::MOGGPreElab::SLICED_ATTR);
  }

private:
  /// Get the materialized mojo kernel. If the kernel is not materialized, the
  /// materialization happens in place.
  KGEN::GeneratorOp
  getRegisteredMojoFunctionFor(StringRef name,
                               llvm::StringMap<std::string> &mapping);

  /// Load the user libraries lazily.
  ErrorOrSuccess
  loadUserProvidedKernels(ArrayRef<std::string> kernelLibraryPaths);

  /// Helper function to update the module and state data but account for
  /// collisions
  void updateLoadedKernelsForCollisions(
      Operation *userModule,
      llvm::StringMap<KGEN::GeneratorOp> &userRawMojoKernel);

  /// Internalize an MLIR bytecode buffer into the source manager and return a
  /// buffer reference.
  llvm::MemoryBufferRef
  internMLIRFile(std::unique_ptr<llvm::MemoryBuffer> buffer);
  /// Read and internalize an MLIR bytecode file from a filepath into the source
  /// manager and return a buffer reference.
  ErrorOr<llvm::MemoryBufferRef> readAndInternMLIRFile(StringRef filename);
  /// Load and process kernels from an MLIR buffer.
  ErrorOr<Operation *>
  loadMLIRKernels(llvm::MemoryBufferRef bufferRef,
                  llvm::StringMap<KGEN::GeneratorOp> &kernelMap,
                  const Twine &diag);

  /// Materialize the specified mojo kernel. It does nothing if the kernel has
  /// been materialized.
  LogicalResult ready(KGEN::GeneratorOp gen);

  /// Replace all symbol names in the specified mojo kernel so that all symbol
  /// name conflicts are resolved after calling this method.
  LogicalResult replaceSymbols(KGEN::GeneratorOp gen);

  /// Get the iterator range of the mojo kernels within the given kernel map.
  inline auto toRawMojoKernelRange(
      const llvm::StringMap<KGEN::GeneratorOp> &userKernelMap) {
    return llvm::map_range(
        llvm::iterator_range(userKernelMap),
        [&](const llvm::StringMapEntry<KGEN::GeneratorOp> &entry) {
          return entry.getValue();
        });
  }

  /// Hold the symbol name mapping for registered mojo kernels.
  llvm::StringMap<std::string> registeredMojoKernelSymbol;

  /// Hold the symbol name mapping for registered mojo shape functions.
  llvm::StringMap<std::string> registeredShapeFunctionSymbol;

  /// MLIR bytecode readers for lazy loading the mojo libraries. Each library
  /// has a dedicated reader.
  llvm::SmallVector<std::unique_ptr<mlir::BytecodeReader>> readers;

  /// This block holds the MLIR modules (might be partially loaded) from mojo
  /// libraries. It holds all of the Module OPs. Each library (either the
  /// main binding or the user-provided ones) would have a module op, which
  /// stays inside the block. Inside each module op, we have a bunch of
  /// generator ops.
  Block block;

  /// This module holds all mojo kernels from all mojo libraries.
  /// User-provided modules are cloned into this module after the full
  /// materialization (enforced by the `getMojoBindings` call).
  ModuleOp module;

  MLIRContext *context;

  /// Should libraries with debuginfo be loaded if they exist?
  bool withDebuginfo;

  mlir::ParserConfig config;

  /// This map captures all the mojo kernels (main and user-provided), which
  /// might not be materialized yet.
  llvm::StringMap<KGEN::GeneratorOp> rawMojoKernel;

  /// Stores all re-naming information needed to resolve the symbol name
  /// conflicts (duplication).
  using RenameEntry = std::pair<std::string, FlatSymbolRefAttr>;
  llvm::DenseMap<Operation *, llvm::SmallVector<RenameEntry, 16>>
      symbolNameRemapping;

  /// Source manager needed for parsing.
  std::shared_ptr<llvm::SourceMgr> sourceMgr;
};

/// An analysis "pass" for our passes to use to share the loaded library state.
struct LoadMojoAnalysisPass {
  LoadMojoAnalysisPass(Operation *op) : analysis(nullptr) {}

  /// Aggressively assume it is valid so the passes don't force a load each time
  /// even when they don't touch it.
  bool isInvalidated(const mlir::AnalysisManager::PreservedAnalyses &) {
    return false;
  }

  ErrorOr<MojoLibraryAnalysis *>
  initIfNeeded(MLIRContext *ctx, ArrayRef<std::string> kernelLibraryPaths,
               bool withDebuginfo) {
    // Reuse the library if it exists.
    if (analysis)
      return analysis.get();
    analysis = std::make_unique<MojoLibraryAnalysis>(ctx, withDebuginfo);
    auto loadError = analysis->init(kernelLibraryPaths);
    if (loadError.isError()) {
      analysis = nullptr;
      return loadError.takeError();
    }
    return analysis.get();
  }

  std::shared_ptr<MojoLibraryAnalysis> analysis;
};

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOLIBRARYANALYSIS_H
