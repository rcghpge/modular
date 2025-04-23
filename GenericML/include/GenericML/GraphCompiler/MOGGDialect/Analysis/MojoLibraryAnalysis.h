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

#include "KGEN/KGENDialect/KGENOps.h"
#include "Support/SymbolExport.h"
#include "mlir/Bytecode/BytecodeReader.h"
#include "mlir/Pass/AnalysisManager.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::MOGG {

/// This analysis lazily loads the standard and extra Mojo libraries and exposes
/// an interface to inquire about their content.
class MODULAR_CXX_EXPORT MojoLibraryAnalysis {
public:
  MojoLibraryAnalysis(MLIRContext *ctx);

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
  ErrorOrSuccess init(ArrayRef<std::string> kernelLibraryPaths = {});

  /// Load an extra kernel library.
  /// Update in place the bindings and registered kernels.
  ErrorOrSuccess addKernelLibraryPath(const std::string &kernelLibraryPath);

  /// Get the fully materialized module which contains all exported generators
  /// from mojo libraries.
  inline mlir::ModuleOp getMojoBindings() { return module; }

  /// Query the mojo kernel with the specified `name`. Return nullptr if not
  /// existed.
  Operation *lookup(StringRef name) const;

  /// Get the MLIR context.
  inline MLIRContext *getContext() { return module->getContext(); }

  /// Looks up a registered Mojo symbol using the name that was used for
  /// registration.
  KGEN::GeneratorOp getRegisteredMojoKernelFor(StringRef name) const;

  /// Returns true if a registered Mojo symbol using the name exists.
  inline bool hasRegisteredMojoKernelFor(StringRef name) const {
    return registeredMojoKernelSymbol.contains(name);
  }

  struct KernelCandidate {
    KernelCandidate(KGEN::GeneratorOp gen, int64_t priority)
        : generator(gen), priority(priority) {}
    KGEN::GeneratorOp generator;
    int64_t priority;
  };

  /// Return the mapping.
  const llvm::StringMap<KernelCandidate> &
  getRegisteredMojoKernelSymbol() const {
    return registeredMojoKernelSymbol;
  }

  /// Looks up a registered Mojo shape function using the name that was used
  /// for registration.
  KGEN::GeneratorOp getRegisteredMojoShapeFunctionFor(StringRef name) const;

  /// Looks up a registered Mojo execute function using the name that was used
  /// for registration.
  KGEN::GeneratorOp getRegisteredMojoExecuteFunctionFor(StringRef name) const;

  /// Looks up a registered Mojo update_input_view function using the name that
  /// was used for registration.
  KGEN::GeneratorOp
  getRegisteredMojoUpdateViewFunctionFor(StringRef name) const;

  /// Looks up a registered outlined function using the name that was used
  /// for registration.
  KGEN::GeneratorOp getRegisteredMojoOutlinedFunctionFor(StringRef name) const;

  /// Looks up the Mojo symbol of a registered, labeled kernel function
  /// using the name that was used for registration and the function label.
  KGEN::GeneratorOp getRegisteredMojoFunctionFor(StringRef name,
                                                 StringRef functionLabel) const;

  /// Looks up and returns a mojo function as is.
  KGEN::GeneratorOp getForeachKernel() const { return foreachKernel; }

  /// Looks up and returns a mojo function as is.
  KGEN::GeneratorOp getMojoFunctionNamed(StringRef name) const;

  /// Lookup the kernel and specialize it on the given parameters.
  ErrorOr<KGEN::SymbolConstantAttr>
  getSpecializedKernel(StringRef name, ArrayRef<TypedAttr> params) const;

  /// Build the KGEN::ParamOperatorAttr which calls the specified callee with
  /// the given parameters and operands. The callee is resolved using the
  /// library analysis.
  ErrorOr<TypedAttr> buildParameterCall(llvm::StringLiteral callee,
                                        ArrayRef<TypedAttr> parameters,
                                        ArrayRef<TypedAttr> operands) const;

  /// Return true if the provided kernel or shape function has the input
  /// parameter named `parameterName`.
  bool kernelHasInputParameter(KGEN::GeneratorOp kernel,
                               StringRef parameterName) const;

  /// Verify the call to a MO or custom kernel.
  ErrorOrSuccess verifyMOOpMatchesKernel(Operation *op) const;

private:
  /// Get the iterator range of the mojo kernels.
  inline auto getRawMojoKernels() const {
    return llvm::map_range(
        llvm::iterator_range(rawMojoKernel),
        [&](const llvm::StringMapEntry<KGEN::GeneratorOp> &entry) {
          return entry.getValue();
        });
  }

  /// Helper function to update the module and state data but account for
  /// collisions
  ErrorOrSuccess updateLoadedKernelsForCollisions(
      Operation *userModule,
      llvm::StringMap<KGEN::GeneratorOp> &userRawMojoKernel);

  /// Helper functionn to add `generator` to the maps of registered kernels if
  /// needed.
  ErrorOrSuccess updateRegisteredKernelsMap(KGEN::GeneratorOp generator);

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
      const llvm::StringMap<KGEN::GeneratorOp> &userKernelMap) const {
    return llvm::map_range(
        llvm::iterator_range(userKernelMap),
        [&](const llvm::StringMapEntry<KGEN::GeneratorOp> &entry) {
          return entry.getValue();
        });
  }

  KGEN::GeneratorOp foreachKernel;

  /// Hold the symbol name mapping for registered mojo kernels. With
  /// struct-based mojo kernels, this will map to the `execute` method.
  /// The value holds the kernel as well as its priority.
  llvm::StringMap<KernelCandidate> registeredMojoKernelSymbol;

  /// Hold the registered, labeled functions that can be attached to a kernel.
  /// For struct-based mojo kernels, this will map to labeled struct member
  /// methods.
  /// The outer map key is the kernel name. The inner map key is the
  /// function label (example: `mogg.shape` will map to the `shape` method).
  llvm::StringMap<llvm::StringMap<KGEN::GeneratorOp>>
      registeredKernelFunctionSymbols;

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
  initIfNeeded(MLIRContext *ctx, ArrayRef<std::string> kernelLibraryPaths) {
    // Reuse the library if it exists.
    if (analysis)
      return analysis.get();
    analysis = std::make_unique<MojoLibraryAnalysis>(ctx);
    auto loadError = analysis->init(kernelLibraryPaths);
    if (loadError.isError()) {
      analysis = nullptr;
      return loadError.takeError();
    }
    return analysis.get();
  }

  std::shared_ptr<MojoLibraryAnalysis> analysis;
};

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOLIBRARYANALYSIS_H
