//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOLIBRARYANALYSIS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOLIBRARYANALYSIS_H

#include <string>

#include "Support/AssertStream.h"
#include "mlir/Pass/PassInstrumentation.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/MemoryBuffer.h"

#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/LITDialect/LITOps.h"
#include "Support/SymbolExport.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::MOGG {

/// Strip the "__init__." prefix from a Mojo symbol name if present.
/// Symbols defined in __init__.mojo files should be referenced by their
/// simple name only (e.g., "MyStruct" not "__init__.MyStruct").
inline void stripInitModulePrefix(std::string &name) {
  constexpr StringLiteral initPrefix = "__init__.";
  if (StringRef(name).starts_with(initPrefix))
    name.erase(0, initPrefix.size());
}

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
  ErrorOrSuccess init(ArrayRef<std::string> kernelLibraryPaths = {});

  ErrorOrSuccess addKernelLibraryPath(StringRef kernelLibraryPath);

  /// Get the MLIR context.
  MLIRContext *getContext() { return config.getContext(); }

  /// Looks up a registered Mojo symbol using the name that was used for
  /// registration.
  KGEN::LIT::FnOp getRegisteredLITMojoKernelFor(StringRef name) const;

  /// Returns true if a registered LIT Mojo symbol using the name exists.
  bool hasRegisteredLITMojoKernelFor(StringRef name) const {
    return registeredLITMojoKernelSymbol.contains(name);
  }

  struct LITKernelCandidate {
    LITKernelCandidate(KGEN::LIT::FnOp fun, int64_t priority)
        : fun(fun), priority(priority) {}
    KGEN::LIT::FnOp fun;
    int64_t priority;
  };

  /// Return the mapping.
  const llvm::StringMap<LITKernelCandidate> &
  getRegisteredLITMojoKernelSymbol() const {
    return registeredLITMojoKernelSymbol;
  }

  /// Looks up a registered Mojo shape function using the name that was used
  /// for registration.
  KGEN::LIT::FnOp getRegisteredLITMojoShapeFunctionFor(StringRef name) const;

  /// Looks up a registered Mojo execute function using the name that was used
  /// for registration.
  KGEN::LIT::FnOp getRegisteredLITMojoExecuteFunctionFor(StringRef name) const;

  /// Looks up a registered Mojo update_input_view function using the name that
  /// was used for registration.
  KGEN::LIT::FnOp
  getRegisteredLITMojoUpdateViewFunctionFor(StringRef name) const;

  /// Looks up a registered elementwise function using the name that was used
  /// for registration.
  KGEN::LIT::FnOp
  getRegisteredLITMojoElementwiseFunctionFor(StringRef name) const;

  /// Looks up the Mojo symbol of a registered, labeled kernel function
  /// using the name that was used for registration and the function label.
  KGEN::LIT::FnOp
  getRegisteredLITMojoFunctionFor(StringRef name,
                                  StringRef functionLabel) const;

  KGEN::LIT::FnOp getLITForeachKernel() const {
    ASSERT_STREAM(litForeachKernel != nullptr, "foreach kernel not found");
    return litForeachKernel;
  }
  KGEN::LIT::FnOp getLITViewMaterializeKernel() const {
    ASSERT_STREAM(litViewMaterializeKernel != nullptr, "view kernel not found");
    return litViewMaterializeKernel;
  }

  /// Verify the call to a MO or custom kernel.
  ErrorOrSuccess verifyMOOpMatchesKernel(Operation *op) const;

  /// Verify the parameters attached to the MO custom operation.
  ErrorOrSuccess validateCustomOpParameters(MO::CustomOp customOp) const;

  /// Look up the fully qualified Mojo name for an opaque type.
  /// Given a simple type name like "SIMDPair", returns the fully qualified
  /// name like "MOGGKernelAPITest.SIMDPair". Returns an empty optional if
  /// the type is not found.
  std::optional<StringRef>
  getFullyQualifiedOpaqueTypeName(StringRef simpleTypeName) const;

  static ErrorOr<std::shared_ptr<MojoLibraryAnalysis>>
  fromPaths(MLIRContext *ctx, ArrayRef<std::string> paths);

  /// Returns the paths to built-in kernel packages (MOGGKernelAPI,
  /// MOGGPrimitives) for use in MEF cache key computation. The order is
  /// stable.
  static ErrorOr<SmallVector<std::string>>
  getBuiltInKernelPackagePaths(MLIRContext *ctx);

private:
  /// Helper function to add `func` to the maps of registered kernels if
  /// needed.
  ErrorOrSuccess
  updateRegisteredKernelsMap(KGEN::LIT::FnOp func,
                             bool isGeneratorFromStandardLibrary);

  /// Register opaque type names from all struct declarations in a module.
  /// Warns on conflicts and keeps the first registered type.
  ErrorOrSuccess registerOpaqueTypesFromModule(ModuleOp module);

  ErrorOrSuccess loadAllLITKernels(ArrayRef<std::string> paths);

  /// Hold all .mojopkg (LIT) packages
  OwningOpRef<ModuleOp> moggKernelAPILITPackage;
  SmallVector<OwningOpRef<ModuleOp>> litPackages;

  KGEN::LIT::FnOp litForeachKernel;
  KGEN::LIT::FnOp litViewMaterializeKernel;

  /// Hold the symbol name mapping for registered mojo kernels. With
  /// struct-based mojo kernels, this will map to the `execute` method.
  /// The value holds the kernel as well as its priority.
  llvm::StringMap<LITKernelCandidate> registeredLITMojoKernelSymbol;

  /// Hold the registered, labeled functions that can be attached to a kernel.
  /// For struct-based mojo kernels, this will map to labeled struct member
  /// methods.
  /// The outer map key is the kernel name. The inner map key is the
  /// function label (example: `mogg.shape` will map to the `shape` method).
  llvm::StringMap<llvm::StringMap<KGEN::LIT::FnOp>>
      registeredLITKernelFunctionSymbols;

  /// Maps simple opaque type names (e.g., "SIMDPair") to their fully qualified
  /// Mojo names (e.g., "MOGGKernelAPITest.SIMDPair"). This is populated during
  /// kernel registration by examining the mogg.result_type_name attributes.
  llvm::StringMap<std::string> opaqueTypeNameToFullyQualified;

  mlir::ParserConfig config;
};

/// A Pass mixin which supplements an MLIR pass with an optional
/// MojoLibraryAnalysis.
template <typename PassBaseT>
class MojoLibraryAnalysisPass : public PassBaseT {
public:
  using Base = MojoLibraryAnalysisPass<PassBaseT>;

  MojoLibraryAnalysisPass() = default;

  MojoLibraryAnalysisPass(std::shared_ptr<MojoLibraryAnalysis> analysis)
      : PassBaseT(), mojoLibraryAnalysis(std::move(analysis)), status() {}

  template <typename PassOptions>
  MojoLibraryAnalysisPass(PassOptions args)
      : PassBaseT(std::move(args)), mojoLibraryAnalysis(nullptr), status() {}

  template <typename PassOptions>
  MojoLibraryAnalysisPass(PassOptions args,
                          std::shared_ptr<MojoLibraryAnalysis> analysis)
      : PassBaseT(std::move(args)), mojoLibraryAnalysis(std::move(analysis)),
        status() {}

  MojoLibraryAnalysisPass(const MojoLibraryAnalysisPass &other)
      : PassBaseT(other), mojoLibraryAnalysis(other.mojoLibraryAnalysis),
        status(other.status.isError() ? other.status.copy().takeError()
                                      : ErrorOrSuccess()) {}

  MojoLibraryAnalysisPass(MojoLibraryAnalysisPass &&other) = default;
  ~MojoLibraryAnalysisPass() = default;

  static StringRef getAnalysisName() {
    return llvm::getTypeName<MOGG::MojoLibraryAnalysis>();
  }

  ErrorOr<MOGG::MojoLibraryAnalysis *> getLibraryAnalysis() {
    if (status.isError())
      return status.copy().takeError();

    mlir::TypeID id = mlir::TypeID::get<MOGG::MojoLibraryAnalysis>();

    mlir::PassInstrumentor *pi =
        PassBaseT::getAnalysisManager().getPassInstrumentor();

    if (!mojoLibraryAnalysis) {
      if (pi)
        pi->runBeforeAnalysis(getAnalysisName(), id, this->getOperation());

      auto guard = llvm::scope_exit([&] {
        if (pi)
          pi->runAfterAnalysis(getAnalysisName(), id, this->getOperation());
      });

      auto libraryPaths =
          llvm::to_vector_of<std::string>(this->kernelLibraryPaths);

      MLIRContext &ctx = this->getContext();

      mojoLibraryAnalysis = std::make_shared<MOGG::MojoLibraryAnalysis>(&ctx);
      status = mojoLibraryAnalysis->init(libraryPaths);

      if (status.isError()) {
        mojoLibraryAnalysis = nullptr;
        return status.copy().takeError();
      }
    }

    assert(mojoLibraryAnalysis != nullptr);
    return mojoLibraryAnalysis.get();
  }

protected:
  /// The loaded analysis
  std::shared_ptr<MOGG::MojoLibraryAnalysis> mojoLibraryAnalysis = nullptr;
  ErrorOrSuccess status;
};

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOLIBRARYANALYSIS_H
