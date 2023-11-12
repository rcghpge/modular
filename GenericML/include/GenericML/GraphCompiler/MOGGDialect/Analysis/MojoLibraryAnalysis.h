//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOLIBRARYANALYSIS_H
#define GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOLIBRARYANALYSIS_H

#include <string>

#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"

#include "GenericML/GraphCompiler/MOGGDialect/MojoDecorators.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "mlir/Dialect/Index/IR/IndexOps.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"

//===----------------------------------------------------------------------===//
// Forward Declarations
//===----------------------------------------------------------------------===//

namespace M::MOGG {

/// This analysis loads the standard and extra Mojo libraries and exposes an
/// interface to inquire about their content.
class MojoLibraryAnalysis {
public:
  /// Wrapper around the constructor for error checking purposes. The standard
  /// Mojo kernel library is directly embedded. Only the extra library paths
  /// must be provided.
  static ErrorOr<MojoLibraryAnalysis> get(MLIRContext *ctx,
                                          const std::string &extraLibPaths);
  ~MojoLibraryAnalysis() = default;

  /// Move constructor for MojoLibraryAnalysis::get.
  MojoLibraryAnalysis(MojoLibraryAnalysis &&other) = default;

  // Looks up a registered Mojo symbol using the name that was used for
  // registration.
  inline KGEN::GeneratorOp getRegisteredMojoKernelFor(StringRef name) {
    return getRegisteredMojoFunctionFor(name, registeredMojoKernelSymbol);
  }

  // Looks up a registered Mojo shape function using the name that was used for
  // registration.
  inline KGEN::GeneratorOp getRegisteredMojoShapeFunctionFor(StringRef name) {
    return getRegisteredMojoFunctionFor(name, registeredShapeFunctionSymbol);
  }

  // Looks up and returns a mojo function as is.
  inline KGEN::GeneratorOp getMojoFunctionNamed(StringRef name) {
    return cast_or_null<KGEN::GeneratorOp>(
        mlir::SymbolTable(*mojoBindings).lookup(name));
  }

  // When we compile we only want to compile the functions we add so the first
  // step is to unmark all the other externals.
  void markAllFunctionsAsNotExported() {
    for (auto func : mojoBindings->getOps<KGEN::FuncOp>())
      func.setNotExported();

    for (auto func : mojoBindings->getOps<KGEN::GeneratorOp>())
      func.setNotExported();
  }

  // Return true if the provided kernel or shape function has the input
  // parameter named `parameterName`.
  bool kernelHasInputParameter(KGEN::GeneratorOp kernel,
                               StringRef parameterName);

  // Returns true if the provided kernel or shape function has the decorator
  // `decoratorName`.
  bool kernelHasDecorator(KGEN::GeneratorOp kernel, StringRef decoratorName);

  // Returns true if the kernel is an elementwise kernel.
  inline bool kernelIsElementwise(KGEN::GeneratorOp kernel) {
    return kernelHasDecorator(kernel, DECORATOR_ELEMENTWISE);
  }

  // Returns true if the kernel is expecting to take indices as last arg.
  inline bool kernelIsViewOp(KGEN::GeneratorOp kernel) {
    return kernelHasDecorator(kernel, DECORATOR_VIEW);
  }

  // Returns true if the kernel is a view operation.
  inline bool kernelTakesIndices(KGEN::GeneratorOp kernel) {
    return kernelHasDecorator(kernel, DECORATOR_TAKES_INDICES);
  }

  // Returns true if this is one of the new experimental kgen-in-mogg kernels.
  bool kernelIsExperimentalKGEN(KGEN::GeneratorOp kernel) {
    return kernelHasDecorator(kernel, DECORATOR_EXPERIMENTAL);
  }

  /// This ModuleOp is free standing meaning it is not embedded into the IR. It
  /// contains all the bindings (KGEN) that were pulled from the libraries.
  OwningOpRef<mlir::ModuleOp> mojoBindings;

private:
  MojoLibraryAnalysis(
      llvm::StringMap<std::string> &registeredMojoKernelSymbol,
      llvm::StringMap<std::string> &registeredShapeFunctionSymbol,
      OwningOpRef<mlir::ModuleOp> &mojoBindings);

  KGEN::GeneratorOp
  getRegisteredMojoFunctionFor(StringRef name,
                               llvm::StringMap<std::string> &mapping);

  llvm::StringMap<std::string> registeredMojoKernelSymbol;
  llvm::StringMap<std::string> registeredShapeFunctionSymbol;
};

//===----------------------------------------------------------------------===//
// Generated Pass Classes and Registration
//===----------------------------------------------------------------------===//

} // namespace M::MOGG

#endif // GENERICML_GRAPHCOMPILER_MOGGDIALECT_MOJOLIBRARYANALYSIS_H
