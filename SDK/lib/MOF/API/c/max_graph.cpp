//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "SDK/MOF/API/c/detail/max_graph.h"
#include "GenericML/GraphCompiler/GraphCompiler.h"
#include "GenericML/GraphCompiler/RMODialect/RMOAttrs.h"
#include "SDK/MOF/API/c/symbol_export.h"
#include "mlir-c/IR.h"
#include "mlir/CAPI/IR.h"
#include "mlir/IR/MLIRContext.h"

using namespace M;
using namespace llvm;
using namespace mlir;

MODULAR_API_EXPORT void MAXG_loadModularDialects(MlirDialectRegistry registry) {
  M::MO::loadDialects(*unwrap(registry));
}

//===----------------------------------------------------------------------===//
// Attribute factories
//===----------------------------------------------------------------------===//

MODULAR_API_EXPORT MlirLocation MAXG_opaqueFrameLocNew(
    MlirContext context, void *frame_ptr, char *(*to_str_fn)(void *),
    void (*incref_fn)(void *), void (*decref_fn)(void *)) {
  mlir::Location loc =
      RMO::PythonFrameAttr::getOpaque(unwrap(context), {.ptr = frame_ptr,
                                                        .to_str = to_str_fn,
                                                        .incref = incref_fn,
                                                        .decref = decref_fn});
  return wrap(loc);
}
