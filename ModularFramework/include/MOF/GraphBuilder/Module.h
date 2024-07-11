//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOF_GRAPHBUILDER_MODULE_H
#define MOF_GRAPHBUILDER_MODULE_H

#include "AsyncRT/Runtime/Runtime.h"
#include "GenericML/GraphCompiler/MODialect/MOOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

// TODO: Decide whether the module should be the carrier of the MLIR context.
// An alternative might be to just carry MLIR objects around.

namespace M::MOF::GraphBuilder {

class Module {
public:
  Module(mlir::Location loc);
  Module(mlir::ModuleOp op);
  Module(mlir::ModuleOp op, mlir::OpBuilder builder)
      : op(op), builder(builder) {}

  mlir::ModuleOp getModuleOp() const { return op; }
  mlir::OpBuilder &getBuilder() { return builder; }

  mlir::Operation *newGraph(mlir::Location loc, llvm::StringRef name,
                            mlir::FunctionType signature);

  MLIRContext *getContext();

private:
  mlir::ModuleOp op;
  mlir::OpBuilder builder;
};

// Guaranteed to take ownership of `data` with a dense resource if `owned`.
// Otherwise, copies `data`; may use dense resource or ArrayElementsAttr.
mlir::NamedAttribute tensorAttr(llvm::StringRef name,
                                llvm::ArrayRef<uint8_t> data, mlir::Type t,
                                bool owned, AsyncRT::Runtime &runtime);

} // namespace M::MOF::GraphBuilder

#endif // MOF_GRAPHBUILDER_MODULE_H
