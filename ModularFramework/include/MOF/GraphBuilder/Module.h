//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOF_GRAPHBUILDER_MODULE_H
#define MOF_GRAPHBUILDER_MODULE_H

#include "MOF/GraphBuilder/Graph.h"
#include "MOF/GraphBuilder/Loc.h"
#include "MOF/GraphBuilder/Type.h"
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

  const mlir::ModuleOp &getModuleOp() const { return op; }
  mlir::OpBuilder &getBuilder() { return builder; }

  Graph newGraph(Loc loc, llvm::StringRef name, const Arity &inTypes,
                 const Arity &outTypes);

  mlir::Type newDType(DType value);
  mlir::Type newTensorType(mlir::Type elementType, bool ranked,
                           llvm::ArrayRef<int64_t> dims);

  mlir::Type f32(llvm::ArrayRef<int64_t> shape);
  mlir::Type ui(uint64_t width, llvm::ArrayRef<int64_t> shape);
  mlir::Type si(uint64_t width, llvm::ArrayRef<int64_t> shape);
  mlir::Type bool_(llvm::ArrayRef<int64_t> shape);

  mlir::NamedAttribute stringAttr(llvm::StringRef name, llvm::StringRef value);
  // Guaranteed to take ownership of `data` with a dense resource if `owned`.
  // Otherwise, copies `data`; may use dense resource or ArrayElementsAttr.
  mlir::NamedAttribute tensorAttr(llvm::StringRef name,
                                  llvm::ArrayRef<uint8_t> data, mlir::Type t,
                                  bool owned);

  MLIRContext *getContext();

private:
  mlir::ModuleOp op;
  mlir::OpBuilder builder;
};

} // namespace M::MOF::GraphBuilder

#endif // MOF_GRAPHBUILDER_MODULE_H
