//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef MOF_GRAPHBUILDER_MODULE_H
#define MOF_GRAPHBUILDER_MODULE_H

#include "MOF/MOBI/Graph.h"
#include "MOF/MOBI/Loc.h"
#include "MOF/MOBI/Type.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

// TODO: Decide whether the module should be the carrier of the MLIR context.
// An alternative might be to just carry MLIR objects around.

namespace M::MOF::Builder {

class Module {
public:
  Module(mlir::Location loc);

  const mlir::ModuleOp &getModuleOp() const { return op; }
  mlir::OpBuilder &getBuilder() { return builder; }

  Graph newGraph(Loc loc, llvm::StringRef name, const Arity &inTypes,
                 const Arity &outTypes);

  Loc unknownLoc();

  int64_t dynamicDim();

  Type f32(llvm::ArrayRef<int64_t> shape);
  Type ui(uint64_t width, llvm::ArrayRef<int64_t> shape);
  Type si(uint64_t width, llvm::ArrayRef<int64_t> shape);
  Type bool_(llvm::ArrayRef<int64_t> shape);

  Attr stringAttr(llvm::StringRef name, llvm::StringRef value);
  Attr tensorAttr(llvm::StringRef name, llvm::ArrayRef<uint8_t> data, Type t);

  MLIRContext *getContext();

private:
  Attr denseResourceAttr(StringRef name, ArrayRef<char> data, Type t);

  mlir::ModuleOp op;
  mlir::OpBuilder builder;
};

} // namespace M::MOF::Builder

#endif // MOF_GRAPHBUILDER_MODULE_H
