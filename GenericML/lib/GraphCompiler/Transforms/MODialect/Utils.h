//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//
//
// Utilities common to MO passes.
//
//===----------------------------------------------------------------------===//

#include "Support/LLVMCompilerForwardDecls.h"
#include "llvm/Support/raw_os_ostream.h"

#ifndef GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_UTILS_H
#define GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_UTILS_H

namespace M::MO {

inline void printValue(llvm::raw_ostream &os, Value value) {
  value.printAsOperand(os, mlir::OpPrintingFlags().printGenericOpForm());
}

inline void printOp(llvm::raw_ostream &os, Operation *op) {
  op->getName().print(os);
}

inline void printGraph(llvm::raw_ostream &os, GraphOp graphOp) {
  graphOp.print(os, mlir::OpPrintingFlags().printGenericOpForm());
  os << "\n";
}

} // namespace M::MO

#endif // GENERICML_GRAPHCOMPILER_TRANSFORMS_MODIALECT_UTILS_H
