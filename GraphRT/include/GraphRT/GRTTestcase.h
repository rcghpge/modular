//===- GraphRT/Tools/bef-executor/GRTTestcase.h ---------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_TOOLS_BEFEXECUTOR_GRTTESTCASE_H
#define GRAPHRT_TOOLS_BEFEXECUTOR_GRTTESTCASE_H

#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

namespace M {
class Invokable;
class TopLevelEntity;

class GRTTestcase {
public:
  /// Provide a constructor from TopLevelOp.
  GRTTestcase(const TopLevelEntity &op);

  /// Enables dyn_cast/cast/isa on this TopLevelOp by providing the name of this
  /// primitive.
  static StringLiteral getEntityName();

  StringRef getSymbolName() const { return symbolName; }

  /// Get the body for this op - a single Invokable.
  Invokable *getBody() const { return body; }

private:
  GRTTestcase(StringRef symbolName, Invokable *body)
      : symbolName(symbolName), body(body) {}
  StringRef symbolName;
  Invokable *body;
};
} // namespace M

#endif // GRAPHRT_TOOLS_BEFEXECUTOR_GRTTESTCASE_H
