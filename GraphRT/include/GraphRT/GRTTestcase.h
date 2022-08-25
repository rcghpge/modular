//===- GraphRT/GRTTestcase.h ----------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#ifndef GRAPHRT_TOOLS_BEFEXECUTOR_GRTTESTCASE_H
#define GRAPHRT_TOOLS_BEFEXECUTOR_GRTTESTCASE_H

#include "Support/ErrorOr.h"
#include "Support/LLVMForwardDecls.h"
#include "llvm/ADT/StringRef.h"

namespace M {
class BEFInvokable;
class PrimitiveDescriptor;

class GRTTestcase {
public:
  /// Provide a constructor from PrimitiveDescriptor.
  static ErrorOr<GRTTestcase>
  getFromDescriptor(const PrimitiveDescriptor &desc) {
    return GRTTestcase(desc);
  }

  /// Enables dyn_cast/cast/isa on this TopLevelOp by providing the name of this
  /// primitive.
  static StringLiteral getEntityName();

  StringRef getName() const { return name; }

  /// Get the body for this op - a single Invokable.
  BEFInvokable *getBody() const { return body; }

private:
  GRTTestcase(const PrimitiveDescriptor &desc);
  GRTTestcase(StringRef name, BEFInvokable *body) : name(name), body(body) {}
  StringRef name;
  BEFInvokable *body;
};
} // namespace M

#endif // GRAPHRT_TOOLS_BEFEXECUTOR_GRTTESTCASE_H
