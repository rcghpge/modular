//===----------------------------------------------------------------------===//
//
// This file is Modular Inc proprietary.
//
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGInterfaces.h"
#include "KGEN/KGENDialect/KGENOps.h"
#include "KGEN/MOGGPreElab/MOGGPreElabDecorators.h"

//===----------------------------------------------------------------------===//
// ODS-Generated Definitions
//===----------------------------------------------------------------------===//

#include "GraphCompiler/GraphCompiler/MOGGDialect/MOGGInterfaces.cpp.inc"

namespace M::MOGG {

FailureOr<DictionaryAttr>
buildMojoObjectInfosOutputMethodsDict(Operation *mojoKernelOp,
                                      llvm::ArrayRef<StringRef> methods) {
  auto mojoKernel = cast<KGEN::GeneratorOp>(mojoKernelOp);
  DictionaryAttr valueWitnessesAttr = dyn_cast_or_null<DictionaryAttr>(
      mojoKernel->getAttr(KGEN::MOGGPreElab::MOGG_RESULT_VALUE_WITNESSES));
  if (!valueWitnessesAttr) {
    return mojoKernel.emitError()
           << "no MOGG type value witnesses found (maybe kernel is missing a "
              "register decorator like `@register_internal`)";
  }

  // Ensure all required methods are present.
  for (StringRef methodName : methods) {
    if (!valueWitnessesAttr.contains(methodName)) {
      return mojoKernel.emitError()
             << "kernel output type doesn't implement the method `"
             << methodName << "`. Maybe it's missing a trait implementation.";
    }
  }
  return valueWitnessesAttr;
}

} // namespace M::MOGG
